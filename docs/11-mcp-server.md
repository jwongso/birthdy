# Doc 11 - MCP Server: Giving the Model Tools

## What is MCP?

MCP (Model Context Protocol) is an open standard that lets a language model call
external tools during a conversation. Instead of the model answering purely from
its weights or retrieved context, it can invoke functions - search a database,
fetch a web page, run a calculation, call an API - and reason over the results.

```
User: "What is the current OCR rate set by the Reserve Bank?"

Without MCP:
  Model answers from training data -> possibly stale, possibly wrong

With MCP:
  Model calls fetch_rbnz_ocr() tool
  Tool fetches current rate from RBNZ website
  Model reasons over the live result -> always current, always accurate
```

MCP is what separates a model that knows things from a model that can do things.

---

## How MCP works

The MCP server is a separate process that exposes a set of tool definitions to
the model. Each tool has:

- A name
- A description (the model reads this to decide when to call it)
- A parameter schema
- An implementation that runs when called

The conversation flow:

```
1. User sends message
2. LLM sees available tools in its context
3. LLM decides to call a tool (generates a structured tool call)
4. MCP server receives the call, runs the implementation
5. Result is injected back into the conversation
6. LLM reasons over the result and responds
```

The model never executes code directly - it requests a tool call, the server
executes it, the result comes back as context.

---

## MCP server in Go

Go is the recommended language for the MCP server for this stack. Reasons:

- Single static binary - easy to deploy alongside llama-server
- Fast startup, low memory overhead
- Strong standard library for HTTP, JSON, subprocess management
- Easy to cross-compile for client hardware

### Project structure

```
mcp-server/
  main.go           - entry point, tool registry, HTTP server
  tools/
    search.go       - Qdrant semantic search tool
    fetch.go        - web page fetcher
    legislation.go  - NZ legislation lookup
    calculator.go   - date/deadline calculations
  schema/
    types.go        - shared request/response types
```

### Tool definition

```go
type Tool struct {
    Name        string          `json:"name"`
    Description string          `json:"description"`
    InputSchema json.RawMessage `json:"inputSchema"`
}

type ToolCall struct {
    Name      string          `json:"name"`
    Arguments json.RawMessage `json:"arguments"`
}

type ToolResult struct {
    Content string `json:"content"`
    IsError bool   `json:"isError"`
}
```

### Example tool - NZ legislation lookup

```go
// tools/legislation.go

package tools

import (
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "net/url"
)

type LegislationArgs struct {
    Act     string `json:"act"`
    Section string `json:"section,omitempty"`
}

// FetchLegislation fetches current statute text from legislation.govt.nz
func FetchLegislation(rawArgs json.RawMessage) (string, error) {
    var args LegislationArgs
    if err := json.Unmarshal(rawArgs, &args); err != nil {
        return "", fmt.Errorf("invalid arguments: %w", err)
    }

    // legislation.govt.nz has a public API for statute text
    query := url.QueryEscape(args.Act)
    endpoint := fmt.Sprintf("https://api.legislation.govt.nz/v1/act/public/%s", query)

    resp, err := http.Get(endpoint)
    if err != nil {
        return "", fmt.Errorf("fetch failed: %w", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", err
    }

    if args.Section != "" {
        // filter to relevant section - implementation depends on API response format
        return extractSection(string(body), args.Section), nil
    }

    return string(body), nil
}
```

### Example tool - Qdrant semantic search

```go
// tools/search.go

package tools

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

type SearchArgs struct {
    Query     string `json:"query"`
    Limit     int    `json:"limit"`
    SessionID string `json:"session_id,omitempty"`
}

type QdrantSearchRequest struct {
    Vector  []float32 `json:"vector"`
    Limit   int       `json:"limit"`
    WithPayload bool  `json:"with_payload"`
}

func SemanticSearch(rawArgs json.RawMessage, qdrantURL string, embedFn func(string) ([]float32, error)) (string, error) {
    var args SearchArgs
    if err := json.Unmarshal(rawArgs, &args); err != nil {
        return "", fmt.Errorf("invalid arguments: %w", err)
    }

    if args.Limit == 0 {
        args.Limit = 5
    }

    vector, err := embedFn(args.Query)
    if err != nil {
        return "", fmt.Errorf("embedding failed: %w", err)
    }

    reqBody, _ := json.Marshal(QdrantSearchRequest{
        Vector:      vector,
        Limit:       args.Limit,
        WithPayload: true,
    })

    resp, err := http.Post(
        fmt.Sprintf("%s/collections/birthdy_memory/points/search", qdrantURL),
        "application/json",
        bytes.NewReader(reqBody),
    )
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    return string(body), nil
}
```

### Main server

```go
// main.go

package main

import (
    "encoding/json"
    "log"
    "net/http"
    "os"

    "mcp-server/tools"
)

var toolRegistry = []Tool{
    {
        Name:        "fetch_legislation",
        Description: "Fetch current NZ statute text from legislation.govt.nz. Use this when asked about specific laws, acts, or regulations to get current wording rather than relying on training knowledge.",
        InputSchema: json.RawMessage(`{
            "type": "object",
            "properties": {
                "act":     {"type": "string", "description": "Name of the act, e.g. Land Transfer Act 2017"},
                "section": {"type": "string", "description": "Specific section number, e.g. 139"}
            },
            "required": ["act"]
        }`),
    },
    {
        Name:        "semantic_search",
        Description: "Search past conversation memory and indexed documents by meaning. Use when the user references something from a previous session or asks about a topic that may have been discussed before.",
        InputSchema: json.RawMessage(`{
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query in natural language"},
                "limit": {"type": "integer", "description": "Number of results to return, default 5"}
            },
            "required": ["query"]
        }`),
    },
    {
        Name:        "calculate_deadline",
        Description: "Calculate legal deadlines from a given date. Use for limitation periods, filing deadlines, notice periods.",
        InputSchema: json.RawMessage(`{
            "type": "object",
            "properties": {
                "from_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                "days":      {"type": "integer", "description": "Number of days to add"},
                "exclude_weekends": {"type": "boolean", "description": "Whether to skip weekends"}
            },
            "required": ["from_date", "days"]
        }`),
    },
}

func handleTools(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(toolRegistry)
}

func handleCall(w http.ResponseWriter, r *http.Request) {
    var call ToolCall
    if err := json.NewDecoder(r.Body).Decode(&call); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    var result ToolResult
    var err error

    switch call.Name {
    case "fetch_legislation":
        result.Content, err = tools.FetchLegislation(call.Arguments)
    case "semantic_search":
        embedFn := tools.OllamaEmbedder(os.Getenv("OLLAMA_URL"), os.Getenv("EMBED_MODEL"))
        result.Content, err = tools.SemanticSearch(call.Arguments, os.Getenv("QDRANT_URL"), embedFn)
    case "calculate_deadline":
        result.Content, err = tools.CalculateDeadline(call.Arguments)
    default:
        result.Content = "unknown tool"
        result.IsError = true
    }

    if err != nil {
        result.Content = err.Error()
        result.IsError = true
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(result)
}

func main() {
    http.HandleFunc("/tools", handleTools)
    http.HandleFunc("/call", handleCall)

    addr := ":8090"
    log.Printf("MCP server listening on %s", addr)
    log.Fatal(http.ListenAndServe(addr, nil))
}
```

---

## Connecting MCP to Birthdy

The Birthdy bot needs to be aware of available tools and inject them into the
system prompt or as a tool list depending on what llama-server supports.

For llama-server with OpenAI-compatible `/v1/chat/completions`, tools are passed
in the request body:

```python
# in llama_client.py

async def chat(self, messages: list[dict], system: str = "", tools: list[dict] = None) -> str:
    payload = {
        "model": "local",
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.url}/v1/chat/completions",
            json=payload,
        ) as resp:
            data = await resp.json()

    choice = data["choices"][0]

    # handle tool call response
    if choice["finish_reason"] == "tool_calls":
        return await self._execute_tool_calls(choice["message"]["tool_calls"])

    return choice["message"]["content"]

async def _execute_tool_calls(self, tool_calls: list) -> str:
    results = []
    for call in tool_calls:
        resp = await aiohttp.ClientSession().post(
            "http://localhost:8090/call",
            json={"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
        )
        result = await resp.json()
        results.append(result["content"])
    return "\n\n".join(results)
```

---

## Tool design principles

**Write descriptions for the model, not for humans.**

The model reads the description to decide whether to call the tool. Be specific
about when it should be used, not just what it does.

Bad:
```
"description": "Fetches legislation text"
```

Good:
```
"description": "Fetch current NZ statute text from legislation.govt.nz.
Use this when asked about specific laws, acts, or regulations to get
current wording rather than relying on training knowledge which may
reference outdated versions."
```

**Keep tools focused.**

One tool does one thing. A tool that fetches legislation AND searches Qdrant AND
calculates deadlines is hard for the model to reason about. Three separate tools
with clear names are better.

**Always handle errors gracefully.**

If a tool fails (network down, API rate limit, bad input), return a clear error
message rather than crashing. The model can then tell the user what went wrong
rather than producing a confusing response.

**Log every tool call.**

Tool calls are the most debuggable part of the system. Log the tool name,
arguments, and result for every call. When the model behaves unexpectedly, the
tool call log tells you exactly what it was working with.

---

## Running as a systemd user service

```ini
# ~/.config/systemd/user/mcp-server.service

[Unit]
Description=Birthdy MCP tool server
After=network.target qdrant.service
Requires=qdrant.service

[Service]
Type=simple
WorkingDirectory=/home/wdha/proj/priv/mcp-server
ExecStart=/home/wdha/proj/priv/mcp-server/mcp-server
Restart=on-failure
RestartSec=5
EnvironmentFile=/home/wdha/proj/priv/birthdy/.env

[Install]
WantedBy=default.target
```

Then update `birthdy.service` to depend on it:

```ini
After=network-online.target llama-server.service qdrant.service mcp-server.service
Requires=llama-server.service qdrant.service mcp-server.service
```

---

## What tools to build first

For a general personal assistant (Birthdy use case):

| Tool | Value | Effort |
|---|---|---|
| Web search | Answers questions about current events | Low - wrap a search API |
| Web page fetch | Read any URL the user pastes | Low - HTTP GET + HTML strip |
| Semantic memory search | Recall past conversations | Already built in Qdrant |
| Calculator | Avoid arithmetic errors | Trivial |
| Date/deadline calculator | Useful for any planning | Low |

For a domain deployment (legal, medical, accounting):

| Tool | Value | Effort |
|---|---|---|
| Legislation lookup | Current statute text always fresh | Medium |
| Case law search | Retrieve court decisions | Medium - depends on data source |
| Document Q&A | RAG over client's own files | Already built with Qdrant |
| Xero / accounting API | Live financial data | Medium per integration |
| Calendar integration | Deadlines, appointments | Low |
