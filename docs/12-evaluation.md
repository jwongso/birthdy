# Doc 12 - Model Evaluation and Benchmarking

## Purpose

Establish a repeatable, standardized measurement framework for comparing models
before fine-tuning, verifying fine-tuning improved performance, and demonstrating
quality across different models and fine-tuning runs.

Three levels of measurement, each serving a different purpose.

---

## Level 1 - Speed (tokens/sec)

Objective, hardware-specific, run on every candidate model before anything else.
If a model cannot meet the minimum speed threshold, do not fine-tune it.

**Tool: llama-bench (built into llama.cpp)**

```bash
cd ~/proj/priv/llama.cpp/build/bin

# Basic benchmark
./llama-bench -m /path/to/model.gguf -ngl 999 -n 512 -p 512

# Full benchmark with different batch sizes
./llama-bench -m /path/to/model.gguf -ngl 999 \
  -p 128,256,512 \
  -n 128,256,512 \
  -r 3
```

**What the output means:**

```
| model          | size   | params | backend | ngl | test   | t/s         |
|----------------|--------|--------|---------|-----|--------|-------------|
| qwen3 8B Q4_KM | 4.68G  | 8.19B  | CUDA    | 999 | pp 512 | 2847 +/- 10 |
| qwen3 8B Q4_KM | 4.68G  | 8.19B  | CUDA    | 999 | tg 128 |   48 +/-  1 |
```

- `pp` = prompt processing (prefill) - how fast it reads your input
- `tg` = token generation - how fast it writes the reply (what users feel)

**Minimum thresholds for deployment:**

| Use case | Minimum tg tok/s | Target tg tok/s |
|----------|-----------------|-----------------|
| Chat interface | 15 | 35+ |
| Document analysis (batch) | 5 | 15+ |
| Real-time voice (future) | 50 | 80+ |

**On RTX 4060 Laptop (8GB VRAM), expected results:**

| Model | Quantization | Expected tg tok/s |
|-------|-------------|------------------|
| Qwen3-8B | Q4_K_M | 35-45 |
| Qwen3-8B | Q5_K_M | 28-35 |
| Qwen3-8B | Q8_0 | 18-22 |
| Llama-3.1-8B | Q4_K_M | 38-48 |
| Phi-4-mini | Q4_K_M | 55-70 |

**On Strix Halo Node 1 (128GB unified memory, future):**

| Model | Quantization | Expected tg tok/s |
|-------|-------------|------------------|
| Qwen3-8B | Q4_K_M | 80-120 |
| Qwen3-72B | Q4_K_M | 25-40 |
| Llama-3.1-70B | Q4_K_M | 20-35 |

---

## Level 2 - General capability benchmarks

Standardized, comparable across models, run once per candidate model.

**Tool: Eleuther AI LM Evaluation Harness**

```bash
pip install lm-eval

# Start llama-server first, then:
lm_eval --model local-chat-completions \
  --tasks mmlu_professional_law,mmlu_jurisprudence \
  --model_args base_url=http://localhost:8080/v1,model=local \
  --num_fewshot 5 \
  --output_path ./eval_results/
```

**Recommended task set for evaluation:**

```bash
lm_eval --model local-chat-completions \
  --tasks mmlu_professional_law,mmlu_jurisprudence,mmlu_professional_accounting,hellaswag,truthfulqa_mc1 \
  --model_args base_url=http://localhost:8080/v1,model=local \
  --num_fewshot 5 \
  --output_path ./eval_results/
```

**What each task measures:**

| Task | What it tests | Relevance to domain work |
|------|--------------|--------------------------|
| mmlu_professional_law | Legal knowledge breadth | Core domain |
| mmlu_jurisprudence | Legal reasoning and philosophy | Core domain |
| mmlu_professional_accounting | Accounting/tax knowledge | Secondary domain |
| hellaswag | Common sense reasoning | General capability |
| truthfulqa_mc1 | Tendency to hallucinate | Critical - legal advice must be accurate |

**Reference scores (pre-fine-tuning baselines):**

| Model | mmlu_law | truthfulqa | hellaswag |
|-------|----------|------------|-----------|
| Qwen3-8B | ~73% | ~68% | ~82% |
| Llama-3.1-8B | ~70% | ~65% | ~81% |
| Phi-4-mini | ~68% | ~70% | ~78% |

After fine-tuning, mmlu_law should increase. If truthfulqa drops significantly,
the model is hallucinating more - a red flag for legal deployment.

**LegalBench (specialized legal reasoning):**

```bash
lm_eval --model local-chat-completions \
  --tasks legalbench_contract_qa,legalbench_statutory_reasoning \
  --model_args base_url=http://localhost:8080/v1,model=local \
  --num_fewshot 3 \
  --output_path ./eval_results/
```

LegalBench has 162 tasks. For legal domain evaluation, focus on:
- `contract_qa` - contract clause understanding
- `statutory_reasoning` - applying statutes to fact patterns
- `issue_spotting` - identifying legal issues in a scenario

---

## Level 3 - Domain-specific benchmark

No standardized NZ legal benchmark exists. Building one gives you a repeatable,
objective quality signal that standardized benchmarks cannot provide.

### Structure

A benchmark question has four components:

```json
{
  "id": "nz-property-001",
  "domain": "property",
  "subdomain": "adverse_possession",
  "difficulty": "intermediate",
  "question": "What is the limitation period for adverse possession of registered land under the Limitation Act 2010?",
  "reference_answer": "Under the Limitation Act 2010, there is no specific limitation period for adverse possession of registered Torrens title land in NZ. The Land Transfer Act 2017 effectively makes adverse possession of registered land extremely difficult - a possessor cannot gain title through adverse possession alone. For unregistered land, the Limitation Act 2010 section 11 sets a 6-year limitation period for actions to recover land.",
  "key_facts": [
    "Limitation Act 2010",
    "Land Transfer Act 2017",
    "Torrens title protection",
    "6-year period for unregistered land"
  ],
  "common_errors": [
    "Citing Land Transfer Act 1989 (repealed)",
    "Stating 5 or 12 year periods from old law",
    "Treating registered and unregistered land the same"
  ]
}
```

### Initial question set (build this incrementally)

**NZ Property Law (20 questions)**
- Adverse possession requirements (registered vs unregistered)
- Freehold vs leasehold vs unit title vs Maori freehold
- Easements: express, implied, prescriptive
- Land Transfer Act 2017 key provisions
- LINZ processes: title search, caveat, transfer
- Ground rent review mechanisms
- Resource Management Act (RMA) and property development
- Building Act 2004 and consent requirements
- Common property in unit title developments

**NZ Employment Law (15 questions)**
- Personal grievance process under Employment Relations Act 2000
- 90-day trial period rules
- Constructive dismissal definition and threshold
- Redundancy requirements
- Collective bargaining obligations
- Holiday Act 2003 leave calculations (notoriously complex)

**NZ Tax (10 questions)**
- Bright-line test for residential property (current 2-year rule post-2024)
- GST registration threshold and obligations
- Provisional tax calculation methods
- IRD's income tax treatment of trusts

### Scoring with Claude as judge

Rather than manual scoring, use Claude to evaluate each answer:

```python
import anthropic
import json

client = anthropic.Anthropic()

JUDGE_PROMPT = """You are evaluating an AI model's answer to a New Zealand legal question.

Question: {question}
Reference answer: {reference_answer}
Key facts that must be present: {key_facts}
Common errors to watch for: {common_errors}

Model answer: {model_answer}

Score the model answer from 1-5:
1 - Factually wrong, dangerous to rely on
2 - Mostly wrong, missing key facts, cites wrong legislation  
3 - Partially correct, captures general concept but wrong on specifics
4 - Mostly correct, minor gaps or imprecision
5 - Accurate, cites correct current legislation, covers key nuances

Respond in JSON: {{"score": N, "reasoning": "...", "errors_found": [...], "missing_facts": [...]}}"""


async def judge_answer(question_data: dict, model_answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question_data["question"],
        reference_answer=question_data["reference_answer"],
        key_facts=", ".join(question_data["key_facts"]),
        common_errors=", ".join(question_data["common_errors"]),
        model_answer=model_answer,
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text)
```

### Running a full benchmark evaluation

```python
import asyncio
import aiohttp
import json
from pathlib import Path


async def query_model(question: str, base_url: str = "http://localhost:8080") -> str:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload
        ) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def run_benchmark(questions_file: str, output_file: str):
    questions = json.loads(Path(questions_file).read_text())
    results = []

    for q in questions:
        print(f"Testing: {q['id']}")
        model_answer = await query_model(q["question"])
        judgment = await judge_answer(q, model_answer)
        results.append({
            "id": q["id"],
            "domain": q["domain"],
            "difficulty": q["difficulty"],
            "score": judgment["score"],
            "reasoning": judgment["reasoning"],
            "errors_found": judgment["errors_found"],
            "model_answer": model_answer,
        })
        print(f"  Score: {judgment['score']}/5 - {judgment['reasoning'][:80]}")

    avg = sum(r["score"] for r in results) / len(results)
    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r["score"])

    summary = {
        "overall_avg": round(avg, 2),
        "by_domain": {d: round(sum(s)/len(s), 2) for d, s in by_domain.items()},
        "results": results,
    }
    Path(output_file).write_text(json.dumps(summary, indent=2))
    print(f"\nOverall average: {avg:.2f}/5")
    print("By domain:", summary["by_domain"])


asyncio.run(run_benchmark("nz_benchmark.json", "results_qwen3_baseline.json"))
```

### Tracking results over time

```
results/
  qwen3-8b_baseline_2026-05.json       <- before fine-tuning
  qwen3-8b_ft-v1_2026-07.json         <- after first fine-tune
  qwen3-8b_ft-v2_2026-10.json         <- after second fine-tune
  llama31-8b_baseline_2026-05.json    <- competitor baseline
  phi4mini_baseline_2026-05.json      <- lightweight option baseline
```

Compare with a simple script:

```bash
python3 compare_results.py results/qwen3-8b_baseline_2026-05.json \
                           results/qwen3-8b_ft-v1_2026-07.json
```

---

## RAGAS - RAG pipeline quality

Once the RAG pipeline is running (Doc 11), measure retrieval quality separately
from generation quality using RAGAS:

```bash
pip install ragas
```

RAGAS measures four things:

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Faithfulness | Does the answer stick to retrieved context? | >0.85 |
| Answer relevance | Does the answer address the question? | >0.80 |
| Context precision | Is the retrieved context relevant? | >0.75 |
| Context recall | Did retrieval find all relevant chunks? | >0.70 |

Low faithfulness = model is hallucinating beyond the documents (dangerous for legal).
Low context recall = Qdrant is missing relevant chunks (chunking strategy problem).

---

## Benchmark maintenance

The NZ benchmark must be updated when:
- New legislation passes (update reference answers citing new act)
- Landmark court decisions change interpretation
- IRD or LINZ policy changes
- A question is found to be ambiguous or wrong

Treat the benchmark like source code - version controlled, reviewed before changes,
changelog maintained.

Quarterly benchmark updates that reflect current NZ law drive the retraining cycle.
