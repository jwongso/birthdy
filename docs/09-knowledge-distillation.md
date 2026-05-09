# Doc 09 - Knowledge Distillation: Teaching a Small Model with a Large One

## What is knowledge distillation?

Knowledge distillation is the process of using a large, capable model (the
**teacher**) to generate high-quality training data that a smaller model (the
**student**) can learn from.

The intuition: GPT-4 or Claude cannot run on your laptop. But if you can get
Claude to write 10,000 examples of exactly the kind of reasoning you want, you
can fine-tune a small local model to reproduce that reasoning style on your
specific domain.

```
Teacher (Claude API)
  -> generates high-quality Q&A pairs, reasoning chains, explanations
  -> on YOUR domain (NZ law, medical notes, accounting, etc.)
  -> reviewed and corrected by a human expert
  -> saved as JSONL fine-tuning dataset

Student (Qwen3-8B local)
  -> LoRA fine-tuned on that dataset
  -> learns the teacher's reasoning style for this specific domain
  -> runs fully locally, no API cost, no data leaving the network
```

This is how you get a cheap local model to reason like an expert on a narrow
domain without training from scratch (which requires billions of examples and
enormous compute).

---

## Why not just use Claude directly?

For a one-person internal tool, Claude API works fine. For a client deployment,
it breaks down:

| Concern | Claude API | Local fine-tuned model |
|---|---|---|
| Data privacy | All prompts leave the network | Stays on-premises |
| Cost at scale | $5,000-50,000+/month heavy usage | Electricity only |
| Latency | 1-5 seconds per response | 35-45 tok/s locally |
| Offline operation | Requires internet | Works air-gapped |
| Customisation | Prompt engineering only | Weights modified |
| Compliance | Cloud jurisdiction issues | On-prem, auditable |

Knowledge distillation lets you use Claude's quality during training and then
never touch the API again in production.

---

## The distillation pipeline

### Step 1 - Identify what the model needs to know

Before generating anything, answer these questions:

- What are the top 50 questions staff ask every day?
- What does "good" look like for each question? (defines your evaluation set)
- What documents contain the ground truth? (legislation, policies, contracts)
- What failure modes exist in the base model? (run the evaluation from Doc 12)

For NZ legal domain the base model (Qwen3-8B) scored 3-4/5 because it:
- Referenced Land Transfer Act 1989 (repealed by LTA 2017)
- Missed NZ-specific concepts (Maori freehold, unit title, ground rent)
- Knew general legal reasoning but not NZ-specific application

These gaps define what the distillation dataset must fix.

### Step 2 - Build the document corpus

Index all relevant source documents into Qdrant. These become the factual
grounding for every generated example - the teacher reads them before writing
each Q&A pair.

```
Source documents -> chunk (500-800 tokens) -> embed (nomic-embed-text)
                                           -> store in Qdrant
```

For a law firm this means: Land Transfer Act 2017, Property Law Act 2007,
Limitation Act 2010, relevant court decisions, firm's internal precedents.

### Step 3 - Generate Q&A pairs with the teacher

Ask Claude to generate examples in the format your fine-tuning framework expects.
Each example has three parts:

- **system** - the persona and task definition
- **user** - a realistic question a staff member would ask
- **assistant** - the ideal response, grounded in retrieved documents

```python
import anthropic
import json

client = anthropic.Anthropic()

def generate_example(question: str, retrieved_docs: str) -> dict:
    prompt = f"""You are an expert NZ property lawyer. Using only the provided
documents, answer the following question with precise legal reasoning.
Cite the specific statute section or case name for every claim.
If the documents do not contain enough information to answer, say so.

Documents:
{retrieved_docs}

Question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "messages": [
            {"role": "system", "content": "You are an expert NZ property lawyer."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response.content[0].text}
        ]
    }
```

Generate 500-2000 examples across the full range of question types. Diversity
matters more than volume - 500 varied examples beats 2000 variations of the
same question type.

### Step 4 - Generate chain-of-thought examples

For complex reasoning tasks, plain Q&A is not enough. You need the model to
learn HOW to reason, not just what the answer is. Generate examples where the
teacher shows its reasoning step by step.

```python
cot_prompt = f"""You are an expert NZ property lawyer. A client has asked:

"{question}"

Think through this step by step:
1. What legal area does this question touch?
2. What is the current governing statute (post-2017)?
3. What are the key elements the client needs to satisfy?
4. Are there any NZ-specific nuances (Maori land, unit title, etc.)?
5. What is the practical answer for the client?

Show your reasoning explicitly before giving the final answer."""
```

Chain-of-thought examples teach the model to reason rather than pattern-match.
This is what makes the fine-tuned model generalise to new questions rather than
just memorising the training set.

### Step 5 - Human review and correction

This step is non-negotiable. Claude makes mistakes, especially on domain-specific
edge cases. Every generated example must be reviewed by a subject matter expert
before it enters the training set.

Three categories after review:

- **Accept as-is** - correct, well-reasoned, good citation
- **Edit** - right direction but needs correction (wrong section number, missing
  nuance) - fix it, keep it
- **Reject** - wrong answer or hallucinated statute - discard entirely

A dataset with 500 reviewed examples beats one with 2000 unreviewed examples
every time. Bad training data teaches bad reasoning.

### Step 6 - Format as JSONL

The final dataset is a JSONL file - one JSON object per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Split into train (90%) and validation (10%) sets before fine-tuning. The
validation set tells you whether the model is learning to generalise or just
memorising.

---

## Dataset size guidelines

| Domain complexity | Minimum examples | Target examples |
|---|---|---|
| Narrow FAQ (30-50 question types) | 200-300 | 500 |
| Moderate domain (property law) | 500-800 | 1,500 |
| Broad domain (general NZ law) | 2,000+ | 5,000+ |
| Reasoning style only (no new facts) | 100-200 | 300 |

For Birthdy's personality fine-tuning: 100-200 examples of the desired
conversation style is enough. Personality is much easier to teach than domain
knowledge.

---

## What distillation teaches vs what RAG handles

After distillation, the fine-tuned model knows:

- **Domain vocabulary** - "Torrens title", "indefeasibility", "caveat lodgement"
  no longer need to be explained by retrieved context
- **Reasoning style** - how to structure a legal analysis, what elements matter,
  what to check first
- **NZ-specific defaults** - Land Transfer Act 2017 not 1989, unit title regime,
  Maori Land Court jurisdiction
- **When to say "I don't know"** - trained on examples where the teacher
  acknowledged uncertainty rather than hallucinating

RAG still handles:

- **Specific statute text** - exact wording of sections retrieved at query time
- **Current case law** - court decisions indexed into Qdrant, retrieved fresh
- **Client-specific documents** - contracts, precedents, internal policies

The combination: fine-tuning gives the model the right mental model for the
domain, RAG gives it the current facts. Neither alone is as good as both together.

---

## Avoiding catastrophic forgetting

The student model (Qwen3-8B) already knows how to code, reason mathematically,
write clearly, and handle general knowledge. Fine-tuning on domain data can
erode these general capabilities if not done carefully.

Two protective measures:

**1. Include general examples in the training mix**

Do not train on domain data alone. Mix in 10-20% general examples (coding,
math, writing) to remind the model it should still be able to do those things.

**2. Use LoRA, not full fine-tuning**

LoRA (Low-Rank Adaptation) modifies only a small set of adapter weights rather
than the full model. The base model weights stay frozen. This is the primary
reason LoRA is used for domain adaptation - it adds domain capability without
overwriting general capability.

Full fine-tuning on 500 domain examples would destroy the base model's general
reasoning. LoRA on 500 domain examples improves domain reasoning while
preserving the rest.

The next doc (Doc 10) covers LoRA in detail.

---

## Iterative distillation

One round of distillation is rarely enough. The process is iterative:

```
Round 1: Generate examples from base model gaps
         -> fine-tune -> evaluate -> find new gaps

Round 2: Generate examples targeting round 1 gaps
         -> fine-tune -> evaluate -> find remaining gaps

Round 3: Focus on edge cases and failure modes
         -> fine-tune -> evaluate
```

Each round the model gets closer to the target. Three rounds of 500 examples
each (1,500 total, all reviewed) typically reaches production quality for a
narrow domain.

The evaluation benchmark (Doc 12) is what tells you when to stop. When the
model hits your quality threshold on the held-out test set, the distillation
is done.

---

## Cost estimate

For a client engagement targeting NZ property law:

| Item | Estimate |
|---|---|
| Claude API for generation (1,500 examples, avg 800 tokens output) | ~$18 USD |
| Human review time (domain expert, 1,500 examples at 2 min each) | 50 hours |
| LoRA training on Node 2 (128GB, 3 rounds) | ~6 hours compute |

The Claude API cost is negligible. The human review time is the real cost -
and it is also what makes the dataset valuable. A law firm that spends 50 hours
curating their training data has created a proprietary asset that no competitor
can replicate without the same effort.

---

## The real cost and the real value

**What is actually hard to replicate:**

The compute is not the bottleneck. Running LoRA fine-tuning on a pre-trained base
model is cheap - hours on a single machine. Any organisation with the same dataset
can reproduce the same fine-tuned model.

The value is the **curated dataset itself**. To reproduce it, you would need to:

- Identify the same gaps in the base model
- Generate covering examples using Claude or equivalent
- Find a domain expert willing to spend 50 hours reviewing and correcting them
- Encode the same organisation-specific interpretation of edge cases and local nuance

That 50 hours of expert review cannot be automated away. It encodes the
organisation's own reasoning preferences, their interpretation of contested areas,
and the specific failure modes they care about fixing. No two organisations will
produce the same dataset even from the same source documents.

The model and the infrastructure are commodities. The reviewed, domain-specific
training data is not. That is what the organisation owns, and what takes real
expertise and time to build.

**A note on "training from scratch":**

This pipeline never trains from scratch. Qwen3-8B already knows how to reason,
read statutes, structure arguments, and write clearly - that took Alibaba
billions of tokens and thousands of GPUs to build into the base weights. LoRA
fine-tuning borrows all of that and adds only the narrow domain layer on top.
Starting from a random initialisation instead would require that same pre-training
investment, which is not economically viable for any domain deployment.

This is why the base model choice matters (Doc 07) - you are inheriting years of
training investment for free. The distillation step then adds the thin layer of
domain expertise on top.
