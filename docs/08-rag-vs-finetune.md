# Doc 08 - RAG vs Fine-Tuning: When to Use Each

## The core question

When you want a model to know about a specific domain (law, medicine, accounting),
you have two fundamentally different approaches:

1. **RAG** - store documents in a vector database, retrieve relevant chunks at
   query time, inject them into the prompt
2. **Fine-tuning** - train the model on domain data so the knowledge is baked
   into the weights

Most people treat this as an either/or choice. It is not. The right answer for
most serious domain deployments is **both, for different purposes**. Understanding
why requires understanding what each approach is actually good at.

---

## What RAG does well

RAG retrieves documents and injects them as context. The model reads the
retrieved text and reasons over it.

```
User: "What is the limitation period for adverse possession under NZ law?"

RAG pipeline:
  1. Embed the question -> search Qdrant
  2. Retrieve: Land Transfer Act 2017 s.139, Limitation Act 2010 s.11
  3. Inject retrieved text into prompt
  4. Model reads the actual statute text and answers from it
```

**RAG excels at:**

- **Current facts** - statutes, regulations, and policies change. RAG reads
  the current document every time. Fine-tuning bakes in knowledge that goes stale.
- **Exact citations** - "section 139 of the Land Transfer Act 2017" comes from
  the retrieved document, not from model weights. No hallucinated section numbers.
- **Verifiability** - you can show exactly which document the answer came from.
  Critical for legal, medical, and compliance domains.
- **Adding new knowledge without retraining** - add a new document to Qdrant
  today, the model can answer about it tomorrow. No retraining required.
- **Domain-specific documents** - contracts, policies, and internal documents
  can be indexed without modifying the model.

**RAG struggles with:**

- **Implicit reasoning** - knowing WHEN to apply a rule, not just what it says.
  A model that has only seen a statute via RAG may not know that case X is an
  exception or that doctrine Y supersedes doctrine Z in this fact pattern.
- **Domain terminology** - if the model does not know what "Torrens title" means,
  it cannot reason about retrieved text containing the term.
- **Synthesizing across many documents** - RAG retrieves chunks. Reasoning that
  requires synthesizing 20 different sources simultaneously is hard to do via
  retrieval alone.
- **Context window limits** - you can only inject so many retrieved chunks.
  For a question requiring broad knowledge, retrieval may not surface everything
  relevant.

---

## What fine-tuning does well

Fine-tuning trains the model on domain-specific data. The knowledge becomes
part of the weights - the model "knows" it the way a trained specialist knows
their field.

```
Training data example:
  {"instruction": "Explain adverse possession under NZ law",
   "output": "Under the Land Transfer Act 2017, adverse possession of
              registered Torrens title land is effectively prohibited...
              For unregistered land, the Limitation Act 2010 s.11..."}

After training:
  Model answers domain questions correctly without needing retrieved context.
  It knows the domain the way a specialist knows their field.
```

**Fine-tuning excels at:**

- **Reasoning style** - how a specialist structures an argument, what questions
  to ask, how to identify the key issues in a fact pattern. This is procedural
  knowledge, not factual.
- **Domain vocabulary** - the model stops confusing jurisdiction-specific terms
  with equivalents from other countries or fields.
- **Implicit knowledge** - knowing that a particular claim is almost certainly
  not viable under current law, even before looking up a statute.
- **Format and tone** - professional domains have specific answer formats
  (e.g. IRAC for legal: Issue, Rule, Application, Conclusion). Fine-tuning
  teaches this style.
- **Reducing hallucination on known facts** - a fine-tuned model is less likely
  to invent names it has been trained to know correctly.

**Fine-tuning struggles with:**

- **Keeping current** - model weights are frozen at training time. New legislation
  or policy changes require retraining.
- **Exact citations** - fine-tuned models can still hallucinate specific section
  numbers even when they know the general law correctly.
- **Document-specific content** - you cannot fine-tune a separate model for each
  deployment's specific documents. Not practical.
- **Cost and time** - LoRA fine-tuning takes hours to days. You cannot retrain
  for every regulatory change.

---

## The combined architecture

The answer is to use each for what it is good at:

```
Fine-tuning handles:                    RAG handles:
  - Domain reasoning style                - Current document text
  - Specialist vocabulary                 - Recent decisions/rulings
  - Issue spotting ability                - Deployment-specific docs
  - Answer structure/format               - Regulatory updates
  - Domain intuition                      - Case-specific facts

         Both together:
  Fine-tuned model with domain intuition
  + RAG retrieval of current authoritative sources
  = Accurate, current, verifiable answers
```

A query flows through both layers:

```
1. User asks question
2. RAG retrieves relevant document sections
3. Retrieved text is injected into prompt with the question
4. Fine-tuned model reasons over retrieved text using its domain expertise
5. Answer is grounded in retrieved documents AND informed by trained reasoning
```

The model's fine-tuned knowledge helps it:
- Know WHICH retrieved chunks are most relevant
- Understand the terminology in retrieved documents
- Apply the retrieved content to the fact pattern correctly
- Identify when retrieved chunks are insufficient and flag uncertainty

---

## Decision framework

Use this to decide what to build for a given deployment:

```
Does the domain have rapidly changing information?
  YES -> RAG is mandatory. Fine-tuning alone will go stale.
  NO  -> Fine-tuning alone may suffice. Consider RAG anyway.

Does the use case need exact citations and sources?
  YES -> RAG is mandatory. Model weights cannot cite sources reliably.
  NO  -> Fine-tuning alone may suffice.

Are there deployment-specific documents (contracts, policies)?
  YES -> RAG is mandatory. Cannot fine-tune for every deployment's documents.
  NO  -> Fine-tuning alone may suffice.

Is the domain terminology highly specialized?
  YES -> Fine-tuning is needed. Base model uses wrong or generic terms.
  NO  -> RAG alone may suffice.

Does the task require multi-step domain reasoning?
  YES -> Fine-tuning is needed. RAG alone cannot teach reasoning style.
  NO  -> RAG alone may suffice.
```

**For a specialist professional domain (law, medicine, compliance): both are required.**
- Fine-tune for domain reasoning, vocabulary, issue spotting
- RAG for current documents, recent decisions, policy updates

**For a simpler use case (FAQ bot, internal knowledge base): RAG alone.**
- Index the relevant documents
- Base model is sufficient for reasoning
- No fine-tuning needed, faster to deploy, cheaper

---

## Update cycles

| Knowledge type | Storage | Update method | Frequency |
|---------------|---------|---------------|-----------|
| Domain reasoning style | Model weights | LoRA fine-tune | Annually |
| Authoritative document text | Qdrant | Re-index when changed | As needed |
| New decisions/rulings | Qdrant | Automated ingestion pipeline | Weekly/monthly |
| Deployment-specific docs | Qdrant | On demand | When updated |
| Regulatory updates | Qdrant | Monitoring + ingestion | Monthly |

---

## Practical timeline for a domain specialist deployment

```
Week 1-2: Discovery
  - Identify the top 50 questions users ask every day
  - Audit available documents: legislation, rulings, policies, guides

Week 3-4: RAG pipeline
  - Index authoritative documents into Qdrant
  - Set up embedding pipeline (nomic-embed-text via Ollama)
  - Test retrieval quality with RAGAS

Week 5-6: Knowledge distillation (Doc 09)
  - Use Claude to generate 500-1000 domain Q&A pairs
  - Review and correct the generated data
  - Build fine-tuning dataset in JSONL format

Week 7-8: Fine-tuning (Doc 10)
  - LoRA fine-tune Qwen3-8B on distilled dataset
  - Run Level 2 + Level 3 benchmarks before and after
  - Confirm: domain scores improve, general scores do not drop significantly

Week 9: Integration
  - Connect fine-tuned model + RAG pipeline
  - Deploy MCP server with domain tools (Doc 11)
  - End-to-end testing with real scenarios

Week 10: Deployment
  - Install on target hardware
  - User training
  - Document benchmark results showing quality improvement
```

---

## Summary

| | RAG only | Fine-tune only | RAG + Fine-tune |
|--|---------|---------------|-----------------|
| Current information | Yes | No | Yes |
| Exact citations | Yes | No | Yes |
| Domain reasoning | No | Yes | Yes |
| Specialist vocabulary | No | Yes | Yes |
| Deployment-specific docs | Yes | No | Yes |
| Fast to deploy | Yes | No | Medium |
| Keeps current | Yes | No | Yes |
| **Professional domain vertical** | Insufficient | Insufficient | **Required** |

Next: Doc 09 covers how to generate the fine-tuning dataset using Claude as
teacher - the knowledge distillation pipeline.
