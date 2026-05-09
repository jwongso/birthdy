# Doc 07 - Model Selection for Specialist LLM Deployments

## Purpose

Before fine-tuning anything, you need to choose the right base model. This
decision affects training cost, inference speed, final quality, and how easily
the model absorbs new domain knowledge. A wrong choice means weeks of wasted
GPU time.

This document covers the decision framework for selecting a base model for
specialist deployment - domain-expert models for professional services such as
law, accounting, medicine, and engineering.

---

## What "model architecture" actually means in practice

When people say "model architecture" they usually mean three distinct things:

### 1. Transformer variant (attention mechanism)

All modern LLMs are transformer-based but differ in how attention is computed:

- **Multi-Head Attention (MHA)** - original transformer, every head attends to
  full sequence. Accurate but memory-hungry at long contexts.
- **Grouped Query Attention (GQA)** - Llama-3, Qwen2.5 - multiple query heads
  share key/value heads. Faster inference, lower memory, negligible quality loss.
  This is the current standard.
- **Sliding Window Attention (SWA)** - Mistral - each token only attends to a
  fixed local window. Very fast but weaker at long-range dependencies. Less
  ideal for reasoning over long documents.

**For domain work requiring long documents: GQA models (Llama-3.1, Qwen2.5)
are the right choice.** They handle 128K context efficiently.

### 2. Dense vs Mixture of Experts (MoE)

**Dense model:**
```
Input token -> ALL parameters are used -> Output
Parameters: 8B, all active
Memory: ~5GB at Q4 quantization
Compute: proportional to all 8B parameters
```

**MoE model:**
```
Input token -> Router selects 2-4 "expert" sub-networks -> only those activate
Total parameters: 47B (e.g. Mixtral 8x7B)
Active parameters per token: ~13B (2 of 8 experts)
Memory: need to fit all 47B in RAM, but compute cost of 13B
```

MoE models store more diverse knowledge (each expert naturally specializes)
but require more RAM to hold all the experts even though most are idle per token.

**For single-domain specialist models: Dense wins.** Here is why:
- A single domain does not benefit from multi-expert diversity
- Easier and more predictable to fine-tune
- Lower RAM requirement on deployment hardware
- One focused expert network is better than eight general ones

MoE makes sense when you need one model to cover many unrelated domains
simultaneously - e.g. a single model that is expert in law AND medicine AND
accounting. Dense is the right starting point for focused deployments.

### 3. Pre-training corpus

This is arguably more important than architecture. A model trained on 2 trillion
tokens of code will reason about code better than one trained on 2 trillion tokens
of web text, regardless of architecture differences.

Key pre-training data signals to look for:

| Model family | Pre-training emphasis | Implication |
|-------------|----------------------|-------------|
| Qwen2.5-Coder | GitHub, code documentation | Strong code generation, weak on prose |
| DeepSeek-Coder | Code, math, reasoning | Excellent code + mathematical reasoning |
| Llama-3.1 | Broad web, books, code | Good generalist, strong instruction following |
| Qwen2.5 (base) | Broad + multilingual + reasoning | Strong at structured reasoning |
| Mistral | Broad web, instruct tuned | Strong instruction following, weaker long context |
| Phi-4 | Synthetic high-quality data | Punches above weight on reasoning, small size |

No major open model has been pre-trained on jurisdiction-specific law. You will
always be fine-tuning a generalist into a specialist. The question is which
generalist has the best foundation for the target domain.

---

## The "learning curve" question - catastrophic forgetting

The technical term for the risk when fine-tuning is **catastrophic forgetting**:
when you fine-tune on new domain data, the model may forget what it previously knew.

Factors that reduce catastrophic forgetting:

**1. LoRA (Low-Rank Adaptation)**
Does not touch the base weights at all. Adds small trainable matrices alongside
frozen base weights. The base model's knowledge is completely preserved. This is
why LoRA is the standard fine-tuning approach - it solves catastrophic forgetting
by design.

**2. Model size**
Larger models forget less. An 8B model fine-tuned on a new domain retains more
general reasoning ability than a 1B model fine-tuned on the same data. Larger
models have more capacity to absorb new knowledge without overwriting old.

**3. Training data quality over quantity**
500 high-quality, carefully curated Q&A pairs outperforms 5,000 scraped web pages.
The model learns the reasoning pattern, not just the facts.

**4. Learning rate**
Lower learning rate = less forgetting. Fine-tuning at 1e-4 preserves more base
knowledge than 1e-3. This is a training hyperparameter you control.

---

## The "creative / out-of-box" question

Models that can reason creatively - spot analogies, suggest novel arguments,
reason across domain boundaries - use **chain-of-thought reasoning** or
**extended thinking**. Models that support this natively:

- **Qwen3** - has a "thinking mode" where it reasons step by step before answering.
  Activated with `/think` in the prompt or a specific system prompt flag.
- **DeepSeek-R1** - trained specifically for long chain-of-thought reasoning.
  Extraordinarily good at multi-step problems. Heavy (671B full size, but 7B/8B
  distilled versions available).
- **Llama-3.1** - supports chain-of-thought via prompting but not natively trained
  for it.

Creative reasoning matters for domain work when:
- Applying a precedent from one area to an analogous situation in another
- Spotting weaknesses in an argument
- Suggesting novel interpretations of ambiguous rules or regulations

**Recommendation: Qwen3-8B with thinking mode** as the starting point. It has
native reasoning capability, fits in 8GB VRAM for inference, strong instruction
following, and the Alibaba team has published detailed fine-tuning guides.

---

## Model comparison

| Model | Size (Q4) | VRAM needed | Fine-tune ease | Reasoning | Context | Verdict |
|-------|-----------|-------------|---------------|-----------|---------|---------|
| Qwen3-8B | ~5.2GB | 8GB | Good | Excellent (thinking mode) | 128K | **Top pick** |
| Llama-3.1-8B | ~5.0GB | 8GB | Excellent | Good | 128K | Strong alternative |
| Mistral-7B-v0.3 | ~4.4GB | 8GB | Good | Good | 32K | Context too short |
| DeepSeek-R1-8B | ~5.0GB | 8GB | Moderate | Excellent | 128K | Good for reasoning tasks |
| Phi-4-mini | ~2.5GB | 6GB | Good | Good | 128K | Best for low-resource hardware |
| Qwen2.5-14B | ~9.0GB | 12GB | Good | Very good | 128K | When more VRAM is available |
| Llama-3.1-70B | ~42GB | 48GB+ | Good | Excellent | 128K | Cluster only |

---

## How to profile a model before committing to fine-tune

Before spending days fine-tuning, run a 30-minute benchmark:

### Step 1 - Download and start the model

```bash
./llama-server --hf-repo bartowski/Qwen_Qwen3-8B-GGUF \
  --hf-file Qwen_Qwen3-8B-Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 8192 --port 8080
```

### Step 2 - Measure tokens/sec

```bash
./llama-bench -m /path/to/model.gguf -ngl 999 -n 512 -p 512
```

Output gives you:
- `pp` - prompt processing speed (tokens/sec reading input)
- `tg` - text generation speed (tokens/sec writing output)

For a real-time chat application, `tg` is what users feel. Target: >20 tok/s
for acceptable UX, >40 tok/s for good UX.

Expected on RTX 4060 (8GB):
- Qwen3-8B Q4_K_M: ~35-45 tok/s generation
- Qwen2.5-14B Q4_K_M: ~15-20 tok/s (partial CPU offload needed)

### Step 3 - Domain baseline test

Before fine-tuning, test the base model on your target domain with 10-20 real
questions. Score the answers 1-5 for accuracy. This is your **baseline score**
before fine-tuning. After fine-tuning, run the same prompts and compare.

### Step 4 - Reasoning test

Test the thinking/reasoning capability with a scenario question that requires
applying rules to a fact pattern. A model with strong reasoning will identify
relevant issues and reason through them step by step. A weak model gives generic
answers. See Doc 12 for the full evaluation framework.

---

## Recommended architecture for a specialist model deployment

```
Base model: Qwen3-8B (or Llama-3.1-8B as fallback)
Quantization: Q4_K_M for inference (best quality/size tradeoff)
Fine-tuning method: LoRA (rank 16-32, alpha 32-64)
Fine-tuning hardware: Strix Halo cluster Node 2 (128GB unified memory)
Context window: 8K-32K for inference depending on document length
Retrieval: Qdrant + MCP server for live document lookup
Update cycle: Quarterly fine-tune as domain knowledge accumulates
```

One model per vertical rather than one giant model for all domains. Smaller,
more accurate, easier to evaluate, easier to update.

---

## Next steps

- **Doc 08** - RAG vs fine-tuning: when to use each, when to combine
- **Doc 09** - Knowledge distillation: using Claude to generate training data
- **Doc 10** - LoRA fine-tuning pipeline: actual training on Strix Halo
- **Doc 11** - MCP server: building domain-specific tool interfaces
- **Doc 12** - Evaluation: measuring whether fine-tuning actually worked
