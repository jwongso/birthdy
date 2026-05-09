# Doc 10 - LoRA Fine-Tuning: Adapting a Model to a Domain

## What is LoRA?

LoRA (Low-Rank Adaptation) is a technique for fine-tuning large language models
without modifying their original weights. Instead of updating the full model
during training, LoRA inserts small trainable adapter matrices alongside the
frozen base weights.

```
Base model weights (frozen)
  +
LoRA adapter weights (trainable, ~1-5% of base model size)
  =
Fine-tuned behaviour
```

When training is done, the adapter can be merged into the base model for
inference, or kept separate and loaded on demand. The base model is never
modified - only the adapter changes.

---

## Why LoRA instead of full fine-tuning?

Full fine-tuning updates every weight in the model. For Qwen3-8B that is
8 billion parameters. Problems:

- **Memory** - storing gradients and optimiser state for 8B parameters requires
  ~100GB+ VRAM. Not feasible on consumer hardware.
- **Catastrophic forgetting** - training on 500 domain examples overwrites
  general capability baked in during pre-training. The model becomes good at
  your domain and bad at everything else.
- **Cost** - training time scales with the number of parameters being updated.

LoRA solves all three:

- **Memory** - adapters are tiny. A rank-16 LoRA on Qwen3-8B is ~50-100MB.
  Gradients only flow through the adapter, not the full model.
- **Forgetting** - base weights stay frozen. General capability is preserved.
  The adapter adds domain capability on top of it.
- **Cost** - training adapters on a 128GB unified memory machine takes hours,
  not days.

---

## The math (brief)

A weight matrix W in the model has shape [d_in, d_out]. Full fine-tuning updates
W directly. LoRA instead represents the update as two small matrices:

```
W_updated = W + (A x B)

where:
  A has shape [d_in, rank]
  B has shape [rank, d_out]
  rank << d_in, d_out  (typically 8, 16, or 32)
```

The number of trainable parameters drops from d_in x d_out to
(d_in + d_out) x rank - a massive reduction for large matrices.

During training only A and B are updated. W is never touched.

---

## Choosing LoRA rank

Rank controls the expressiveness of the adapter - how much the model can change
its behaviour. Higher rank = more expressive = more parameters = more memory.

| Rank | Use case |
|---|---|
| 4-8 | Style and tone only (personality, formality, response format) |
| 16 | Narrow domain vocabulary and reasoning (recommended default) |
| 32 | Broader domain adaptation with complex reasoning patterns |
| 64+ | Large domain shifts - rarely needed, diminishing returns |

For domain fine-tuning (legal, medical, accounting): **rank 16** is the right
starting point. Increase to 32 only if evaluation shows the model is still
missing domain reasoning after training.

---

## Tools: Unsloth

Unsloth is the recommended fine-tuning framework for this stack. It is optimised
for consumer and prosumer hardware (including AMD unified memory architectures)
and dramatically reduces memory usage compared to standard HuggingFace Trainer.

Key advantages:
- 2x faster training than standard transformers
- 60-70% less VRAM usage through kernel optimisation
- Native LoRA support with GGUF export
- Works with Qwen3, Llama, Mistral, Phi families

Install:

```bash
pip install unsloth
```

---

## Training script

A minimal but complete training script for domain fine-tuning:

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load base model with LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=4096,
    load_in_4bit=True,       # QLoRA: quantise base, train adapters in fp16
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    target_modules=[         # which weight matrices to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,           # scaling factor, typically equal to rank
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# Load dataset (JSONL with "messages" field)
dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/val.jsonl",
})

# Format messages into a single string the model trains on
def format_messages(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_messages)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
)

trainer.train()
```

---

## QLoRA: quantisation + LoRA

The script above uses `load_in_4bit=True`. This is QLoRA - the base model is
loaded in 4-bit quantisation to save memory, but the LoRA adapter weights are
trained in full float16 precision.

Result: you can fine-tune an 8B model on hardware with 24-32GB of memory. On a
128GB unified memory machine (AMD Ryzen AI Max+ 395) you can go to larger models
or larger batch sizes comfortably.

---

## Exporting to GGUF

After training, export the merged model to GGUF format for use with llama-server:

```python
# Merge adapter into base model and export
model.save_pretrained_gguf(
    "exports/domain-qwen3-8b",
    tokenizer,
    quantization_method="q4_k_m",  # same quantisation as the base model
)
```

This produces a `.gguf` file you can drop into llama-server directly:

```bash
llama-server \
  --model exports/domain-qwen3-8b/domain-qwen3-8b-Q4_K_M.gguf \
  --n-gpu-layers 999 \
  --ctx-size 8192 \
  --port 8080
```

The fine-tuned model is a drop-in replacement. Nothing else in the stack changes.

---

## Training hyperparameters explained

| Parameter | Value | Why |
|---|---|---|
| num_train_epochs | 3 | Enough passes for small datasets without overfitting |
| learning_rate | 2e-4 | Standard LoRA learning rate - higher than full fine-tuning |
| lora_alpha | 16 | Equal to rank - balanced scaling |
| lora_dropout | 0.05 | Light regularisation to reduce overfitting |
| gradient_accumulation_steps | 4 | Simulates larger batch size without more memory |
| warmup_steps | 50 | Prevents early instability in learning rate |

If validation loss stops decreasing before epoch 3 ends, stop early. If
validation loss is still decreasing at epoch 3, add more epochs cautiously -
watch for the gap between train loss and val loss widening (overfitting signal).

---

## Monitoring training

Watch two numbers:

- **Training loss** - should decrease steadily. If it plateaus early, learning
  rate may be too low or data too repetitive.
- **Validation loss** - should track training loss closely. If it starts rising
  while training loss falls, the model is memorising the training set.

```
Epoch 1: train=1.42, val=1.38  (good - both decreasing)
Epoch 2: train=1.21, val=1.19  (good - gap is small)
Epoch 3: train=0.98, val=1.22  (warning - gap widening, possible overfit)
```

If you see overfitting: increase lora_dropout, reduce num_train_epochs, or add
more diverse training examples.

---

## Protecting general capability

To avoid the model losing general reasoning capability during domain fine-tuning,
mix general examples into the training set:

- 80% domain-specific examples (generated via distillation, Doc 09)
- 20% general examples (coding, math, general Q&A)

The 20% reminds the adapter not to overspecialise. Without it, after fine-tuning
on NZ legal data, the model may start applying legal reasoning patterns to
unrelated questions.

---

## Evaluating the fine-tuned model

After training, run the same evaluation benchmark used in Doc 12 on both the
base model and the fine-tuned model side by side:

```
Base Qwen3-8B:
  NZ legal Q1: 3/5 (cites LTA 1989)
  NZ legal Q2: 4/5 (wrong statute section)
  C++ RAII:    5/5

Fine-tuned:
  NZ legal Q1: 5/5 (cites LTA 2017 s.139 correctly)
  NZ legal Q2: 5/5 (correct section, correct reasoning)
  C++ RAII:    5/5 (general capability preserved)
```

The C++ score is the canary. If it drops after fine-tuning, general capability
has been eroded - increase the general example ratio and retrain.

---

## Iteration cycle

Fine-tuning is not a one-shot process:

```
1. Train on distillation dataset (Doc 09)
2. Run evaluation benchmark (Doc 12)
3. Identify remaining gaps
4. Generate more distillation examples targeting those gaps
5. Retrain (or continue training from last checkpoint)
6. Re-evaluate
7. Repeat until quality threshold is met
```

Each iteration tightens the model's domain performance. Three rounds is typical
for a narrow domain starting from a capable base model like Qwen3-8B.
