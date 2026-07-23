# 🏦 Fine-Tuning TinyLlama for Financial Research Q&A

> **Honest summary**: This is a small-scale, educational fine-tuning exercise built in ~1.5 days.
> It demonstrates hands-on use of QLoRA, PEFT, and HuggingFace Transformers — not a production system.

---

## What this project does

Takes a pre-trained 1.1B-parameter language model (TinyLlama) and fine-tunes it on finance-domain
question-answer pairs using **QLoRA** — a memory-efficient technique that makes it possible to
fine-tune large language models on free, consumer-grade GPUs.

**Before fine-tuning**: TinyLlama answers finance questions with generic, sometimes off-topic responses.

**After fine-tuning**: Responses are more finance-specific in framing and vocabulary, and the model's
perplexity (confusion score) on held-out finance text measurably decreases.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base language model |
| `HuggingFace Transformers` | Load and run the model |
| `PEFT` | Apply LoRA adapters |
| `bitsandbytes` | 4-bit NF4 quantization (fits model on free GPU) |
| `TRL` (SFTTrainer) | Supervised fine-tuning training loop |
| `datasets` | Load the finance-alpaca dataset |
| Google Colab (T4 GPU) | Free compute |

---

## Dataset

**[finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)** by gbharti on HuggingFace

- ~68,000 finance Q&A pairs in Alpaca instruction format
- Covers: investing, macroeconomics, financial instruments, corporate finance, risk
- We use: **2,000 training samples** + **200 validation samples**
- Reason for subset: time and compute constraints (full dataset would take 12+ hours on T4)

---

## Fine-Tuning Approach

### Why QLoRA and not full fine-tuning?

Full fine-tuning of a 1.1B model requires storing all gradients and optimizer states — roughly
~12-16 GB of VRAM in fp32. A free Colab T4 has 16GB total, leaving no headroom.

QLoRA solves this in two ways:
1. **Quantization**: Compress model weights from 32-bit to 4-bit (NF4 format) — ~4x less memory
2. **LoRA**: Instead of updating all 1.1B parameters, add small adapter matrices (r=16) to the
   attention layers and only train those — ~8 million trainable parameters (less than 1% of total)

### LoRA configuration

```python
LoraConfig(
    r=16,                  # Rank: size of adapter matrices
    lora_alpha=32,         # Scaling = 2x rank (standard)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
```

### Training settings

| Setting | Value | Reason |
|---|---|---|
| Epochs | 3 | Enough passes to learn patterns without overfitting small dataset |
| Batch size | 4 x 4 accum = 16 effective | Memory limit on T4 |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Max seq length | 512 tokens | Covers most Q&A pairs; longer = more memory |
| Optimizer | paged_adamw_8bit | Memory-efficient variant of Adam |

### Training run summary

| Metric | Value |
|---|---|
| Trainable parameters | 4,505,600 / 1,104,553,984 (0.41%) |
| Training time | ~23 minutes 39 seconds (T4 GPU) |
| Final training loss | 1.3265 |
| Final validation loss | 1.3818 |

---

## How to Run

### Option 1: Google Colab (Recommended — free)

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `notebook.ipynb`
3. Set runtime: **Runtime > Change runtime type > T4 GPU**
4. Run all cells top to bottom

### Option 2: Kaggle Notebooks

1. Create a new notebook on [Kaggle](https://www.kaggle.com)
2. Upload `notebook.ipynb`
3. Enable GPU accelerator in settings

---

## Results

### Perplexity (lower = better)

| | Perplexity |
|---|---|
| Baseline (before fine-tuning) | `9.19` |
| After fine-tuning (3 epochs) | `3.82` |
| Improvement | `+58.4%` |

*Measured on 100 held-out validation examples from finance-alpaca.*

---

## What I Learned

1. **QLoRA makes fine-tuning accessible**: Being able to fine-tune a 1.1B model on a free GPU
   by training less than 1% of its parameters is genuinely impressive engineering.

2. **Data formatting matters**: TinyLlama expects a specific chat template. Mismatching the format
   during training vs. inference produces garbage outputs. Small detail, big impact.

3. **Perplexity is a proxy, not a ground truth**: Lower perplexity means the model understands
   finance text better, but it doesn't tell you if answers are factually correct. Real-world
   evaluation requires human review or domain benchmarks.

4. **Small datasets have limits**: 2,000 examples is enough to shift domain vocabulary and style,
   but not enough to teach new factual knowledge. The model still relies on pre-training for facts.

---

## Honest Limitations

| Limitation | Why it exists |
|---|---|
| Only 2,000 training samples | Free GPU time constraint |
| No external benchmark evaluation | Would require established finance benchmarks (e.g., FinBench) |
| Perplexity is the only quantitative metric | ROUGE/BLEU not meaningful for open-ended Q&A without reference answers |
| TinyLlama-1.1B is small | Chosen for feasibility, not performance |
| Results may vary between runs | Small models + small datasets = higher variance |

---

## What this project demonstrates

- Hands-on fine-tuning of an open-weight LLM (not an API call)
- PEFT/LoRA implementation using the HuggingFace ecosystem
- 4-bit quantization (QLoRA) for memory-constrained training
- Before/after evaluation with a quantitative metric (perplexity)
- Honest documentation of methodology and limitations

---

## Author

Built as a personal deep learning project to explore LLM fine-tuning techniques.
