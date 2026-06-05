# AdaptiveThink-MoE

A lightweight Mixture-of-Experts system using LoRA adapters and semantic routing. One shared base model dynamically swaps specialized expert adapters based on prompt classification.

## Architecture

```
User Prompt
     │
     ▼
┌──────────────┐
│   Router     │  (MiniLM embeddings + LogisticRegression)
│  96% accuracy│
└──────┬───────┘
       │ classifies → code / math / general
       ▼
┌──────────────┐
│  Base Model  │  Qwen2.5-0.5B-Instruct (494M params, loaded once)
│   + LoRA     │  Expert adapter swapped dynamically (~2M params each)
└──────┬───────┘
       │
       ▼
   Response
```

**Key idea:** Instead of running multiple full models, we use one frozen base model and swap tiny LoRA adapters (~0.4% of base params) per task — achieving specialization without multiplying model size.

## Results

| Metric | Value |
|--------|-------|
| Code Expert training loss | 1.57 → 0.69 |
| Math Expert training loss | 1.24 → 0.28 |
| General Expert training loss | 1.87 → 1.13 |
| Router accuracy | 96.1% |
| Trainable params per expert | ~2.16M (0.44% of base) |
| Adapter size on disk | ~15MB each |

## Project Structure

```
├── configs/               # YAML configs for each expert
│   ├── code_expert.yaml
│   ├── math_expert.yaml
│   └── general_expert.yaml
├── datasets/              # Dataset preparation scripts
│   ├── code/              # MBPP → instruction format
│   ├── math/              # GSM8K → CoT format
│   └── general/           # Alpaca-cleaned → instruction format
├── training/              # LoRA training scripts
│   ├── train_code_expert.py
│   ├── train_math_expert.py
│   └── train_general_expert.py
├── evaluation/            # Validation scripts
│   ├── validate_code_expert.py
│   ├── validate_math_expert.py
│   └── validate_general_expert.py
├── router/                # Prompt classification
│   ├── create_training_data.py
│   ├── train_router.py
│   └── router.py
├── scripts/               # Inference & utilities
│   ├── pipeline.py        # Core MoE pipeline
│   ├── cli.py             # Interactive CLI
│   ├── setup_and_run.py   # One-click Colab setup
│   ├── benchmark_comparison.py
│   └── test_pipeline.py
└── outputs/               # Trained adapters (gitignored)
```

## Quick Start

### Prerequisites

- Python 3.10+
- GPU recommended for training (Google Colab T4 works well)
- ~2GB disk space for model + adapters

### Installation

```bash
git clone https://github.com/Yogesh-001/AdaptiveThink_MoE.git
cd AdaptiveThink_MoE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### One-Click Setup

Trains everything from scratch and runs a demo:

```bash
python scripts/setup_and_run.py
```

### Step-by-Step

```bash
# 1. Prepare datasets
python datasets/code/prepare_dataset.py
python datasets/math/prepare_dataset.py
python datasets/general/prepare_dataset.py

# 2. Train expert adapters (GPU recommended)
python training/train_code_expert.py
python training/train_math_expert.py
python training/train_general_expert.py

# 3. Train router (CPU is fine)
python router/create_training_data.py
python router/train_router.py

# 4. Run the system
python scripts/cli.py
```

### Interactive CLI

```bash
python scripts/cli.py
```

```
You: Write a function to check if a number is prime
  → Expert: CODE | Confidence: 94.2%

You: What is 25% of 80?
  → Expert: MATH | Confidence: 91.8%

You: Explain how vaccines work
  → Expert: GENERAL | Confidence: 88.5%
```

### Benchmark

```bash
python scripts/benchmark_comparison.py
```

Compares base model vs MoE on code correctness, math accuracy, and general quality.

## Technical Details

### Base Model
- **Qwen2.5-0.5B-Instruct** — 494M parameters, instruction-tuned

### LoRA Configuration
- Rank: 16, Alpha: 32, Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- ~2.16M trainable params per expert (0.44% of base)

### Training Data
| Expert | Dataset | Samples | Epochs |
|--------|---------|---------|--------|
| Code | MBPP | 500 | 2 |
| Math | GSM8K | 500 | 3 |
| General | Alpaca-cleaned | 500 | 2 |

### Router
- Embedding: `all-MiniLM-L6-v2` (384-dim sentence embeddings)
- Classifier: LogisticRegression (C=10)
- Training samples: 251 labeled prompts
- Test accuracy: 96.1%

## References

- [Mixtral of Experts](https://arxiv.org/pdf/2401.04088) — Jiang et al., 2024
- [Switch Transformers](https://arxiv.org/pdf/2101.03961v3) — Fedus et al., 2022
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — Wei et al., 2022
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — Reimers & Gurevych, 2019

## License

This project is for research and educational purposes.
