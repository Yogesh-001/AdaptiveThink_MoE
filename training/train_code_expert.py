"""
Code Expert LoRA Training Script
=================================

This script fine-tunes a LoRA (Low-Rank Adaptation) adapter on top of
Qwen2.5-0.5B-Instruct to create a specialized "Code Expert".

Key Concepts Explained:
-----------------------

1. WHY LoRA (not full fine-tuning)?
   - Full fine-tuning updates ALL model parameters (500M+ for even small models)
   - LoRA only trains ~0.1-1% of parameters by inserting small trainable matrices
   - This means: less memory, faster training, and you can swap experts easily
   - Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

2. HOW LoRA works:
   - For a weight matrix W (e.g., in attention), instead of updating W directly,
     LoRA adds a low-rank decomposition: W + BA where B is (d×r) and A is (r×d)
   - r (rank) is tiny (e.g., 16) compared to d (e.g., 1024)
   - So instead of training d×d = 1M params, you train 2×d×r = 32K params
   - The "alpha" parameter scales the LoRA output: scaling = alpha/r

3. WHY target specific modules?
   - We target attention layers (q_proj, k_proj, v_proj, o_proj) because
     that's where the model "decides what to attend to"
   - For code, attention patterns are crucial (matching brackets, variable refs)
   - Some papers also target gate_proj/up_proj/down_proj (MLP layers)
   - More targets = more capacity but slower training

4. WHY instruction format for training?
   - The model learns: "When I see this instruction pattern, generate code"
   - At inference time, we present the same instruction format → code expert activates
   - This is called "supervised fine-tuning" (SFT) — the standard first step
     in RLHF pipelines (InstructGPT, LLaMA2, etc.)

5. WHAT is gradient accumulation?
   - If batch_size=1 and gradient_accumulation_steps=16, the effective batch = 16
   - We process 1 sample at a time (fits in limited RAM) but accumulate gradients
     over 16 steps before updating weights
   - This simulates a larger batch without needing 16× the memory
"""

import json
import os
import sys
import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import Dataset


def load_config(config_path: str) -> dict:
    """
    Load training configuration from a YAML file.

    Why YAML?
    ---------
    - Separates hyperparameters from code (easy to experiment)
    - Human-readable, easy to version control
    - Common practice in ML research projects
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_formatted_dataset(data_path: str) -> Dataset:
    """
    Load our pre-formatted instruction-response JSON into a HuggingFace Dataset.

    Why HuggingFace Dataset object?
    --------------------------------
    - Integrates seamlessly with the Trainer API
    - Handles batching, shuffling, and memory-mapping automatically
    - Can handle datasets larger than RAM via Apache Arrow format
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    # Convert list of dicts → HuggingFace Dataset
    return Dataset.from_list(data)


def format_for_training(example: dict, tokenizer) -> dict:
    """
    Convert an instruction-response pair into the format the model expects.

    Why this specific format?
    -------------------------
    Qwen2.5 uses a chat template with special tokens:
      <|im_start|>system\n...<|im_end|>
      <|im_start|>user\n...<|im_end|>
      <|im_start|>assistant\n...<|im_end|>

    By training with this template, the model learns to:
    1. Recognize the instruction (user turn)
    2. Generate the appropriate code (assistant turn)

    The tokenizer's apply_chat_template() handles this formatting automatically.

    We set max_length=1024 because:
    - Most MBPP solutions are short (<200 tokens)
    - 1024 gives headroom for longer problems
    - Longer sequences = more memory, so we cap it
    """

    # Build the conversation in the chat format
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Write clean, correct Python code."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    # apply_chat_template converts messages → token IDs with special tokens
    # tokenize=True → returns token IDs directly
    # This is the EXACT format the model was pre-trained with
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Get the string first so we can inspect it
        add_generation_prompt=False,  # We include the assistant response (training, not inference)
    )

    # Now tokenize with truncation and padding
    tokenized = tokenizer(
        text,
        truncation=True,        # Cut off if longer than max_length
        max_length=1024,
        padding="max_length",   # Pad shorter sequences to max_length
        return_tensors=None,    # Return as lists (Dataset handles conversion)
    )

    # For causal LM training, labels = input_ids (model predicts next token)
    # The loss is computed only on non-padding tokens
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def train_code_expert(
    config_path: str = "configs/code_expert.yaml",
    data_path: str = "datasets/code/mbpp_formatted_subset.json",
    output_dir: str = "outputs/code_expert",
):
    """
    Main training function for the Code Expert LoRA adapter.

    Training Flow:
    1. Load base model (frozen — weights don't change)
    2. Attach LoRA adapters (small trainable matrices)
    3. Train only the LoRA weights on our code dataset
    4. Save just the LoRA weights (~5-20MB vs 1GB+ for full model)

    At inference time:
    - Load base model once
    - Load whichever LoRA adapter matches the task (code/math/general)
    - This is the core idea of our Mixture-of-Experts architecture
    """

    # ========================================
    # 1. LOAD CONFIGURATION
    # ========================================
    config = load_config(config_path)
    model_name = config["model_name"]

    print(f"Training Code Expert")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  LoRA rank: {config['lora']['r']}, alpha: {config['lora']['alpha']}")

    # ========================================
    # 2. LOAD TOKENIZER
    # ========================================
    # The tokenizer converts text → numbers (token IDs) and back
    # We use the same tokenizer as the base model (vocabulary must match)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # pad_token is needed for batching (padding shorter sequences)
    # Some models don't have one by default, so we set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================
    # 3. LOAD BASE MODEL
    # ========================================
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # On GPU: use float16 (half precision) to save memory
        # On CPU: must use float32 (CPU doesn't support float16 well)
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Move to CPU explicitly if no GPU
    if device == "cpu":
        model = model.to(device)

    # ========================================
    # 4. CONFIGURE LoRA
    # ========================================
    # LoraConfig defines WHERE and HOW to add LoRA adapters
    lora_config = LoraConfig(
        # task_type: Tells PEFT this is a causal language model
        # (affects which layers get modified)
        task_type=TaskType.CAUSAL_LM,

        # r (rank): Size of the low-rank matrices
        # Higher r = more parameters = more capacity but slower
        # 16 is a good default; 8 for very small models, 64 for large ones
        r=config["lora"]["r"],

        # lora_alpha: Scaling factor for LoRA weights
        # The actual scaling applied is: alpha / r
        # So alpha=32, r=16 → scale=2 (LoRA weights are doubled)
        # Higher alpha = LoRA has stronger influence on output
        lora_alpha=config["lora"]["alpha"],

        # lora_dropout: Regularization to prevent overfitting
        # Randomly zeros out LoRA weights during training
        # 0.05 = 5% dropout (conservative, good for small datasets)
        lora_dropout=config["lora"]["dropout"],

        # target_modules: WHICH layers get LoRA adapters
        # These are the attention projection matrices in Qwen:
        #   q_proj: Query projection (what am I looking for?)
        #   k_proj: Key projection (what do I contain?)
        #   v_proj: Value projection (what info do I pass forward?)
        #   o_proj: Output projection (combine attention heads)
        # Targeting all 4 gives the model maximum flexibility to
        # learn new attention patterns for code
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

        # bias: Whether to train bias terms too
        # "none" = only train LoRA matrices (standard practice)
        bias="none",
    )

    # get_peft_model: Wraps the base model with LoRA adapters
    # After this, only LoRA parameters are trainable
    # The original weights are FROZEN (requires_grad=False)
    model = get_peft_model(model, lora_config)

    # Print how many parameters are trainable vs frozen
    model.print_trainable_parameters()
    # Expected output: "trainable params: ~300K || all params: ~500M || 0.06%"

    # ========================================
    # 5. PREPARE DATASET
    # ========================================
    print(f"\nLoading dataset from: {data_path}")
    dataset = load_formatted_dataset(data_path)
    print(f"  Samples loaded: {len(dataset)}")

    # Tokenize all samples using our formatting function
    # batched=False: process one sample at a time (simpler, handles edge cases)
    # remove_columns: drop the original text columns, keep only token IDs
    tokenized_dataset = dataset.map(
        lambda example: format_for_training(example, tokenizer),
        batched=False,
        remove_columns=dataset.column_names,
    )

    print(f"  Tokenized samples: {len(tokenized_dataset)}")

    # ========================================
    # 6. CONFIGURE TRAINING
    # ========================================
    # TrainingArguments: All hyperparameters for the training loop
    training_args = TrainingArguments(
        # Where to save checkpoints
        output_dir=output_dir,

        # Number of full passes through the dataset
        # 2-3 epochs is standard for LoRA fine-tuning
        # More epochs on small data = overfitting risk
        num_train_epochs=config["training"]["epochs"],

        # Batch size per device (1 for CPU to avoid OOM)
        # On GPU you might use 2-4
        per_device_train_batch_size=config["training"]["batch_size"],

        # Accumulate gradients over N steps before updating weights
        # effective_batch_size = batch_size × gradient_accumulation_steps
        # = 1 × 16 = 16 (good balance of stability and speed)
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],

        # Learning rate: How big each weight update step is
        # 2e-4 is standard for LoRA (higher than full fine-tuning's ~2e-5)
        # LoRA can use higher LR because it only updates small matrices
        learning_rate=float(config["training"]["learning_rate"]),

        # Warmup: Gradually increase LR from 0 to target over first 10% of steps
        # Prevents early instability when weights are randomly initialized
        warmup_ratio=0.1,

        # Logging: Print loss every N steps (monitor training progress)
        logging_steps=10,

        # Save checkpoint every N steps (recover from crashes)
        save_steps=100,
        save_total_limit=2,  # Keep only 2 most recent checkpoints (save disk)

        # fp16/bf16: Use half-precision during training (2× faster, half memory)
        # Only works on GPU; disabled on CPU
        fp16=device == "cuda",

        # Disable wandb if not configured (avoids login prompts)
        report_to="none",

        # Gradient checkpointing: Trade compute for memory
        # Re-computes activations during backward pass instead of storing them
        # Saves ~30% memory at cost of ~20% slower training
        gradient_checkpointing=True if device == "cuda" else False,

        # DataLoader workers: Parallel data loading
        # 0 for CPU (multiprocessing overhead not worth it)
        dataloader_num_workers=0,
    )

    # ========================================
    # 7. CREATE TRAINER AND TRAIN
    # ========================================
    # The Trainer handles the entire training loop:
    # - Batching & shuffling data
    # - Forward pass (compute predictions)
    # - Loss computation (cross-entropy on next-token prediction)
    # - Backward pass (compute gradients)
    # - Gradient accumulation
    # - Weight updates (AdamW optimizer)
    # - Logging & checkpointing
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # DataCollatorForLanguageModeling with mlm=False:
        # Handles dynamic padding and creates proper attention masks
        # mlm=False means causal LM (predict next token), not masked LM (BERT-style)
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal language modeling (GPT-style)
        ),
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"  Total samples: {len(tokenized_dataset)}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"  Effective batch: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Device: {device}")
    print()

    # Actually run training
    trainer.train()

    # ========================================
    # 8. SAVE THE LoRA ADAPTER
    # ========================================
    # Only saves the LoRA weights (~5-20MB), NOT the full model
    # This is the key advantage: each expert is tiny and swappable
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"LoRA adapter saved to: {adapter_path}")
    print(f"You can load it later with:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"  model = PeftModel.from_pretrained(model, '{adapter_path}')")


if __name__ == "__main__":
    # Allow overriding paths via command line
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/code_expert.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/code/mbpp_formatted_subset.json"

    train_code_expert(config_path=config_path, data_path=data_path)
