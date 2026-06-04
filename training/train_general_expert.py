"""
General Expert LoRA Training Script
=====================================

Trains a LoRA adapter for general instruction-following on Alpaca data.

This expert is the "catch-all" — it handles tasks that aren't code or math:
- Open-ended Q&A
- Explanations and teaching
- Summarization
- Creative writing
- General advice

In the MoE system, the router sends prompts here when they don't match
the code or math expert patterns. Think of it as the "default" expert.

Why do we need a General Expert at all?
---------------------------------------
Without it, non-code/non-math prompts would go to the base model without
any LoRA adapter. The General Expert:
1. Improves instruction-following quality over base model
2. Provides consistent response format across all experts
3. Gives the router a clear third option (code / math / general)
4. Teaches the model to give helpful, complete answers

Architecture: Same as Code and Math experts (LoRA on attention layers).
Only the training data and system prompt differ.
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
)
from datasets import Dataset


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_formatted_dataset(data_path: str) -> Dataset:
    """Load formatted JSON dataset into HuggingFace Dataset object."""
    with open(data_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_for_training(example: dict, tokenizer, max_length: int = 768) -> dict:
    """
    Convert an instruction-response pair into tokenized training format.

    The General Expert uses a broad system prompt that emphasizes:
    - Being helpful and informative
    - Giving complete, well-structured answers
    - Adapting tone to the question type

    This is intentionally generic (unlike "coding assistant" or "math assistant")
    so it handles the full range of general queries.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, knowledgeable assistant. Provide clear, "
                "accurate, and well-organized responses to any question or task."
            ),
        },
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def train_general_expert(
    config_path: str = "configs/general_expert.yaml",
    data_path: str = "datasets/general/alpaca_formatted_subset.json",
    output_dir: str = "outputs/general_expert",
):
    """
    Main training function for the General Expert LoRA adapter.

    Same pattern as code and math experts:
    Load config → Load model → Attach LoRA → Train → Save adapter
    """

    # 1. LOAD CONFIGURATION
    config = load_config(config_path)
    model_name = config["model_name"]
    max_length = config.get("max_length", 768)

    print(f"Training General Expert")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  LoRA rank: {config['lora']['r']}, alpha: {config['lora']['alpha']}")
    print(f"  Max length: {max_length}")

    # 2. LOAD TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. LOAD BASE MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    # 4. CONFIGURE AND ATTACH LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. PREPARE DATASET
    print(f"\nLoading dataset from: {data_path}")
    dataset = load_formatted_dataset(data_path)
    print(f"  Samples loaded: {len(dataset)}")

    tokenized_dataset = dataset.map(
        lambda example: format_for_training(example, tokenizer, max_length),
        batched=False,
        remove_columns=dataset.column_names,
    )
    print(f"  Tokenized samples: {len(tokenized_dataset)}")

    # 6. CONFIGURE TRAINING
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=device == "cuda",
        report_to="none",
        gradient_checkpointing=True if device == "cuda" else False,
        dataloader_num_workers=0,
    )

    # 7. TRAIN
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
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

    trainer.train()

    # 8. SAVE ADAPTER
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/general_expert.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/general/alpaca_formatted_subset.json"

    train_general_expert(config_path=config_path, data_path=data_path)
