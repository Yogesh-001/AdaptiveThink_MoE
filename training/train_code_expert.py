"""Train Code Expert LoRA adapter on MBPP dataset."""

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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_formatted_dataset(data_path: str) -> Dataset:
    with open(data_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_for_training(example: dict, tokenizer) -> dict:
    """Tokenize instruction-response pair using chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Write clean, correct Python code."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train_code_expert(
    config_path: str = "configs/code_expert.yaml",
    data_path: str = "datasets/code/mbpp_formatted_subset.json",
    output_dir: str = "outputs/code_expert",
):
    """Train LoRA adapter for code generation."""
    config = load_config(config_path)
    model_name = config["model_name"]

    print(f"Training Code Expert")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_path}")
    print(f"  LoRA rank: {config['lora']['r']}, alpha: {config['lora']['alpha']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    # Attach LoRA
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

    # Prepare dataset
    print(f"\nLoading dataset from: {data_path}")
    dataset = load_formatted_dataset(data_path)
    print(f"  Samples: {len(dataset)}")

    tokenized_dataset = dataset.map(
        lambda example: format_for_training(example, tokenizer),
        batched=False,
        remove_columns=dataset.column_names,
    )

    # Training
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"\n{'=' * 60}")
    print("STARTING TRAINING")
    print(f"{'=' * 60}")
    print(f"  Samples: {len(tokenized_dataset)} | Epochs: {config['training']['epochs']}")
    print(f"  Effective batch: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print()

    trainer.train()

    # Save adapter
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE — Adapter saved to: {adapter_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/code_expert.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/code/mbpp_formatted_subset.json"
    train_code_expert(config_path=config_path, data_path=data_path)
