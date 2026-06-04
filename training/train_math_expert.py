"""
Math Expert LoRA Training Script
==================================

Trains a LoRA adapter specialized in mathematical reasoning on GSM8K data.

Architecture is IDENTICAL to the code expert — same base model, same LoRA
config, just trained on different data. This is the core MoE principle:
- Same base model (shared knowledge)
- Different adapters (specialized skills)
- Router picks the right adapter at inference time

Key differences from code expert:
- System prompt: "step-by-step math" instead of "coding assistant"
- max_length: 512 (math solutions are shorter)
- epochs: 3 (math reasoning benefits from more exposure)

Why a separate script instead of parameterizing the code expert script?
----------------------------------------------------------------------
1. Clarity: Each expert's training is self-contained and readable
2. Customization: Math might need different preprocessing later
3. Research: Easy to experiment with one expert without breaking others
4. In production, you'd refactor to a shared training framework,
   but for research/learning, explicit is better than implicit
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


def format_for_training(example: dict, tokenizer, max_length: int = 512) -> dict:
    """
    Convert a math instruction-response pair into tokenized training format.

    The system prompt is crucial here — it tells the model:
    "You are a math expert, show step-by-step reasoning"

    This primes the model to:
    1. Break problems into steps (Chain-of-Thought)
    2. Show intermediate calculations
    3. Arrive at a clear final answer

    By training with this system prompt, the model associates
    this pattern with mathematical reasoning — so when the router
    sends a math query with this system prompt, the math expert
    "activates" its specialized knowledge.
    """

    # Math-specific system prompt emphasizing step-by-step reasoning
    # This is different from the code expert's "coding assistant" prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math assistant. Solve problems step by step, "
                "showing your reasoning clearly. End with the final numerical answer."
            ),
        },
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    # Convert to the model's chat template format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize with truncation
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # Labels = input_ids for causal LM training
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def train_math_expert(
    config_path: str = "configs/math_expert.yaml",
    data_path: str = "datasets/math/gsm8k_formatted_subset.json",
    output_dir: str = "outputs/math_expert",
):
    """
    Main training function for the Math Expert LoRA adapter.

    This follows the exact same pattern as train_code_expert.py:
    1. Load config → 2. Load model → 3. Attach LoRA → 4. Train → 5. Save

    The ONLY differences are:
    - Different training data (GSM8K instead of MBPP)
    - Different system prompt (math reasoning instead of coding)
    - Different max_length (512 instead of 1024)
    - Different output directory (outputs/math_expert/)
    """

    # 1. LOAD CONFIGURATION
    config = load_config(config_path)
    model_name = config["model_name"]
    max_length = config.get("max_length", 512)

    print(f"Training Math Expert")
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/math_expert.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/math/gsm8k_formatted_subset.json"

    train_math_expert(config_path=config_path, data_path=data_path)
