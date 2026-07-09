"""QLoRA fine-tuning of Qwen2.5-7B-Instruct to generate IGCSE / A-Level CS MCQs.

Run prepare_dataset.py first to create MCQ_balanced.txt, then:
    python train_qlora.py

Smoke test (small model, 32 examples, a few steps — checks the script
end-to-end before committing hours to the real run):
    python train_qlora.py --smoke

Why this replaces ai.py (FLAN-T5-large): the 780M T5 learned the MCQ
*format* but invents factually wrong questions — it has no real CS
knowledge to draw on. A 7B instruct model already knows the subject from
pretraining; the fine-tune only has to teach it the exam style and
format, which is exactly what LoRA is good at.

Fits an RTX 3060 (12 GB):
- base weights quantized to 4-bit NF4 (~4.5 GB) and frozen;
- LoRA adapters (r=16 on all attention + MLP projections) are the only
  trained weights — a few tens of MB;
- gradient checkpointing + paged 8-bit AdamW for the rest of the margin;
- sequences are short (~150 tokens), so activations are cheap.
"""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "MCQ_balanced.txt"
OUTPUT_DIR = str(PROJECT_DIR / "results_qlora")
FINAL_DIR = str(PROJECT_DIR / "mcq_qlora")

# Must stay identical to PROMPT_TEMPLATE in generate_qlora.py
PROMPT_TEMPLATE = "Generate a {level} Computer Science multiple-choice question. Topic: {topic}"

MAX_LENGTH = 256          # prompt + question + 4 options + answer fits easily
NUM_EPOCHS = 3            # instruct models converge fast on style transfer
BATCH_SIZE = 2
GRAD_ACCUM = 8            # effective batch size 16
LEARNING_RATE = 2e-4      # standard for LoRA (only adapters are trained)
EVAL_FRACTION = 0.1
SEED = 42


def load_groups(filename) -> list[list[tuple[str, str]]]:
    """Load (prompt, target) pairs grouped by question text.

    Grouping matters: MCQ_balanced.txt contains several shuffled copies
    of each question, and all copies must stay on the same side of the
    train/validation split.
    """
    blocks = Path(filename).read_text(encoding="utf-8").strip().split("\n\n")
    groups: dict[str, list[tuple[str, str]]] = {}
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) != 8:
            continue
        level = lines[0].removeprefix("Level:").strip()
        topic = lines[1].removeprefix("Topic:").strip()
        prompt = PROMPT_TEMPLATE.format(level=level, topic=topic)
        target = "\n".join(lines[2:])  # Question + options + Answer
        groups.setdefault(lines[2], []).append((prompt, target))
    print(f"Loaded {sum(len(g) for g in groups.values())} examples "
          f"({len(groups)} unique questions)")
    return list(groups.values())


class MCQChatDataset(Dataset):
    """Chat-formatted examples with the prompt tokens masked out of the loss.

    The model only learns to produce the assistant turn (the MCQ itself),
    not to reproduce the user prompt.
    """

    def __init__(self, pairs: list[tuple[str, str]], tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, target = self.pairs[idx]
        messages = [{"role": "user", "content": prompt}]
        # transformers v5 returns a BatchEncoding, not a plain id list.
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        target_ids = self.tokenizer(
            target + self.tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]

        input_ids = (prompt_ids + target_ids)[:MAX_LENGTH]
        labels = ([-100] * len(prompt_ids) + target_ids)[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def collate(batch, pad_token_id):
    """Right-pad a batch; label padding is -100 so it is ignored by the loss."""
    max_len = max(len(item["input_ids"]) for item in batch)
    for item in batch:
        pad = max_len - len(item["input_ids"])
        item["input_ids"] = item["input_ids"] + [pad_token_id] * pad
        item["attention_mask"] = item["attention_mask"] + [0] * pad
        item["labels"] = item["labels"] + [-100] * pad
    return {
        key: torch.tensor([item[key] for item in batch])
        for key in ("input_ids", "attention_mask", "labels")
    }


def train_model(smoke: bool = False):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" if smoke else MODEL_NAME
    print(f"Model: {model_name}{' (SMOKE TEST)' if smoke else ''}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    groups = load_groups(DATA_FILE)
    if smoke:
        groups = groups[:32]
    train_groups, eval_groups = train_test_split(
        groups, test_size=EVAL_FRACTION, random_state=SEED
    )
    train_pairs = [p for g in train_groups for p in g]
    eval_pairs = [p for g in eval_groups for p in g]
    print(f"Train: {len(train_pairs)} examples, eval: {len(eval_pairs)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1 if smoke else NUM_EPOCHS,
        max_steps=4 if smoke else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="paged_adamw_8bit",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=0 if smoke else 30,
        # In smoke mode training stops after 4 steps, mid-epoch, so
        # epoch-based eval/save would never fire.
        eval_strategy="steps" if smoke else "epoch",
        eval_steps=2 if smoke else None,
        save_strategy="steps" if smoke else "epoch",
        save_steps=2 if smoke else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1 if smoke else 20,
        bf16=True,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MCQChatDataset(train_pairs, tokenizer),
        eval_dataset=MCQChatDataset(eval_pairs, tokenizer),
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training started...")
    trainer.train()

    # Only the LoRA adapter is saved (tens of MB); the 4-bit base is
    # re-downloaded/loaded from the HF cache at generation time.
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"Adapter saved to {FINAL_DIR}. Run generate_qlora.py to use it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="quick end-to-end check on a 0.5B model")
    train_model(smoke=parser.parse_args().smoke)
