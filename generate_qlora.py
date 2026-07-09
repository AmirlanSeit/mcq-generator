"""Generate IGCSE / A-Level CS MCQs with the QLoRA-tuned Qwen2.5-7B-Instruct.

    python generate_qlora.py

Loads the 4-bit base model from the HF cache and applies the LoRA
adapter from mcq_qlora/ (produced by train_qlora.py). Same interactive
loop and format validation as generate_question.py.
"""

import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# MCQ_BASE_MODEL override exists so the pipeline can be smoke-tested with
# an adapter trained on the small model (train_qlora.py --smoke).
BASE_MODEL = os.environ.get("MCQ_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# On a 16 GB RAM machine, loading the bf16 checkpoint (~15 GB) and
# quantizing it on the fly swaps heavily and takes ~10 minutes. The
# first run therefore saves the already-quantized model (~5.5 GB) here;
# later runs load it directly in a fraction of the time.
QUANTIZED_DIR = Path(__file__).parent / "qwen_base_4bit"

# The model sometimes reproduces training questions verbatim; those get
# flagged so new questions can be told apart from recalled ones.
DATASET_FILE = Path(__file__).parent / "MCQ_clean1.txt"
ADAPTER_PATH = str(Path(__file__).parent / "mcq_qlora")

# Must stay identical to PROMPT_TEMPLATE in train_qlora.py
PROMPT_TEMPLATE = "Generate a {level} Computer Science multiple-choice question. Topic: {topic}"

# 0.95 pushes the model away from reciting training questions towards
# composing new ones; the format filter below catches the extra noise.
NUM_CANDIDATES = 6
TEMPERATURE = 0.95
TOP_P = 0.95
MAX_NEW_TOKENS = 160

# Qwen's tokenizer keeps newlines, but accept one-line output too.
MCQ_RE = re.compile(
    r"Question:\s*(?P<question>.+?)\s*"
    r"A[.)]\s*(?P<a>.+?)\s*"
    r"B[.)]\s*(?P<b>.+?)\s*"
    r"C[.)]\s*(?P<c>.+?)\s*"
    r"D[.)]\s*(?P<d>.+?)\s*"
    r"Answer:\s*(?P<answer>[A-D])",
    re.DOTALL,
)

use_local = QUANTIZED_DIR.exists() and "MCQ_BASE_MODEL" not in os.environ
if use_local:
    print(f"Loading pre-quantized base from {QUANTIZED_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        QUANTIZED_DIR, device_map={"": 0}
    )
else:
    print(f"Loading {BASE_MODEL} (4-bit)... first run is slow (~10 min), "
          "the quantized copy will be cached for the next runs.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map={"": 0}
    )
    if "MCQ_BASE_MODEL" not in os.environ:
        print(f"Saving quantized base to {QUANTIZED_DIR} (~5.5 GB, one time)...")
        model.save_pretrained(QUANTIZED_DIR)

print(f"Applying adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def load_known_questions() -> set[str]:
    """Normalized question texts from the hand-made dataset."""
    if not DATASET_FILE.exists():
        return set()
    return {
        _normalize(line.removeprefix("Question:"))
        for line in DATASET_FILE.read_text(encoding="utf-8").splitlines()
        if line.startswith("Question:")
    }


KNOWN_QUESTIONS = load_known_questions()


def generate_raw(level: str, topic: str) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(level=level, topic=topic)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=NUM_CANDIDATES,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Strip the prompt: only decode the newly generated tokens.
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    return [tokenizer.decode(o, skip_special_tokens=True) for o in new_tokens]


def format_mcq(raw: str) -> str | None:
    """Validate and re-format model output, or None if malformed."""
    m = MCQ_RE.search(raw)
    if not m:
        return None
    options = [m.group(k).strip().lower() for k in "abcd"]
    # Reject duplicate options and options with a stray "B." etc. glued
    # inside (a T5-era failure mode; cheap to keep checking).
    if len(set(options)) < 4:
        return None
    if any(re.search(r"\b[A-D][.)]\s", opt) for opt in options):
        return None
    answer = m.group("answer")
    answer_text = m.group(answer.lower()).strip()
    question = m.group("question").strip()
    origin = (" [из датасета]" if _normalize(question) in KNOWN_QUESTIONS
              else " [новый]")
    return "\n".join([
        f"Question: {question}{origin}",
        f"A. {m.group('a').strip()}",
        f"B. {m.group('b').strip()}",
        f"C. {m.group('c').strip()}",
        f"D. {m.group('d').strip()}",
        f"Answer: {answer} ({answer_text})",
    ])


def interactive_mode():
    print("Type 'quit' as the level to exit.\n")
    while True:
        level = input("Level (IGCSE / A-Level): ").strip()
        if level.lower() in {"quit", "exit", "q"}:
            break
        topic = input("Topic: ").strip()

        raw_candidates = generate_raw(level, topic)
        seen, printed = set(), 0
        for raw in raw_candidates:
            formatted = format_mcq(raw)
            if formatted is None or formatted in seen:
                continue
            seen.add(formatted)
            printed += 1
            print(f"\n--- Question {printed} ---\n{formatted}")
        if printed == 0:
            print("\nNo well-formed question generated. Raw output was:")
            print(raw_candidates[0])
        print()


if __name__ == "__main__":
    interactive_mode()
