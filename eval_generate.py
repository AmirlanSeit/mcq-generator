"""Build a blind evaluation set comparing three MCQ generation systems.

Systems compared under one protocol (same prompts, same sampling):
    finetuned  - Qwen2.5-7B-Instruct + LoRA adapter from mcq_qlora/
    zeroshot   - plain Qwen2.5-7B-Instruct, same one-line prompt
    fewshot    - plain Qwen2.5-7B-Instruct, prompt with 3 example MCQs

For every (level, topic) pair each system produces K_RAW raw samples.
Format validity is scored automatically over ALL raw samples (that is a
per-system metric on its own); the first K_KEEP well-formed items go into
the blind set that human raters see.

Run on the GPU machine, in the project folder:
    python eval_generate.py            # real run, needs GPU + adapter
    python eval_generate.py --mock     # pipeline check without a model

Output (eval_out/):
    blind_items.csv   what raters see: item id + question, no system names
    rater_sheet.csv   blind_items + empty rubric columns; one copy PER RATER
    key.csv           item id -> system + novelty flag; do NOT give to raters
    auto_stats.csv    per-system format validity over raw samples
Then collect the filled sheets and run eval_analyze.py.
"""

import argparse
import csv
import random
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATASET_FILE = PROJECT_DIR / "MCQ_clean1.txt"
ADAPTER_PATH = str(PROJECT_DIR / "mcq_qlora")
OUT_DIR = PROJECT_DIR / "eval_out"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
K_RAW = 6    # raw samples per system per (level, topic)
K_KEEP = 2   # well-formed items per system per (level, topic) in the blind set

# Must stay identical to PROMPT_TEMPLATE in train_qlora.py
PROMPT_TEMPLATE = "Generate a {level} Computer Science multiple-choice question. Topic: {topic}"

# Evaluation pairs: frequent topics of the corpus, both exam levels.
EVAL_PAIRS = [
    (level, topic)
    for level in ("IGCSE", "A-Level")
    for topic in ("Networking", "Databases", "Programming",
                  "Security", "Data representation",
                  "Logic gates and logic circuits")
]

# Few-shot examples come from topics OUTSIDE the evaluation list so the
# baseline cannot copy content from the test topics.
FEWSHOT_TOPICS = ("Algorithm Design", "Software Licensing", "Data Validation")

TEMPERATURE = 0.95
TOP_P = 0.95
MAX_NEW_TOKENS = 160

MCQ_RE = re.compile(
    r"Question:\s*(?P<question>.+?)\s*"
    r"A[.)]\s*(?P<a>.+?)\s*"
    r"B[.)]\s*(?P<b>.+?)\s*"
    r"C[.)]\s*(?P<c>.+?)\s*"
    r"D[.)]\s*(?P<d>.+?)\s*"
    r"Answer:\s*(?P<answer>[A-D])",
    re.DOTALL,
)

RUBRIC_COLUMNS = [
    # 0/1: question + 4 options + answer readable and self-contained
    "format_valid",
    # 0/1: the marked answer is the single correct one
    "factually_correct",
    # 0/1/2: 0 = distractors nonsensical, 1 = weak, 2 = plausible
    "distractor_quality",
    # 0/1: difficulty matches the stated exam level
    "difficulty_appropriate",
    # 0/1/2: 0 = reject, 1 = usable after a small edit, 2 = usable as is
    "usable",
]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def load_known_questions() -> set[str]:
    if not DATASET_FILE.exists():
        return set()
    return {
        normalize(line.removeprefix("Question:"))
        for line in DATASET_FILE.read_text(encoding="utf-8").splitlines()
        if line.startswith("Question:")
    }


def load_fewshot_examples() -> list[str]:
    """First corpus question of each FEWSHOT_TOPICS, verbatim block."""
    text = DATASET_FILE.read_text(encoding="utf-8")
    examples = []
    for topic in FEWSHOT_TOPICS:
        m = re.search(
            rf"Level:[^\n]*\nTopic:[ \t]*{re.escape(topic)}\n.*?Answer:[ \t]*[A-D]",
            text, re.DOTALL,
        )
        if m:
            examples.append(m.group(0).strip())
    if len(examples) < len(FEWSHOT_TOPICS):
        raise SystemExit("Could not find all few-shot example topics in the corpus")
    return examples


def build_prompt(system: str, level: str, topic: str, fewshot: list[str]) -> str:
    base = PROMPT_TEMPLATE.format(level=level, topic=topic)
    if system != "fewshot":
        return base
    shots = "\n\n".join(fewshot)
    return (
        "Here are examples of exam multiple-choice questions:\n\n"
        f"{shots}\n\n"
        f"{base}\n"
        "Answer in exactly the same format: Question, options A-D, Answer."
    )


def parse_mcq(raw: str) -> dict | None:
    m = MCQ_RE.search(raw)
    if not m:
        return None
    options = [m.group(k).strip() for k in "abcd"]
    if len({o.lower() for o in options}) < 4:
        return None
    if any(re.search(r"\b[A-D][.)]\s", o) for o in options):
        return None
    return {
        "question": re.sub(r"\s+", " ", m.group("question")).strip(),
        "a": options[0], "b": options[1], "c": options[2], "d": options[3],
        "answer": m.group("answer"),
    }


# ---------------------------------------------------------------- generation

def generate_real(pairs, fewshot):
    """Yield (system, level, topic, [raw strings]) using the actual models."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Loading {BASE_MODEL} (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def sample(prompt: str) -> list[str]:
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
                num_return_sequences=K_RAW,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        return [tokenizer.decode(o, skip_special_tokens=True) for o in new_tokens]

    # Baselines first (plain base model), then the adapter goes on top.
    for system in ("zeroshot", "fewshot"):
        for level, topic in pairs:
            print(f"[{system}] {level} / {topic}")
            yield system, level, topic, sample(build_prompt(system, level, topic, fewshot))

    print(f"Applying adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    for level, topic in pairs:
        print(f"[finetuned] {level} / {topic}")
        yield "finetuned", level, topic, sample(
            build_prompt("finetuned", level, topic, fewshot))


def generate_mock(pairs, fewshot, rng):
    """Fake raw outputs to test the pipeline end to end without a GPU."""
    for system in ("zeroshot", "fewshot", "finetuned"):
        for level, topic in pairs:
            raws = []
            for i in range(K_RAW):
                if rng.random() < 0.2:
                    raws.append(f"Sorry, here is a question about {topic}...")
                else:
                    raws.append(
                        f"Question: Mock {system} question {i} on {topic} ({level})?\n"
                        f"A. Option one\nB. Option two\nC. Option three\nD. Option four\n"
                        f"Answer: {'ABCD'[i % 4]}"
                    )
            yield system, level, topic, raws


# ------------------------------------------------------------------- output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="pipeline check without loading a model")
    args = parser.parse_args()

    rng = random.Random(SEED)
    known = load_known_questions()
    fewshot = load_fewshot_examples()
    OUT_DIR.mkdir(exist_ok=True)

    source = (generate_mock(EVAL_PAIRS, fewshot, rng) if args.mock
              else generate_real(EVAL_PAIRS, fewshot))

    kept = []       # dicts with system/level/topic/parsed
    auto_stats = {} # system -> [raw_total, valid_total]
    for system, level, topic, raws in source:
        stats = auto_stats.setdefault(system, [0, 0])
        n_kept = 0
        for raw in raws:
            stats[0] += 1
            parsed = parse_mcq(raw)
            if parsed is None:
                continue
            stats[1] += 1
            if n_kept < K_KEEP:
                n_kept += 1
                kept.append({
                    "system": system, "level": level, "topic": topic,
                    "recalled": int(normalize(parsed["question"]) in known),
                    **parsed,
                })
        if n_kept < K_KEEP:
            print(f"  WARNING: {system} {level}/{topic}: "
                  f"only {n_kept}/{K_KEEP} well-formed items")

    # Blind order: shuffle and assign neutral ids.
    rng.shuffle(kept)
    for i, item in enumerate(kept, 1):
        item["item_id"] = f"Q{i:03d}"

    def write_csv(path, fieldnames, rows):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    blind_fields = ["item_id", "level", "topic",
                    "question", "a", "b", "c", "d", "answer"]
    write_csv(OUT_DIR / "blind_items.csv", blind_fields, kept)
    write_csv(OUT_DIR / "rater_sheet.csv",
              blind_fields + RUBRIC_COLUMNS,
              [{**item, **{c: "" for c in RUBRIC_COLUMNS}} for item in kept])
    write_csv(OUT_DIR / "key.csv",
              ["item_id", "system", "level", "topic", "recalled"], kept)
    write_csv(OUT_DIR / "auto_stats.csv",
              ["system", "raw_samples", "well_formed", "format_valid_rate"],
              [{"system": s, "raw_samples": t, "well_formed": v,
                "format_valid_rate": f"{v / t:.3f}"}
               for s, (t, v) in sorted(auto_stats.items())])

    print(f"\n{len(kept)} items in the blind set -> {OUT_DIR}")
    print("Give each rater their own copy of rater_sheet.csv;")
    print("keep key.csv to yourself until the sheets are filled.")


if __name__ == "__main__":
    main()
