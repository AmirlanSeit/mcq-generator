"""Prepare a balanced MCQ dataset for fine-tuning.

Fixes the two problems of the hand-made dataset:

1. Answer-position bias: the correct answer was "B" in 41% of the
   questions and "D" in only 5%, so the model simply learned to always
   answer "B". Here the options of every question are reshuffled so the
   correct letter is distributed exactly evenly across A-D.
2. Duplicates: 688 duplicated questions (some repeated 8 times) made the
   model memorise instead of generalise. Duplicates are removed.

Each question is written AUGMENT_COPIES times with the correct answer in
a different position each time. This teaches the model that the answer
letter depends on the option content, not on the position.

Usage:
    python prepare_dataset.py
"""

import random
import re
from collections import Counter
from pathlib import Path

SOURCE_FILE = "MCQ_clean1.txt"
OUTPUT_FILE = "MCQ_balanced.txt"
SEED = 42
AUGMENT_COPIES = 2  # shuffled copies of each question (1-4)

# Tolerant block parser: questions and options may span several lines,
# options may be written as "A." or "A)".
BLOCK_RE = re.compile(
    r"Level:[ \t]*(?P<level>[^\n]+)\n"
    r"\s*Topic:[ \t]*(?P<topic>[^\n]+)\n"
    r"\s*Question:[ \t]*(?P<question>.+?)\n"
    r"\s*A[.)][ \t]*(?P<a>.+?)\n"
    r"\s*B[.)][ \t]*(?P<b>.+?)\n"
    r"\s*C[.)][ \t]*(?P<c>.+?)\n"
    r"\s*D[.)][ \t]*(?P<d>.+?)\n"
    r"\s*Answer:[ \t]*(?P<answer>[A-D])",
    re.DOTALL,
)


def normalize_level(raw: str) -> str | None:
    """Map noisy level names (including mojibake like 'IGCSГ€') to canon."""
    low = raw.lower()
    if "igcs" in low:
        return "IGCSE"
    if "level" in low:
        return "A-Level"
    return None


def squash(text: str) -> str:
    """Collapse internal whitespace/newlines to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def parse_questions(text: str) -> list[dict]:
    questions = []
    for m in BLOCK_RE.finditer(text):
        level = normalize_level(m.group("level"))
        if level is None:
            continue
        options = [squash(m.group(k)) for k in "abcd"]
        correct_idx = "ABCD".index(m.group("answer"))
        questions.append({
            "level": level,
            "topic": squash(m.group("topic")),
            "question": squash(m.group("question")),
            "correct": options[correct_idx],
            "distractors": options[:correct_idx] + options[correct_idx + 1:],
        })
    return questions


def deduplicate(questions: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for q in questions:
        key = (q["level"], q["question"].lower(), q["correct"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)
    return unique


def balanced_blocks(questions: list[dict], rng: random.Random) -> list[str]:
    """Emit AUGMENT_COPIES shuffled copies of each question.

    The position of the correct answer follows a global A-B-C-D cycle,
    which gives an exactly even letter distribution, and consecutive
    copies of one question always land on different letters.
    """
    rng.shuffle(questions)
    blocks = []
    position = 0
    for q in questions:
        for _ in range(AUGMENT_COPIES):
            slot = position % 4
            position += 1
            distractors = q["distractors"][:]
            rng.shuffle(distractors)
            options = [None] * 4
            options[slot] = q["correct"]
            for i, d in zip((i for i in range(4) if i != slot), distractors):
                options[i] = d
            blocks.append("\n".join([
                f"Level: {q['level']}",
                f"Topic: {q['topic']}",
                f"Question: {q['question']}",
                *(f"{letter}. {opt}" for letter, opt in zip("ABCD", options)),
                f"Answer: {'ABCD'[slot]}",
            ]))
    return blocks


def main():
    rng = random.Random(SEED)
    text = Path(SOURCE_FILE).read_text(encoding="utf-8")
    raw_count = len(re.findall(r"(?m)^Question:", text))

    questions = parse_questions(text)
    unique = deduplicate(questions)
    blocks = balanced_blocks(unique, rng)
    Path(OUTPUT_FILE).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    answers = Counter(b.rsplit("Answer: ", 1)[1] for b in blocks)
    levels = Counter(b.split("\n", 1)[0].removeprefix("Level: ") for b in blocks)
    print(f"Blocks in source file:        {raw_count}")
    print(f"Parsed successfully:          {len(questions)}")
    print(f"After removing duplicates:    {len(unique)}")
    print(f"Written to {OUTPUT_FILE}:     {len(blocks)} (x{AUGMENT_COPIES} augmentation)")
    print(f"Answer distribution:          " +
          ", ".join(f"{k}: {answers[k]}" for k in "ABCD"))
    print(f"Levels:                       " +
          ", ".join(f"{k}: {v}" for k, v in levels.most_common()))


if __name__ == "__main__":
    main()
