"""Aggregate expert rating sheets into per-system results + agreement stats.

Usage (after eval_generate.py and after the raters filled their sheets):
    python eval_analyze.py eval_out/key.csv rater1.csv rater2.csv [rater3.csv ...]

Every rater file is a filled copy of eval_out/rater_sheet.csv. The script
joins them with the hidden key, reports per-system means for each rubric
dimension, novelty (computed automatically at generation time), and
inter-rater agreement (Fleiss' kappa per dimension, plus raw percent
agreement). Results go to stdout and eval_out/results.md.
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUBRIC = ["format_valid", "factually_correct", "distractor_quality",
          "difficulty_appropriate", "usable"]
CATEGORIES = {  # allowed values per dimension
    "format_valid": (0, 1),
    "factually_correct": (0, 1),
    "distractor_quality": (0, 1, 2),
    "difficulty_appropriate": (0, 1),
    "usable": (0, 1, 2),
}


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fleiss_kappa(item_category_counts: list[Counter], categories) -> float | None:
    """Fleiss' kappa; items must all have the same number of ratings."""
    n_raters = sum(item_category_counts[0].values())
    if n_raters < 2:
        return None
    n_items = len(item_category_counts)
    p_i = []
    category_totals = Counter()
    for counts in item_category_counts:
        if sum(counts.values()) != n_raters:
            return None  # unbalanced ratings; kappa undefined here
        agree = sum(c * (c - 1) for c in counts.values())
        p_i.append(agree / (n_raters * (n_raters - 1)))
        category_totals.update(counts)
    p_bar = sum(p_i) / n_items
    total = n_items * n_raters
    p_e = sum((category_totals[c] / total) ** 2 for c in categories)
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    key_path, *rater_paths = sys.argv[1:]

    key = {row["item_id"]: row for row in read_rows(key_path)}
    raters = [
        {row["item_id"]: row for row in read_rows(p)} for p in rater_paths
    ]
    for path, sheet in zip(rater_paths, raters):
        missing = [
            item_id for item_id, row in sheet.items()
            if any(row.get(c, "") == "" for c in RUBRIC)
        ]
        if missing:
            raise SystemExit(f"{path}: unfilled rubric cells for {missing[:5]}"
                             f"{'...' if len(missing) > 5 else ''}")

    # ratings[dimension][item_id] -> Counter of category -> votes
    ratings = {c: defaultdict(Counter) for c in RUBRIC}
    for sheet in raters:
        for item_id, row in sheet.items():
            if item_id not in key:
                raise SystemExit(f"Unknown item id {item_id} in a rater sheet")
            for c in RUBRIC:
                value = int(row[c])
                if value not in CATEGORIES[c]:
                    raise SystemExit(f"{item_id}: {c}={value} outside {CATEGORIES[c]}")
                ratings[c][item_id][value] += 1

    systems = sorted({row["system"] for row in key.values()})
    lines = []
    out = lines.append

    out(f"# Evaluation results ({len(key)} items, {len(raters)} raters)\n")

    # ---------------- per-system table (mean over items and raters) ---------
    header = ["system", "n_items", "novel_share"] + [f"mean_{c}" for c in RUBRIC] \
             + ["usable_as_is_share"]
    out("| " + " | ".join(header) + " |")
    out("|" + "---|" * len(header))
    for system in systems:
        ids = [i for i, row in key.items() if row["system"] == system]
        novel = sum(1 - int(key[i]["recalled"]) for i in ids) / len(ids)
        cells = [system, str(len(ids)), f"{novel:.2f}"]
        for c in RUBRIC:
            votes = [v for i in ids for v, n in ratings[c][i].items() for _ in range(n)]
            cells.append(f"{sum(votes) / len(votes):.2f}")
        as_is = [ratings["usable"][i][2] / sum(ratings["usable"][i].values())
                 for i in ids]
        cells.append(f"{sum(as_is) / len(as_is):.2f}")
        out("| " + " | ".join(cells) + " |")

    # ---------------- agreement ---------------------------------------------
    out("\n## Inter-rater agreement\n")
    out("| dimension | Fleiss' kappa | percent agreement |")
    out("|---|---|---|")
    for c in RUBRIC:
        counts = [ratings[c][i] for i in key]
        kappa = fleiss_kappa(counts, CATEGORIES[c])
        exact = sum(1 for cnt in counts if len(cnt) == 1) / len(counts)
        kappa_text = "n/a" if kappa is None else f"{kappa:.2f}"
        out(f"| {c} | {kappa_text} | {exact:.2f} |")

    out("\nKappa reading guide: <0.20 poor, 0.21-0.40 fair, 0.41-0.60 moderate, "
        "0.61-0.80 substantial, >0.80 almost perfect (Landis & Koch).")

    report = "\n".join(lines)
    print(report)
    results_path = Path(key_path).parent / "results.md"
    results_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
