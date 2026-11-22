"""Pull a random sample of judged items into one CSV for hand-labelling, then measure agreement.

Usage:
    python eval/make_labels.py make     # writes eval/results/hand_labels.csv (fill the my_* columns)
    python eval/make_labels.py score    # agreement between you and the judges
"""
import csv
import json
import random
import re
import sys
from pathlib import Path

RES = Path("eval/results")
LABELS = RES / "hand_labels.csv"
FIELDS = ["kind", "link", "question_or_bullet", "passages", "judge", "my_label"]


def bullets_of(advice):
    """The advice's bullet points, in order."""
    return [l.strip()[1:].strip() for l in advice.splitlines() if l.strip().startswith(("-", "*", "•"))]


def make(n_retrieval=10, n_bullets=10, seed=42):
    rng = random.Random(seed)

    with open(RES / "retrieval.csv", encoding="utf-8") as f:
        retrieval = list(csv.DictReader(f))
    rows = []
    for r in rng.sample(retrieval, n_retrieval):
        rows.append({"kind": "retrieval",
                     "link": f"https://github.com/{r['repo']}",
                     "question_or_bullet": r["question"],
                     "passages": r["passages"],
                     "judge": r["found"],          # True = the passages answer the question
                     "my_label": ""})

    answers = [json.loads(l) for l in (RES / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l]
    pool = []
    for rec in answers:
        for side in ["A", "B"]:
            model = rec[f"{side.lower()}_model"]
            texts = bullets_of(rec["answers"][model]["advice"])
            for text, verdict in zip(texts, rec["verdict"].get(side, [])):
                pool.append({"kind": "bullet_accurate",
                             "link": f"https://github.com/{rec['repo']}/pull/{rec['number']}",
                             "question_or_bullet": text, "passages": "",
                             "judge": bool(verdict.get("accurate")), "my_label": ""})
    rows += rng.sample(pool, n_bullets)

    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} items to {LABELS}.")
    print("Fill my_label with True or False. Don't look at the 'judge' column until you're done.")
    print("  retrieval rows: do the passages answer the question?")
    print("  bullet rows:    is what the bullet says about the PR true? (open the link)")


def score():
    with open(LABELS, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["my_label"].strip()]
    if not rows:
        print("No labels filled in yet.")
        return
    for kind in ["retrieval", "bullet_accurate"]:
        rs = [r for r in rows if r["kind"] == kind]
        if rs:
            agree = sum(r["judge"].strip().lower() == r["my_label"].strip().lower() for r in rs)
            print(f"{kind:16} agreement {agree}/{len(rs)} = {agree / len(rs):.0%}")
    agree = sum(r["judge"].strip().lower() == r["my_label"].strip().lower() for r in rows)
    print(f"{'overall':16} agreement {agree}/{len(rows)} = {agree / len(rows):.0%}")
    for r in rows:
        if r["judge"].strip().lower() != r["my_label"].strip().lower():
            print(f"  disagree ({r['kind']}): judge={r['judge']} you={r['my_label']} | "
                  f"{r['question_or_bullet'][:80]}")


if __name__ == "__main__":
    make() if sys.argv[1:] == ["make"] else score()