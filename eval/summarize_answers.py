"""Turn eval/results/answers.jsonl into the scores table for Haiku vs Sonnet."""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

records = [json.loads(l) for l in Path("eval/results/answers.jsonl").read_text(encoding="utf-8").splitlines() if l]
models = ["claude-haiku-5-5", "claude-sonnet-5-5"]

# ---- Per-bullet scores, per model ----
rows = []
for r in records:
    for side in ["A", "B"]:
        model = r[f"{side.lower()}_model"]
        for b in r["verdict"].get(side, []):
            rows.append({"model": model, "grounded": bool(b.get("grounded")),
                         "accurate": bool(b.get("accurate")), "useful": bool(b.get("useful"))})
bullets = pd.DataFrame(rows)
table = bullets.groupby("model").agg(bullets=("grounded", "size"),
                                     grounded=("grounded", "mean"),
                                     accurate=("accurate", "mean"),
                                     useful=("useful", "mean"))
table["invented_rules"] = 1 - table["grounded"]

# ---- Cost and speed, per model ----
for m in models:
    table.loc[m, "cost_per_pr"] = sum(r["answers"][m]["cost"] for r in records) / len(records)
    table.loc[m, "seconds_per_pr"] = sum(r["answers"][m]["seconds"] for r in records) / len(records)

pd.set_option("display.width", 120)
print(f"PRs judged: {len(records)}\n")
print(table.round(3).to_string())

# ---- Overall preference ----
wins = {m: 0 for m in models}
ties = 0
a_wins = b_wins = 0
for r in records:
    p = r["verdict"].get("preferred")
    if p == "A":
        wins[r["a_model"]] += 1; a_wins += 1
    elif p == "B":
        wins[r["b_model"]] += 1; b_wins += 1
    else:
        ties += 1
print(f"\nPreferred: Haiku {wins[models[0]]}, Sonnet {wins[models[1]]}, tie {ties}")

decided = wins[models[0]] + wins[models[1]]
test = binomtest(wins[models[1]], decided, p=0.5)
print(f"Sign test (Sonnet vs Haiku, ties dropped): p = {test.pvalue:.4f}")

# ---- Position bias check: does slot 'A' win more than half the time? ----
pos = binomtest(a_wins, a_wins + b_wins, p=0.5)
print(f"Position check: answer A chosen {a_wins}, answer B chosen {b_wins} (p = {pos.pvalue:.3f})")

judge_cost = sum(r["judge_cost"] for r in records)
print(f"\nJudge cost total: ${judge_cost:.2f}")