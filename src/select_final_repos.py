import random
import pandas as pd

random.seed(42)
PER_GROUP = 10

df = pd.read_csv("data/candidates_rules_5_6.csv").drop_duplicates("full_name", keep="last")

# Keep repos passing "Accepts outside contributors" and "Merges through GitHub PRs", with at least 10 outsider PRs in the sample
eligible = df[
    (df["overall_merge_rate"] > 0.20)
    & (df["outsider_merge_rate"] > 0.10)
    & (df["n_outsiders"] >= 10)
].copy()

# Combine JavaScript and TypeScript into one group
eligible["group"] = eligible["language"].replace({"javascript": "js_ts", "typescript": "js_ts"})

print("Eligible per group:")
print(eligible.groupby("group").size())

# Random sample of 10 per group
final = []
for group, repos in eligible.groupby("group"):
    names = sorted(repos["full_name"])            # sort first so the seed gives the same result every time
    chosen = random.sample(names, min(PER_GROUP, len(names)))
    final.append(repos[repos["full_name"].isin(chosen)])

final = pd.concat(final).sort_values(["group", "full_name"])

final.to_csv("data/repos_final.csv", index=False)
final.to_csv("repos.csv", index=False)           # a copy outside data/, so it gets committed to GitHub

print("\nFinal repos per group:")
print(final.groupby("group").size())
print("\nTotal:", len(final))
print(final[["group", "full_name", "pr_count", "outsider_merge_rate"]].to_string(index=False))