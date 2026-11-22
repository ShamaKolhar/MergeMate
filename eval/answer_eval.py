"""Evaluation 2 + 4: answer quality, Haiku vs Sonnet, judged blind by Opus."""
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import duckdb
import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from coach import Coach

load_dotenv()
CANDIDATES = ["claude-haiku-5-5", "claude-sonnet-5-5"]
JUDGE_MODEL = "claude-opus-5-5"
PRICES = {"claude-haiku-5-5": (0.10, 0.50),      # USD per million tokens (input, output), Nov 2026
          "claude-sonnet-5-5": (2.00, 10.00),
          "claude-opus-5-5": (4.00, 20.00)}
N_PRS = 20
OUT = Path("eval/results")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "answers.jsonl"

JUDGE_PROMPT = """You are grading two review comments written for the same pull request by two
different assistants. You don't know which assistant wrote which. Judge each bullet point on its
own merits. Do not reward length, formatting or tone; a short correct answer beats a long one.

For EVERY bullet point in each answer, decide:
- grounded: the rule it cites really appears in the PROJECT RULES below (not invented)
- accurate: what it says about this pull request is true, judging from the PULL REQUEST below
- useful: a first-time contributor could act on it

Then say which answer would help the contributor more overall: "A", "B" or "tie".

Reply with JSON only:
{{"A": [{{"bullet": "<first few words>", "grounded": true, "accurate": true, "useful": true}}],
  "B": [ ...same shape... ],
  "preferred": "A" or "B" or "tie",
  "reason": "<one sentence>"}}

PROJECT RULES:
{rules}

PULL REQUEST
Title: {title}
Description:
{body}

Files changed:
{files}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}"""


def text_of(reply):
    c = reply.content
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return c


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def cost(model, usage):
    p_in, p_out = PRICES[model]
    return usage["input_tokens"] / 1e6 * p_in + usage["output_tokens"] / 1e6 * p_out


def pick_prs(n=N_PRS, seed=42):
    """n test-set PRs from n different repos, half merged and half not."""
    df = pd.read_parquet("data/features.parquet", columns=["repo", "number", "split", "merged_30d"])
    test = df[df["split"] == "test"].sample(frac=1, random_state=seed)   # shuffle once
    picked, repos = [], set()
    for want in [1, 0] * n:                                             # alternate merged / not
        if len(picked) == n:
            break
        pool = test[(test["merged_30d"] == want) & (~test["repo"].isin(repos))]
        if pool.empty:
            continue
        row = pool.iloc[0]
        picked.append((row["repo"], int(row["number"]), int(row["merged_30d"])))
        repos.add(row["repo"])
    return picked


def load_pr(con, repo, number):
    title, body = con.execute("SELECT title, body FROM pr_details WHERE repo = ? AND number = ?",
                              [repo, number]).fetchall()[0]
    files = [dict(zip(["path", "additions", "deletions"], r)) for r in con.execute(
        "SELECT path, additions, deletions FROM pr_files WHERE repo = ? AND number = ?",
        [repo, number]).fetchall()]
    return {"title": title, "body": body, "files": files}


def load_done():
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    done = {(r["repo"], r["number"]) for r in load_done()}
    con = duckdb.connect("data/pass2.duckdb", read_only=True)
    coaches = {m: Coach(llm_model=m) for m in CANDIDATES}
    judge = ChatAnthropic(model=JUDGE_MODEL, max_tokens=8000)
    rng = random.Random(7)
    run_cost = 0.0

    for repo, number, merged in pick_prs():
        flip = rng.random() < 0.5            # drawn for every PR, so the order is reproducible
        if (repo, number) in done:
            continue
        pr = load_pr(con, repo, number)

        answers = {}
        for m, coach in coaches.items():
            t0 = time.perf_counter()
            out = coach.review(repo, pr)
            answers[m] = {"advice": out["advice"], "seconds": round(time.perf_counter() - t0, 1),
                          "cost": cost(m, out["usage"])}
            rules = out["rules"]                     # identical for both models

        a_model, b_model = (CANDIDATES[1], CANDIDATES[0]) if flip else (CANDIDATES[0], CANDIDATES[1])
        files = "\n".join(f"- {f['path']} (+{f['additions']} -{f['deletions']})" for f in pr["files"][:30])
        prompt = JUDGE_PROMPT.format(rules="\n\n---\n\n".join(rules), title=pr["title"],
                                     body=pr["body"] or "(empty)", files=files,
                                     answer_a=answers[a_model]["advice"],
                                     answer_b=answers[b_model]["advice"])
        verdict, judge_cost = None, 0.0
        for attempt in range(2):
            reply = judge.invoke(prompt)
            judge_cost += cost(JUDGE_MODEL, reply.usage_metadata)
            verdict = parse_json(text_of(reply))
            if verdict:
                break
        if verdict is None:
            print(f"{repo}#{number}: SKIPPED (judge gave no valid JSON)")
            continue

        record = {"repo": repo, "number": number, "merged_30d": merged,
                  "a_model": a_model, "b_model": b_model, "verdict": verdict,
                  "answers": answers, "judge_cost": judge_cost}
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        pr_cost = judge_cost + sum(a["cost"] for a in answers.values())
        run_cost += pr_cost
        winner = {"A": a_model, "B": b_model}.get(verdict["preferred"], "tie")
        print(f"{repo}#{number}: preferred {winner} | ${pr_cost:.3f} | run so far ${run_cost:.2f}")

    print(f"\nThis run cost ≈ ${run_cost:.2f}. Results in {RESULTS}")