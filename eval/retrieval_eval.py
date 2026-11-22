"""Evaluation 1: does retrieval find the rules a reviewer would check?"""
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from coach import Coach, QUERIES

load_dotenv()
JUDGE_MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.0, 10.0          # USD per million tokens for Sonnet 5.5 (Nov 2026)
GUIDES = Path("data/guidelines")
OUT = Path("eval/results")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "retrieval.csv"
FIELDS = ["repo", "question", "covered", "found", "reason", "passages", "my_covered", "my_found"]

JUDGE_PROMPT = """You are checking a search system. Below are a question, the 3 passages the search
returned, and the project's full CONTRIBUTING file.

Decide:
1. covered: does the full CONTRIBUTING file contain information that answers the question?
2. found: if covered, does at least one of the 3 passages contain that information? (false if not covered)

Reply with JSON only, on one line, with a short reason (under 25 words):
{{"covered": true or false, "found": true or false, "reason": "<short reason>"}}

QUESTION: {question}

PASSAGES:
{passages}

FULL CONTRIBUTING FILE:
{full}"""


def text_of(reply):
    """Plain text from a reply that may contain thinking blocks."""
    c = reply.content
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return c


def parse_verdict(text):
    """The JSON verdict, or None if the reply isn't valid JSON."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def pick_repos(per_group=2, min_chars=3000, seed=42):
    """2 random repos per language group, among those with a substantial CONTRIBUTING file."""
    with open("repos.csv") as f:
        rows = list(csv.DictReader(f))
    by_group = {}
    for r in rows:
        path = GUIDES / f"{r['full_name'].replace('/', '__')}__contributing.md"
        if path.exists() and len(path.read_text(encoding="utf-8")) >= min_chars:
            by_group.setdefault(r["group"], []).append(r["full_name"])
    rng = random.Random(seed)
    return [repo for g in sorted(by_group) for repo in rng.sample(sorted(by_group[g]), per_group)]


def load_done():
    """Cases already judged in an earlier run, so a rerun never pays for them twice."""
    if not RESULTS.exists():
        return []
    with open(RESULTS, encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    done = load_done()
    done_keys = {(r["repo"], r["question"]) for r in done}
    if not RESULTS.exists():
        with open(RESULTS, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    coach = Coach()
    judge = ChatAnthropic(model=JUDGE_MODEL, max_tokens=4000)
    tok_in = tok_out = 0

    for repo in pick_repos():
        full = (GUIDES / f"{repo.replace('/', '__')}__contributing.md").read_text(encoding="utf-8")
        for q in QUERIES:
            if (repo, q) in done_keys:
                continue
            docs = coach.store.similarity_search(q, k=3, filter={"$and": [
                {"repo": repo}, {"doc_type": "contributing"}]})
            passages = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))

            verdict = None
            for attempt in range(2):                      # one retry if the JSON is broken
                reply = judge.invoke(JUDGE_PROMPT.format(question=q, passages=passages, full=full))
                tok_in += reply.usage_metadata["input_tokens"]
                tok_out += reply.usage_metadata["output_tokens"]
                verdict = parse_verdict(text_of(reply))
                if verdict:
                    break
            if verdict is None:
                print(f"{repo:40} SKIPPED (no valid JSON after 2 tries) | {q[:40]}")
                continue

            row = {"repo": repo, "question": q, "covered": verdict["covered"],
                   "found": verdict["found"], "reason": verdict.get("reason", ""),
                   "passages": passages, "my_covered": "", "my_found": ""}
            with open(RESULTS, "a", newline="", encoding="utf-8") as f:   # save each case at once
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"{repo:40} covered={verdict['covered']!s:5} found={verdict['found']!s:5} | {q[:40]}")

    # Summary over everything saved so far (this run and earlier runs)
    results = load_done()
    covered = [r for r in results if r["covered"] == "True"]
    hits = sum(r["found"] == "True" for r in covered)
    print(f"\nCases: {len(results)} | covered: {len(covered)} | "
          f"hit rate: {hits}/{len(covered)} = {hits / max(len(covered), 1):.0%}")
    for q in QUERIES:
        c = [r for r in covered if r["question"] == q]
        if c:
            print(f"  {sum(r['found'] == 'True' for r in c)}/{len(c)}  {q}")
    cost = tok_in / 1e6 * PRICE_IN + tok_out / 1e6 * PRICE_OUT
    print(f"\nThis run: {tok_in:,} tokens in / {tok_out:,} out | cost ≈ ${cost:.2f}")