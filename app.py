"""MergeMate web app: paste a pull request link, see its chance of being merged and what to fix."""
import csv
import sys

sys.path.insert(0, "src")
import streamlit as st

from pipeline import MergeMate, check_url
from quota import remaining, take_quota

MODELS = {"Haiku (fast, default)": ("haiku", "claude-haiku-5-5"),
          "Sonnet (more accurate, limited per day)": ("sonnet", "claude-sonnet-5-5")}
BASELINE = 0.375          # share of newcomer PRs merged within 30 days in 2026

# What to tell the contributor when a fixable feature is pulling the score down
IMPROVE_TEXT = {
    "Lines added": "The PR is large. Smaller pull requests are much more likely to be merged; consider splitting it.",
    "Files changed": "It touches many files. A narrower change is easier to review.",
    "Top-level folders touched": "It spreads across several parts of the project. Keep each PR to one area.",
    "Touches tests": "It doesn't add or update tests.",
    "Touches CI config": "It changes CI configuration. Consider moving that into a separate PR.",
    "Touches dependencies": "It changes dependencies. Consider moving that into a separate PR.",
    "Links an issue": "It doesn't link an issue with a closing keyword (for example `Closes #123`). "
                      "The model only recognises that short form, not full issue links.",
    "Has a checklist": "The PR template's checklist isn't ticked.",
    "Title length": "The title is very short or very long. Aim for a clear one-line summary.",
    "Conventional-commit title": "The title doesn't use the `type: summary` format (for example `fix: ...`).",
    "Description length": "The description is short. Explain what changed and why.",
    "Empty description": "The description is empty. Explain what changed and why.",
}

st.set_page_config(page_title="MergeMate", page_icon="🔀", layout="centered")


@st.cache_resource(show_spinner="Loading the model and the project rules (first visit only)…")
def load():
    return MergeMate()


@st.cache_data
def covered_repos():
    with open("repos.csv") as f:
        return sorted(r["full_name"] for r in csv.DictReader(f))


st.title("MergeMate")
st.write("Will a first-time contributor's pull request be merged within 30 days, and what would improve "
         "its chances? Paste a link to a pull request in one of the supported projects.")

with st.expander(f"Supported projects ({len(covered_repos())})"):
    st.write(", ".join(covered_repos()))

url = st.text_input("Pull request link", placeholder="e.g. https://github.com/JabRef/jabref/pull/17894")
col1, col2 = st.columns([2, 1])
with col1:
    choice = st.radio("Coach model", list(MODELS), horizontal=False)
with col2:
    want_coach = st.toggle("Include the coach's advice", value=True)

key, model_id = MODELS[choice]

if st.button("Analyse", type="primary", disabled=not url):
    try:
        url = check_url(url)            # reject bad links before spending any quota or API calls
    except ValueError as e:
        st.error(str(e))
        st.stop()
    mm = load()
    use_coach = want_coach and take_quota(key)
    with st.spinner("Fetching the pull request and scoring it…"):
        try:
            result = mm.analyze(url, with_coach=use_coach, llm_model=model_id)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.subheader(f"{result['repo']}#{result['number']}: {result['title']}")
    for w in result["warnings"]:
        st.warning(w)
    if want_coach and not use_coach:
        st.warning(f"Today's limit for {key} reviews has been reached, so only the score is shown. "
                   "Try again tomorrow, or choose the other model.")

    p = result["probability"]
    diff = p - BASELINE
    st.metric("Chance of being merged within 30 days", f"{p:.0%}",
              delta=f"{diff * 100:+.0f} points vs the 2026 average for newcomer PRs ({BASELINE:.0%})")
    st.caption("A rough estimate, not a verdict. Given one merged and one unmerged PR, the model ranks the merged "
               "one higher about 71% of the time, and it's least precise for very weak and very strong PRs.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### What you can improve")
        if result["improve"]:
            for item in result["improve"]:
                st.markdown(f"- {IMPROVE_TEXT.get(item['label'], item['label'])}")
        else:
            st.markdown("Nothing the model sees in the PR itself is pulling the score down.")
    with right:
        st.markdown("#### Already helping")
        if result["strengths"]:
            for item in result["strengths"]:
                st.markdown(f"- {item['label']}")
        else:
            st.markdown("—")

    st.markdown("#### Context (outside your control)")
    for item in result["context"]:
        effect = "helps" if item["push"] > 0 else "lowers the chance"
        st.markdown(f"- {item['label']}: {effect}")

    if use_coach and result.get("coach"):
        st.markdown(f"#### Checked against {result['repo']}'s own contribution rules")
        st.markdown(result["coach"])
        st.caption(f"Written by Claude {key.capitalize()} from the project's CONTRIBUTING file and PR template. "
                   "It only gives advice; please write your PR yourself.")
    if want_coach:
        st.caption(f"{remaining(key)} {key} reviews left today.")

st.divider()
st.caption("MergeMate is a portfolio project. Source, method and evaluation: "
           "[github.com/ShamaKolhar/MergeMate](https://github.com/ShamaKolhar/MergeMate)")