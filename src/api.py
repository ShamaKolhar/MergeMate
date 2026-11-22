"""HTTP API for MergeMate: send a PR link, get the score, explanation and coach advice."""
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import MergeMate

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
# Daily limits on coach reviews, to protect the API credits (change with environment variables)
DAILY_CAPS = {"haiku": int(os.getenv("HAIKU_DAILY_CAP", "200")),
              "sonnet": int(os.getenv("SONNET_DAILY_CAP", "20"))}
USAGE_FILE = Path(os.getenv("USAGE_FILE", "data/usage.json"))
_usage_lock = threading.Lock()
state = {}


def take_quota(model):
    """Count one coach review against today's limit. False if the limit is already reached."""
    with _usage_lock:                                     # two requests at once can't both slip through
        today = datetime.now(timezone.utc).date().isoformat()
        usage = json.loads(USAGE_FILE.read_text()) if USAGE_FILE.exists() else {}
        if usage.get("day") != today:                     # new day (UTC): reset the counters
            usage = {"day": today, "haiku": 0, "sonnet": 0}
        if usage[model] >= DAILY_CAPS[model]:
            return False
        usage[model] += 1
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps(usage))
        return True


@asynccontextmanager
async def lifespan(app):
    state["mergemate"] = MergeMate()                      # load models once, at startup
    yield


app = FastAPI(title="MergeMate API",
              description="Will a newcomer's first pull request be merged within 30 days, and what should they fix?",
              lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    url: str = Field(..., examples=["https://github.com/JabRef/jabref/pull/17894"])
    coach: bool = Field(True, description="Include the coach's advice (uses the Claude API)")
    model: str = Field("haiku", pattern="^(haiku|sonnet)$", description="Coach model: haiku or sonnet")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    use_coach = req.coach and take_quota(req.model)
    try:
        result = state["mergemate"].analyze(req.url, with_coach=use_coach, llm_model=MODELS[req.model])
    except ValueError as e:                               # bad link, repo not covered, PR not found
        raise HTTPException(status_code=400, detail=str(e))

    if req.coach and not use_coach:
        result["warnings"].append(f"Today's limit for {req.model} coach reviews has been reached, so only "
                                  "the score is shown. Try again tomorrow, or choose the other model.")
    result["coach_model"] = req.model if use_coach else None
    return result