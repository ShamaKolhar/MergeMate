"""HTTP API for MergeMate: send a PR link, get the score, explanation and coach advice."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import MergeMate, check_url
from quota import take_quota

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
state = {}


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
    try:
        url = check_url(req.url)                          # reject bad links before spending quota
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    use_coach = req.coach and take_quota(req.model)
    try:
        result = state["mergemate"].analyze(url, with_coach=use_coach, llm_model=MODELS[req.model])
    except ValueError as e:                               # PR not found
        raise HTTPException(status_code=400, detail=str(e))

    if req.coach and not use_coach:
        result["warnings"].append(f"Today's limit for {req.model} coach reviews has been reached, so only "
                                  "the score is shown. Try again tomorrow, or choose the other model.")
    result["coach_model"] = req.model if use_coach else None
    return result