"""Daily limits on coach reviews, shared by the API and the Streamlit app, to protect the API credits."""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

DAILY_CAPS = {"haiku": int(os.getenv("HAIKU_DAILY_CAP", "200")),
              "sonnet": int(os.getenv("SONNET_DAILY_CAP", "20"))}
USAGE_FILE = Path(os.getenv("USAGE_FILE", "data/usage.json"))
_lock = threading.Lock()


def _load_today():
    today = datetime.now(timezone.utc).date().isoformat()
    usage = json.loads(USAGE_FILE.read_text()) if USAGE_FILE.exists() else {}
    if usage.get("day") != today:                          # new day (UTC): reset the counters
        usage = {"day": today, "haiku": 0, "sonnet": 0}
    return usage


def take_quota(model):
    """Count one coach review against today's limit. False if the limit is already reached."""
    with _lock:                                            # two requests at once can't both slip through
        usage = _load_today()
        if usage[model] >= DAILY_CAPS[model]:
            return False
        usage[model] += 1
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps(usage))
        return True


def remaining(model):
    with _lock:
        return max(DAILY_CAPS[model] - _load_today()[model], 0)