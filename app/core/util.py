from __future__ import annotations

from datetime import datetime
from datetime import timezone
import re
import time


def iso_utc(ts: float | None = None) -> str:
    value = time.time() if ts is None else float(ts)
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc_unix(value: str | None) -> float | None:
    try:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def safe_token(value: str, *, fallback: str = "item") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text[:120] or fallback

