from __future__ import annotations

from collections import deque
from typing import Any


class RequestScheduler:
    def __init__(self, *, runner_slots: int) -> None:
        self._runner_slots = max(1, int(runner_slots))
        self.queue: deque[str] = deque()

    def enqueue(self, rec: Any) -> None:
        self.queue.append(str(rec.request_id))

    def depth(self) -> int:
        return int(len(self.queue))

    def queue_position(self, rec: Any) -> int | None:
        if str(rec.state) != "queued":
            return None
        try:
            return int(list(self.queue).index(str(rec.request_id)) + 1)
        except ValueError:
            return None

    def remove(self, request_id: str) -> None:
        try:
            self.queue.remove(str(request_id))
        except ValueError:
            return

    def dequeue_next(self, *, records: dict[str, Any]) -> str | None:
        while self.queue:
            rid = self.queue.popleft()
            rec = records.get(rid)
            if rec is None or rec.state != "queued":
                continue
            return str(rid)
        return None
