"""ExecutionLogger: 结构化执行日志 (JSON lines)。"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import IO, Any, Dict, Optional

from interfaces import SkillResult


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def make_record(
    skill_id: str,
    status: str,
    steps: int,
    message: str = "",
    failure_type: Optional[str] = None,
    *,
    event: str = "skill_result",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    record: Dict[str, Any] = {
        "timestamp": now,
        "iso": _iso(now),
        "event": event,
        "skill_id": skill_id,
        "status": status,
        "steps": steps,
        "message": message,
        "failure_type": failure_type,
    }
    if extra:
        record.update(extra)
    return record


class ExecutionLogger:
    """把每次技能执行写成一整行 JSON。

    输出目标优先级: stream > path > sys.stdout。
    """

    def __init__(
        self,
        path: Optional[str] = None,
        stream: Optional[IO[str]] = None,
        flush: bool = True,
    ) -> None:
        self.flush_enabled = flush
        self._own_stream = False
        if stream is not None:
            self._stream = stream
        elif path is not None:
            self._stream = open(path, "a", encoding="utf-8")
            self._own_stream = True
        else:
            self._stream = sys.stdout

    def log_result(
        self,
        result: SkillResult,
        *,
        event: str = "skill_result",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        failure_type = result.failure.type.value if result.failure is not None else None
        record = make_record(
            result.skill_id,
            result.status.value,
            result.steps,
            message=result.message,
            failure_type=failure_type,
            event=event,
            extra=extra,
        )
        self.log(record)
        return record

    def log(self, record: Dict[str, Any]) -> Dict[str, Any]:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        self._stream.write(line + "\n")
        if self.flush_enabled:
            self._stream.flush()
        return record

    def close(self) -> None:
        if self._own_stream:
            self._stream.close()
