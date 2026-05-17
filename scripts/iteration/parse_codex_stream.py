#!/usr/bin/env python3
"""
parse_codex_stream.py - consume `codex exec --json` JSONL and print readable logs.

The Codex JSON event schema may evolve, so this parser is intentionally
defensive: it stores the full raw stream and extracts common text, tool,
command, error, and completion fields when present.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any


RAW_LOG = os.path.expanduser("~/.codex_iter_raw.jsonl")
TEXT_TRUNCATE = 700
OUTPUT_TRUNCATE = 1200
DEFAULT_RAW_MAX_BYTES = 50 * 1024 * 1024
DESKTOP_GIT_DIRECTIVE_PREFIXES = (
    "::git-stage{",
    "::git-commit{",
    "::git-push{",
    "::git-create-branch{",
    "::git-create-pr{",
)


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def emit(line: str) -> None:
    print(line, flush=True)


def truncate(value: Any, n: int = TEXT_TRUNCATE) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def find_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = [find_text(v) for v in value]
        joined = "\n".join(p for p in parts if p)
        return joined.strip() or None
    if isinstance(value, dict):
        for key in ("text", "message", "content", "delta", "summary", "result"):
            if key in value:
                found = find_text(value[key])
                if found:
                    return found
    return None


def has_plan_marker(text: str) -> bool:
    if text.startswith("[PLAN]") or text.startswith("[PLAN-REVISE]"):
        return True
    for line in text.splitlines():
        stripped = line.lstrip("* ").strip()
        if stripped.startswith("[PLAN]") or stripped.startswith("[PLAN-REVISE]"):
            return True
        if stripped.startswith("CRITIC 报告") or stripped.startswith("**CRITIC"):
            return True
    return False


def sanitize_desktop_directives(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if any(stripped.startswith(prefix) for prefix in DESKTOP_GIT_DIRECTIVE_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def emit_text(text: str) -> None:
    text = sanitize_desktop_directives(text)
    if not text:
        emit(f"[{ts()}] [TEXT] [suppressed Codex Desktop git directive]")
        return
    if has_plan_marker(text):
        emit(f"[{ts()}] === PLAN/CRITIC ===")
        for line in text.splitlines():
            emit(f"  {line}")
        emit(f"[{ts()}] === /PLAN/CRITIC ===")
    else:
        emit(f"[{ts()}] [TEXT] {truncate(text)}")


def summarize_command(item: dict[str, Any], completed: bool) -> None:
    command = item.get("command") or item.get("cmd") or "?"
    status = item.get("status") or "?"
    exit_code = item.get("exit_code")
    output = item.get("aggregated_output") or ""
    if not completed:
        emit(f"[{ts()}] [CMD] {truncate(command, 420)}")
        return

    tag = "[CMD-OK]" if exit_code == 0 else "[CMD-FAIL]"
    emit(
        f"[{ts()}] {tag} exit={exit_code} status={status} "
        f"cmd={truncate(command, 360)}"
    )
    if output:
        emit(f"[{ts()}] [OUT] {truncate(output, OUTPUT_TRUNCATE)}")


def summarize_item_event(event_type: str, item: dict[str, Any]) -> bool:
    item_type = str(item.get("type") or "")
    completed = event_type == "item.completed"

    if item_type == "command_execution":
        summarize_command(item, completed=completed)
        return True

    if item_type == "agent_message":
        text = item.get("text") or find_text(item)
        if text:
            emit_text(str(text).strip())
            return True

    if item_type == "collab_tool_call":
        tool = item.get("tool") or "collab_tool"
        status = item.get("status") or "?"
        receivers = item.get("receiver_thread_ids") or []
        emit(
            f"[{ts()}] [TOOL] {tool} status={status} "
            f"receivers={truncate(receivers, 160)}"
        )
        return True

    if item_type:
        label = "[ITEM-DONE]" if completed else "[ITEM]"
        emit(f"[{ts()}] {label} {item_type} {truncate(item, 500)}")
        return True

    return False


def summarize_event(obj: dict[str, Any]) -> None:
    event_type = str(obj.get("type") or obj.get("event") or "?")

    item = obj.get("item")
    if isinstance(item, dict) and summarize_item_event(event_type, item):
        return

    if "error" in event_type.lower() or obj.get("error"):
        message = obj.get("message") or obj.get("error") or obj
        emit(f"[{ts()}] [ERROR] {event_type} {truncate(message)}")
        return

    if "started" in event_type.lower():
        emit(f"[{ts()}] [START] {event_type}")
        return

    if "completed" in event_type.lower() or "done" in event_type.lower():
        usage = obj.get("usage") or obj.get("token_usage") or obj.get("metrics") or ""
        suffix = f" usage={truncate(usage, 220)}" if usage else ""
        emit(f"[{ts()}] [DONE] {event_type}{suffix}")
        return

    text = find_text(obj)
    if text and event_type not in {"?", "event"}:
        emit_text(text)
        return

    emit(f"[{ts()}] [EVENT] {truncate(event_type, 220)}")


class RawLogWriter:
    def __init__(self, path: str, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.fp = None
        self.bytes_written = 0

    def open(self) -> None:
        try:
            existing_size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
            if existing_size > self.max_bytes:
                with open(self.path, "w", encoding="utf-8"):
                    pass
                emit(
                    f"[{ts()}] [PARSER-WARN] raw log exceeded cap "
                    f"({existing_size} > {self.max_bytes}); truncated {self.path}"
                )
                existing_size = 0
            self.fp = open(self.path, "a", encoding="utf-8")
            self.bytes_written = existing_size
        except OSError as exc:
            emit(f"[{ts()}] [PARSER-WARN] cannot open raw log {self.path}: {exc}")
            self.fp = None

    def write(self, line: str) -> None:
        if self.fp is None:
            return
        encoded_len = len(line.encode("utf-8", "replace"))
        if encoded_len > self.max_bytes:
            emit(
                f"[{ts()}] [PARSER-WARN] skip oversized raw event "
                f"({encoded_len} > {self.max_bytes})"
            )
            return
        if self.bytes_written + encoded_len > self.max_bytes:
            try:
                self.fp.close()
                with open(self.path, "w", encoding="utf-8"):
                    pass
                self.fp = open(self.path, "a", encoding="utf-8")
                self.bytes_written = 0
                emit(
                    f"[{ts()}] [PARSER-WARN] raw log hit cap "
                    f"({self.max_bytes}); truncated {self.path}"
                )
            except OSError as exc:
                emit(f"[{ts()}] [PARSER-WARN] cannot truncate raw log {self.path}: {exc}")
                self.fp = None
                return
        try:
            self.fp.write(line)
            self.fp.flush()
            self.bytes_written += encoded_len
        except OSError:
            pass

    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()


def raw_logging_enabled() -> bool:
    value = os.environ.get("CODEX_PARSER_WRITE_RAW", "").lower()
    return value in {"1", "true", "yes", "on"}


def raw_max_bytes() -> int:
    value = os.environ.get("CODEX_RAW_LOG_MAX_BYTES", "")
    if not value:
        return DEFAULT_RAW_MAX_BYTES
    try:
        return max(1024 * 1024, int(value))
    except ValueError:
        emit(f"[{ts()}] [PARSER-WARN] invalid CODEX_RAW_LOG_MAX_BYTES={value!r}; using default")
        return DEFAULT_RAW_MAX_BYTES


def main() -> int:
    raw_writer = None
    if raw_logging_enabled():
        raw_writer = RawLogWriter(RAW_LOG, raw_max_bytes())
        raw_writer.open()
        emit(f"[{ts()}] [PARSER] start (raw JSONL -> {RAW_LOG}, cap={raw_writer.max_bytes})")
    else:
        emit(f"[{ts()}] [PARSER] start (raw JSONL disabled; set CODEX_PARSER_WRITE_RAW=1 to enable)")
    line_count = 0

    for line in sys.stdin:
        line_count += 1
        if raw_writer is not None:
            raw_writer.write(line)

        s = line.strip()
        if not s:
            continue

        try:
            obj = json.loads(s)
        except ValueError:
            emit(f"[{ts()}] [RAW] {truncate(s)}")
            continue

        if isinstance(obj, dict):
            try:
                summarize_event(obj)
            except Exception as exc:
                emit(f"[{ts()}] [PARSER-ERR] {type(exc).__name__}: {exc}")
        else:
            emit(f"[{ts()}] [RAW] {truncate(obj)}")

    emit(f"[{ts()}] [PARSER] end (consumed {line_count} lines)")
    if raw_writer is not None:
        raw_writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
