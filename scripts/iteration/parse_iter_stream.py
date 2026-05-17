#!/usr/bin/env python3
"""
parse_iter_stream.py - 消费 claude CLI 的 stream-json NDJSON 流，输出人类可读日志。

用法：
    由 scripts/iteration/run_iteration.py 启动，并消费 Claude CLI 的
    stream-json 输出。PowerShell 入口只负责调用统一运行器。

输出格式：
    [HH:MM:SS] [INIT] session=abc12345 model=claude-opus-4-7
    [HH:MM:SS] [PLAN] (整段 plan 完整保留)
    [HH:MM:SS] [TEXT] 1 行摘要 / 超长截断
    [HH:MM:SS] [TOOL] ToolName(arg1=..., arg2=...)
    [HH:MM:SS] [RES] tool result 前 200 字符
    [HH:MM:SS] [DONE] cost=$X.XX usage={...}

原始 NDJSON 旁路写到 ~/.claude_iter_raw.jsonl（append 模式，完整 audit trail）。
"""

import json
import os
import sys
from datetime import datetime

RAW_LOG = os.path.expanduser("~/.claude_iter_raw.jsonl")
TEXT_TRUNCATE = 400
TOOL_INPUT_TRUNCATE = 200
TOOL_RESULT_TRUNCATE = 300


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def emit(line: str) -> None:
    print(line, flush=True)


def truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[:n] + "..."


def summarize_tool_input(inp) -> str:
    if not isinstance(inp, dict):
        return truncate(str(inp), TOOL_INPUT_TRUNCATE)
    parts = []
    for k, v in inp.items():
        if isinstance(v, str):
            vs = truncate(v, 80)
        else:
            vs = truncate(json.dumps(v, ensure_ascii=False), 80)
        parts.append(f"{k}={vs}")
    return truncate(", ".join(parts), TOOL_INPUT_TRUNCATE)


def _has_plan_marker(text: str) -> bool:
    """检测 text 任一行是否以 [PLAN] / [PLAN-REVISE] 开头，
    或包含 'CRITIC 报告' / '**CRITIC' 等审稿人评议标记。"""
    if text.startswith("[PLAN]") or text.startswith("[PLAN-REVISE]"):
        return True
    for ln in text.splitlines():
        s = ln.lstrip("* ").strip()
        if s.startswith("[PLAN]") or s.startswith("[PLAN-REVISE]"):
            return True
        if s.startswith("CRITIC 报告") or s.startswith("**CRITIC"):
            return True
    return False


def handle_assistant(obj: dict) -> None:
    msg = obj.get("message", {}) or {}
    for blk in msg.get("content", []) or []:
        if not isinstance(blk, dict):
            continue
        bt = blk.get("type")
        if bt == "text":
            text = (blk.get("text") or "").strip()
            if not text:
                continue
            if _has_plan_marker(text):
                emit(f"[{ts()}] === PLAN/CRITIC ===")
                for ln in text.splitlines():
                    emit(f"  {ln}")
                emit(f"[{ts()}] === /PLAN/CRITIC ===")
            else:
                emit(f"[{ts()}] [TEXT] {truncate(text, TEXT_TRUNCATE)}")
        elif bt == "tool_use":
            tool = blk.get("name", "?")
            inp = blk.get("input", {})
            emit(f"[{ts()}] [TOOL] {tool}({summarize_tool_input(inp)})")
        elif bt == "thinking":
            think = (blk.get("thinking") or "").strip()
            if think:
                emit(f"[{ts()}] [THINK] {truncate(think, 200)}")


def handle_user(obj: dict) -> None:
    msg = obj.get("message", {}) or {}
    for blk in msg.get("content", []) or []:
        if not isinstance(blk, dict):
            continue
        if blk.get("type") != "tool_result":
            continue
        is_err = blk.get("is_error", False)
        content = blk.get("content", "")
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        snippet = truncate(str(content), TOOL_RESULT_TRUNCATE)
        tag = "[RES-ERR]" if is_err else "[RES]"
        emit(f"[{ts()}] {tag} {snippet}")


def handle_system(obj: dict) -> None:
    sub = obj.get("subtype", "")
    if sub == "init":
        sid = (obj.get("session_id") or "?")[:8]
        model = obj.get("model", "?")
        cwd = obj.get("cwd", "?")
        emit(f"[{ts()}] [INIT] session={sid} model={model} cwd={cwd}")
    elif sub == "api_retry":
        attempt = obj.get("attempt", "?")
        max_retries = obj.get("max_retries", "?")
        status = obj.get("error_status", "?")
        delay_ms = obj.get("retry_delay_ms", 0) or 0
        delay_s = float(delay_ms) / 1000.0
        emit(
            f"[{ts()}] [API-RETRY] attempt={attempt}/{max_retries} "
            f"status={status} next_delay={delay_s:.1f}s"
        )


def handle_result(obj: dict) -> None:
    subtype = obj.get("subtype", "")
    is_err = obj.get("is_error", False)
    duration_ms = obj.get("duration_ms", 0)
    num_turns = obj.get("num_turns", 0)
    cost = obj.get("total_cost_usd", 0)
    usage = obj.get("usage", {}) or {}
    in_tok = usage.get("input_tokens", 0)
    cache_r = usage.get("cache_read_input_tokens", 0)
    cache_c = usage.get("cache_creation_input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    duration_s = duration_ms / 1000.0 if duration_ms else 0
    tag = "[ERROR]" if is_err else "[DONE]"
    emit(
        f"[{ts()}] {tag} subtype={subtype} turns={num_turns} "
        f"duration={duration_s:.1f}s cost=${cost:.4f} "
        f"in={in_tok} cache_r={cache_r} cache_c={cache_c} out={out_tok}"
    )


def main() -> int:
    try:
        raw_fp = open(RAW_LOG, "a", encoding="utf-8")
    except OSError as e:
        emit(f"[{ts()}] [PARSER-WARN] cannot open raw log {RAW_LOG}: {e}")
        raw_fp = None

    emit(f"[{ts()}] [PARSER] start (raw NDJSON -> {RAW_LOG})")

    line_count = 0
    for line in sys.stdin:
        line_count += 1
        if raw_fp is not None:
            try:
                raw_fp.write(line)
                raw_fp.flush()
            except OSError:
                pass

        s = line.strip()
        if not s:
            continue

        try:
            obj = json.loads(s)
        except ValueError:
            emit(f"[{ts()}] [RAW] {truncate(s, TEXT_TRUNCATE)}")
            continue

        t = obj.get("type")
        try:
            if t == "system":
                handle_system(obj)
            elif t == "assistant":
                handle_assistant(obj)
            elif t == "user":
                handle_user(obj)
            elif t == "result":
                handle_result(obj)
        except Exception as e:
            emit(f"[{ts()}] [PARSER-ERR] {type(e).__name__}: {e}")

    emit(f"[{ts()}] [PARSER] end (consumed {line_count} lines)")
    if raw_fp is not None:
        raw_fp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
