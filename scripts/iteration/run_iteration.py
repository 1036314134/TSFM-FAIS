#!/usr/bin/env python3
"""
Single-run iteration driver for the paper automation scripts.

The PowerShell entry points call this file. Keeping process management here
avoids platform-specific process-control assumptions.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROMPT_FILE = SCRIPT_DIR / "iterate_prompt.md"
CODEX_PARSER = SCRIPT_DIR / "parse_codex_stream.py"
CLAUDE_PARSER = SCRIPT_DIR / "parse_iter_stream.py"


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def compact_now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def log(message: str) -> None:
    print(message, flush=True)


def configure_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = fp
    sys.stderr = fp


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def positive_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, 0)


def run_capture(argv: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (result.stdout or "").rstrip()
    if output:
        log(output)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(argv)}")
    return result


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError((result.stdout or "").strip() or f"git {' '.join(args)} failed")
    return (result.stdout or "").strip()


def git_dirty() -> str:
    return git_output("status", "--short")


def ensure_git_ready(prefix: str) -> None:
    if run_capture(["git", "remote", "get-url", "origin"]).returncode != 0:
        raise RuntimeError("git origin missing")

    pull = run_capture(["git", "pull", "--rebase", "origin", "main"])
    if pull.returncode == 0:
        return

    log(f"[WARN] git pull --rebase failed; attempting {prefix} dirty-tree stash rescue")
    dirty = git_dirty()
    if not dirty:
        raise RuntimeError("clean tree but git pull --rebase failed")

    stash_msg = f"{prefix}-autopilot-rescue-{compact_now()}"
    run_capture(["git", "stash", "push", "-u", "-m", stash_msg])
    log(f"[RESCUE] stashed dirty tree as \"{stash_msg}\"")
    second_pull = run_capture(["git", "pull", "--rebase", "origin", "main"])
    if second_pull.returncode != 0:
        raise RuntimeError("git pull --rebase still failed after stash rescue")


def rescue_dirty_tree(prefix: str) -> None:
    dirty = git_dirty()
    if not dirty:
        return
    stash_msg = f"{prefix}-watchdog-rescue-{compact_now()}"
    log("[RESCUE] working tree has residual changes; stashing:")
    for line in dirty.splitlines():
        log(f"  {line}")
    run_capture(["git", "stash", "push", "-u", "-m", stash_msg])
    log(f"[RESCUE] saved residuals as \"{stash_msg}\"")


def resolve_binary(env_name: str, default_name: str) -> str:
    configured = os.environ.get(env_name, "").strip()
    if configured:
        return configured
    found = shutil.which(default_name)
    if not found:
        raise RuntimeError(f"{default_name} CLI missing")
    return found


def valid_skill_path(path: Path) -> bool:
    return (path / "README.md").is_file() or (path / "SKILL.md").is_file() or (path / "skills").is_dir()


def find_research_writing_skill() -> Path:
    configured = os.environ.get("RESEARCH_WRITING_SKILL_REPO", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if valid_skill_path(path):
            return path.resolve()
        raise RuntimeError(f"research-writing-skill path is invalid: {path}")

    home = Path.home()
    candidates = [
        REPO_ROOT / "research-writing-skill",
        home / ".codex" / "skills" / "research-writing-skill",
        home / ".codex" / "research-writing-skill",
        home / ".agents" / "skills" / "research-writing",
        home / ".claude" / "skills" / "research-writing",
    ]
    for candidate in candidates:
        if valid_skill_path(candidate):
            return candidate.resolve()
    raise RuntimeError(
        "research-writing-skill not found. Install Norman-bury/research-writing-skill "
        "or set RESEARCH_WRITING_SKILL_REPO to its local path."
    )


def read_project_prompt() -> str:
    if not PROMPT_FILE.is_file():
        raise RuntimeError(f"prompt missing: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def codex_prompt(skill_path: Path) -> str:
    return f"""You are running under Codex CLI, not Claude. Interpret any references to
"claude" in the project prompt as "Codex" for this run. Use the current Codex
agent, full autonomy, and maximal reasoning effort. Keep the same hard
constraints: plan first, edit carefully, compile/check, commit, push, and leave
the working tree clean.

Codex Desktop transcript safety:
- Do not print Codex Desktop UI directives such as ::git-stage, ::git-commit,
  ::git-push, ::git-create-branch, or ::git-create-pr in the final response.
- Report git actions as plain text only. These directive lines can make a
  Windows transcript fail to reopen when cwd contains backslashes.
- If a higher-priority runtime requires one of those directives, use forward
  slashes in the cwd value, for example cwd="{REPO_ROOT.as_posix()}".

Mandatory skill dependency:
- Use Norman-bury/research-writing-skill before any manuscript-writing action.
- Local skill path: {skill_path}
- In Codex, read the corresponding SKILL.md files under the local skill path and follow them. Do not rely on memory of the skill.
- Required modules for this project: using-research-writing, paper-orchestration, evidence-driven-writing, writing-core, latex-output, figures-diagram, peer-review, verification.
- If the skill cannot be loaded or read, stop the iteration before editing and report the missing path in the log. Do not silently proceed as a normal writing prompt.

{read_project_prompt()}"""


def claude_prompt(skill_path: Path) -> str:
    return f"""You are running inside Claude Code for the TSFM-UA-MI paper iteration loop.

Mandatory skill dependency:
- Use Norman-bury/research-writing-skill before any manuscript-writing action.
- Local skill path: {skill_path}
- In Claude Code, explicitly invoke the relevant Skill modules when available; otherwise read the matching SKILL.md files under the local skill path and follow them.
- Required modules for this project: using-research-writing, paper-orchestration, evidence-driven-writing, writing-core, latex-output, figures-diagram, peer-review, verification.
- If the skill cannot be loaded or read, stop the iteration before editing and report the missing path in the log. Do not silently proceed as a normal writing prompt.

{read_project_prompt()}"""


def stream_to_parser(
    stream,
    parser_stdin,
    parser_lock: threading.Lock,
    parser_proc: subprocess.Popen[str],
) -> None:
    try:
        for line in stream:
            wrote = False
            with parser_lock:
                if parser_proc.poll() is None and parser_stdin is not None:
                    try:
                        parser_stdin.write(line)
                        parser_stdin.flush()
                        wrote = True
                    except OSError:
                        wrote = False
            if not wrote:
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        try:
            stream.close()
        except OSError:
            pass


def feed_process_stdin(proc: subprocess.Popen[str], prompt: str) -> None:
    try:
        if proc.stdin is None:
            return
        proc.stdin.write(prompt)
        if not prompt.endswith("\n"):
            proc.stdin.write("\n")
        proc.stdin.close()
    except OSError:
        pass


def process_creation_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def stop_process_tree(proc: subprocess.Popen[str], *, force: bool) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        args = ["taskkill", "/PID", str(proc.pid), "/T"]
        if force:
            args.append("/F")
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not force:
            try:
                proc.terminate()
            except OSError:
                pass
        return

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(proc.pid, sig)
    except OSError:
        try:
            proc.kill() if force else proc.terminate()
        except OSError:
            pass


def wait_with_watchdog(proc: subprocess.Popen[str], soft_seconds: int, grace_seconds: int, name: str) -> int:
    if soft_seconds <= 0:
        return proc.wait()

    try:
        return proc.wait(timeout=soft_seconds)
    except subprocess.TimeoutExpired:
        log(f"[SOFT-TIMEOUT] {name} (PID {proc.pid}) ran {soft_seconds}s; requesting stop")
        stop_process_tree(proc, force=False)

    try:
        return proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        log(f"[HARD-TIMEOUT] {name} still running after {grace_seconds}s; forcing stop")
        stop_process_tree(proc, force=True)
        return proc.wait(timeout=30)


def run_agent(
    *,
    name: str,
    argv: list[str],
    prompt: str,
    parser_path: Path,
    parser_env: dict[str, str],
    agent_env: dict[str, str],
    timeout_soft: int,
    timeout_grace: int,
) -> int:
    if not parser_path.is_file():
        raise RuntimeError(f"parser missing: {parser_path}")

    parser_proc = subprocess.Popen(
        [sys.executable, str(parser_path)],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=parser_env,
    )

    proc = subprocess.Popen(
        argv,
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=agent_env,
        **process_creation_kwargs(),
    )

    assert proc.stdout is not None
    assert proc.stderr is not None
    parser_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=stream_to_parser,
            args=(proc.stdout, parser_proc.stdin, parser_lock, parser_proc),
            daemon=True,
        ),
        threading.Thread(
            target=stream_to_parser,
            args=(proc.stderr, parser_proc.stdin, parser_lock, parser_proc),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    stdin_thread = threading.Thread(target=feed_process_stdin, args=(proc, prompt), daemon=True)
    stdin_thread.start()

    exit_code = wait_with_watchdog(proc, timeout_soft, timeout_grace, name)
    stdin_thread.join(timeout=10)
    for thread in threads:
        thread.join(timeout=10)

    with parser_lock:
        if parser_proc.stdin is not None:
            try:
                parser_proc.stdin.close()
            except OSError:
                pass
    try:
        parser_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        stop_process_tree(parser_proc, force=True)
        parser_proc.wait(timeout=10)

    return exit_code


def secret_label(value: str) -> str:
    if not value:
        return "configured-auth"
    if len(value) <= 12:
        return "***"
    return f"{value[:8]}...{value[-4:]}"


class IterationLock:
    def __init__(self, runner: str) -> None:
        digest = hashlib.sha1(str(REPO_ROOT).encode("utf-8", "replace")).hexdigest()[:12]
        lock_root = Path(os.environ.get("MMDC_ITER_LOCK_ROOT", tempfile.gettempdir()))
        if os.environ.get("MMDC_ITER_LOCK_DIR"):
            self.path = Path(os.environ["MMDC_ITER_LOCK_DIR"])
        elif env_flag("MMDC_ALLOW_PARALLEL_ITERATE"):
            self.path = lock_root / f"mmdc-{runner}-iterate-{digest}.lock"
        else:
            self.path = lock_root / f"mmdc-iterate-{digest}.lock"
        self.acquired = False

    def __enter__(self) -> "IterationLock":
        self._clear_stale_lock()
        try:
            self.path.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            raise RuntimeError(
                f"another iterate process appears to be running: {self.path}. "
                "Set MMDC_ALLOW_PARALLEL_ITERATE=1 only for isolated testing."
            )
        self.acquired = True
        (self.path / "info.txt").write_text(
            f"pid={os.getpid()}\nrepo={REPO_ROOT}\nstarted={iso_now()}\n",
            encoding="utf-8",
        )
        atexit.register(self.release)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)
            self.acquired = False

    def _clear_stale_lock(self) -> None:
        if not self.path.exists():
            return
        info = self.path / "info.txt"
        if not info.is_file():
            self._remove_lock_path()
            return
        pid = None
        for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("pid="):
                try:
                    pid = int(line.split("=", 1)[1])
                except ValueError:
                    pid = None
        if pid and process_is_alive(pid):
            return
        self._remove_lock_path()

    def _remove_lock_path(self) -> None:
        if self.path.is_dir():
            shutil.rmtree(self.path, ignore_errors=True)
            return
        try:
            self.path.unlink()
        except OSError:
            pass


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("NO_COLOR", "1")
    return env


def run_codex() -> int:
    log_path = Path(os.environ.get("CODEX_ITER_LOG", Path.home() / ".codex_iter.log")).expanduser()
    configure_log(log_path)

    env = base_env()
    codex_bin = resolve_binary("CODEX_BIN", "codex")
    model = os.environ.get("CODEX_MODEL", "gpt-5.5")
    effort = os.environ.get("CODEX_REASONING_EFFORT", "xhigh")
    raw_max = os.environ.get("CODEX_RAW_LOG_MAX_BYTES", "52428800")
    soft = positive_int(os.environ.get("CODEX_TIMEOUT_SOFT"), 0)
    grace = positive_int(os.environ.get("CODEX_TIMEOUT_GRACE"), 120)

    log(f"==== iterate_once_codex start {iso_now()} ====")
    log(f"[info] model={model} reasoning_effort={effort} repo={REPO_ROOT}")

    skill_path = find_research_writing_skill()
    log(f"[info] research-writing-skill={skill_path}")

    with IterationLock("codex"):
        ensure_git_ready("codex")
        commit_before = git_output("rev-parse", "HEAD")

        parser_env = env.copy()
        parser_env["CODEX_PARSER_WRITE_RAW"] = "1"
        parser_env["CODEX_RAW_LOG_MAX_BYTES"] = raw_max

        argv = [
            codex_bin,
            "--search",
            "-a",
            "never",
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--sandbox",
            "danger-full-access",
            "--cd",
            str(REPO_ROOT),
            "--model",
            model,
            "-c",
            f'model_reasoning_effort="{effort}"',
            "-c",
            'shell_environment_policy.inherit="all"',
            "-",
        ]
        exit_code = run_agent(
            name="codex",
            argv=argv,
            prompt=codex_prompt(skill_path),
            parser_path=CODEX_PARSER,
            parser_env=parser_env,
            agent_env=env,
            timeout_soft=soft,
            timeout_grace=grace,
        )

        commit_after = git_output("rev-parse", "HEAD")
        if exit_code == 0 and commit_before != commit_after:
            log(f"[ok] Codex completed with new commit {commit_after}")
        elif exit_code == 0:
            log("[WARN] Codex exited 0 but created no new commit")
        else:
            log(f"[WARN] Codex exited with code {exit_code}")
        rescue_dirty_tree("codex")

    log(f"==== iterate_once_codex end {iso_now()} ====")
    log("")
    return 0


def claude_keys() -> list[str]:
    primary = os.environ.get("ANTHROPIC_API_KEY", "")
    if not primary:
        return [""]
    keys = [primary]
    for name in ("ANTHROPIC_API_KEY_BACKUP1", "ANTHROPIC_API_KEY_BACKUP2", "ANTHROPIC_API_KEY_BACKUP3"):
        value = os.environ.get(name, "")
        if value:
            keys.append(value)
    random.shuffle(keys)
    return keys


def run_claude() -> int:
    log_path = Path(os.environ.get("CLAUDE_ITER_LOG", Path.home() / ".claude_iter.log")).expanduser()
    configure_log(log_path)

    env = base_env()
    claude_bin = resolve_binary("CLAUDE_BIN", "claude")
    model = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7[1m]")
    betas = os.environ.get("CLAUDE_BETAS", "context-1m-2025-08-07")
    effort = os.environ.get("CLAUDE_EFFORT", "max")
    soft = positive_int(os.environ.get("CLAUDE_TIMEOUT_SOFT"), 0)
    grace = positive_int(os.environ.get("CLAUDE_TIMEOUT_GRACE"), 120)

    log(f"==== iterate_once start {iso_now()} ====")
    log(f"[info] model={model} effort={effort} repo={REPO_ROOT}")

    skill_path = find_research_writing_skill()
    log(f"[info] research-writing-skill={skill_path}")
    keys = claude_keys()
    if keys == [""]:
        log("[info] ANTHROPIC_API_KEY is not set; relying on Claude CLI login or local configuration")
    else:
        log(f"[info] available key count: {len(keys)}; randomized order: {' '.join(secret_label(k) for k in keys)}")

    with IterationLock("claude"):
        ensure_git_ready("claude")
        commit_before = git_output("rev-parse", "HEAD")
        final_exit = 99

        for index, key in enumerate(keys, start=1):
            log(f"[try {index}/{len(keys)}] using key {secret_label(key)}")
            agent_env = env.copy()
            if key:
                agent_env["ANTHROPIC_API_KEY"] = key
            argv = [
                claude_bin,
                "--dangerously-skip-permissions",
                "--model",
                model,
                "--betas",
                betas,
                "--effort",
                effort,
                "--output-format",
                "stream-json",
                "--verbose",
                "-p",
            ]
            final_exit = run_agent(
                name="claude",
                argv=argv,
                prompt=claude_prompt(skill_path),
                parser_path=CLAUDE_PARSER,
                parser_env=env,
                agent_env=agent_env,
                timeout_soft=soft,
                timeout_grace=grace,
            )
            commit_after = git_output("rev-parse", "HEAD")
            if final_exit == 0 and commit_before != commit_after:
                log(f"[ok] key {secret_label(key)} succeeded; new commit {commit_after}")
                break
            log(f"[fail] key {secret_label(key)} exited {final_exit} and created no new commit")

        if final_exit != 0 or commit_before == git_output("rev-parse", "HEAD"):
            log(f"[WARN] all {len(keys)} keys failed or created no new commit; next scheduled run can continue")
        rescue_dirty_tree("claude")

    log(f"==== iterate_once end {iso_now()} ====")
    log("")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one cross-platform TSFM-UA-MI iteration.")
    parser.add_argument("--runner", choices=("codex", "claude"), required=True)
    args = parser.parse_args(argv)

    try:
        if args.runner == "codex":
            return run_codex()
        return run_claude()
    except Exception as exc:
        log_path_var = "CODEX_ITER_LOG" if args.runner == "codex" else "CLAUDE_ITER_LOG"
        fallback_log = Path(os.environ.get(log_path_var, Path.home() / f".{args.runner}_iter.log")).expanduser()
        try:
            configure_log(fallback_log)
        except Exception:
            pass
        log(f"[FATAL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
