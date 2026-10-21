"""Teacher runner: asks the big model (via the `claude` CLI) a question and
gets back JSON that matches a schema. Answers are cached on disk so a re-run
never pays twice, and failures are retried so a rate limit doesn't lose work."""

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

from dev.distill.paths import TEACHER_CACHE

RETRY_DELAYS_S = (30, 120)


class TeacherError(Exception):
    """The teacher could not produce a structured answer."""


class TeacherUnavailable(TeacherError):
    """The `claude` program is missing; retrying cannot help."""


LEAN_SYSTEM_PROMPT = ("You are a careful expert assistant. Read any image files you are given with the Read tool, "
                      "then answer only with the requested structured output.")


def _empty_mcp_config() -> str:
    """Path of an MCP config with no servers (for --strict-mcp-config)."""
    path = Path(TEACHER_CACHE).parent / "empty_mcp.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"mcpServers": {}}')
    return str(path)


class ClaudeCodeTeacher:
    def __init__(self, model: str = "claude-opus-5-5", timeout_s: float = 600,
                 cache_dir: Path = TEACHER_CACHE, run=subprocess.run, sleep=time.sleep,
                 lean: bool = True, effort: str | None = None):
        """`lean` drops Claude Code's own context from every call: no MCP servers
        (an empty strict config), no skills, a one-line system prompt, and only
        the Read tool when images are attached. Without it each call carries
        ~200K tokens of connector tool definitions. `effort` maps to --effort."""
        self.model = model
        self.lean = lean
        self.effort = effort
        self.name = f"claude-code:{model}"
        self.timeout_s = timeout_s
        self.cache_dir = Path(cache_dir)
        self._run = run
        self._sleep = sleep
        self.calls: list[dict] = []  # one {cost_usd, cached} per ask()

    def ask(self, prompt: str, schema: dict, images: Sequence[Path] = (),
            validate: Callable[[dict], None] | None = None) -> dict:
        """`validate` raises on a bad answer; that counts as a failed attempt (retried, never cached)."""
        paths = [Path(p).resolve() for p in images]
        key = self._cache_key(prompt, schema, paths)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            self.calls.append({"cost_usd": 0.0, "cached": True})
            return json.loads(cache_file.read_text())

        full_prompt = prompt
        if paths:
            full_prompt += "\n\nRead these image files:\n" + "\n".join(str(p) for p in paths)
        cmd = ["claude", "-p", full_prompt, "--model", self.model,
               "--allowedTools", "Read", "--json-schema", json.dumps(schema),
               "--output-format", "json", "--max-turns", "20"]
        if self.lean:
            cmd += ["--tools", "Read" if paths else "", "--strict-mcp-config", "--mcp-config", _empty_mcp_config(),
                    "--disable-slash-commands", "--system-prompt", LEAN_SYSTEM_PROMPT]
        if self.effort:
            cmd += ["--effort", self.effort]

        for attempt in range(len(RETRY_DELAYS_S) + 1):
            try:
                answer, cost = self._call(cmd)
                if validate is not None:
                    try:
                        validate(answer)
                    except Exception as exc:
                        raise TeacherError(f"answer failed validation: {exc}") from exc
                break
            except TeacherUnavailable:
                raise
            except TeacherError:
                if attempt == len(RETRY_DELAYS_S):
                    raise
                self._sleep(RETRY_DELAYS_S[attempt])

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(answer))
        self.calls.append({"cost_usd": cost, "cached": False})
        return answer

    def _call(self, cmd: list) -> tuple[dict, float]:
        try:
            done = self._run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        except FileNotFoundError as exc:
            raise TeacherUnavailable("`claude` is not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise TeacherError(f"claude timed out after {self.timeout_s}s") from exc
        if done.returncode != 0:
            raise TeacherError(f"claude exited {done.returncode}: {done.stderr.strip()[:300]}")
        try:
            result = json.loads(done.stdout)
        except json.JSONDecodeError as exc:
            raise TeacherError("claude printed non-JSON output") from exc
        if not isinstance(result, dict) or result.get("is_error"):
            raise TeacherError(f"claude reported an error: {str(result)[:300]}")
        answer = result.get("structured_output")
        if not isinstance(answer, dict):
            raise TeacherError("claude result has no structured_output")
        return answer, float(result.get("total_cost_usd") or 0.0)

    def _cache_key(self, prompt: str, schema: dict, paths: list) -> str:
        h = hashlib.sha256()
        h.update(json.dumps([self.name, prompt, schema], sort_keys=True).encode())
        for p in paths:
            h.update(b"\0" + p.read_bytes())
        return h.hexdigest()
