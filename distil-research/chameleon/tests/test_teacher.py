"""Teacher runner: the `claude` CLI is always faked here, never called."""

import json
import subprocess
from pathlib import Path

import pytest

from dev.distill.teacher import ClaudeCodeTeacher, TeacherError

SCHEMA = {"type": "object", "properties": {"winner": {"type": "string"}}}


def ok(answer, cost=0.01):
    out = json.dumps({"is_error": False, "subtype": "success",
                      "structured_output": answer, "total_cost_usd": cost})
    return subprocess.CompletedProcess([], 0, stdout=out, stderr="")


class FakeRun:
    def __init__(self, *results):
        self.results = list(results)
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(cmd)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


def make(tmp_path, run, sleeps=None):
    return ClaudeCodeTeacher(cache_dir=tmp_path / "cache", run=run,
                             sleep=(sleeps if sleeps is not None else []).append)


def test_returns_structured_output(tmp_path):
    run = FakeRun(ok({"winner": "A"}))
    teacher = make(tmp_path, run)
    assert teacher.ask("pick", SCHEMA) == {"winner": "A"}
    assert teacher.name == "claude-code:claude-opus-5-5"
    cmd = run.commands[0]
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--max-turns") + 1] == "20"
    assert cmd[cmd.index("--allowedTools") + 1] == "Read"


def test_error_result_raises(tmp_path):
    bad = subprocess.CompletedProcess(
        [], 0, stdout=json.dumps({"is_error": True, "subtype": "error"}), stderr="")
    run, sleeps = FakeRun(bad), []
    with pytest.raises(TeacherError):
        make(tmp_path, run, sleeps).ask("pick", SCHEMA)
    assert len(run.commands) == 3  # first try + 2 retries
    assert sleeps == [30, 120]


def test_failures_then_success_is_retried(tmp_path):
    run = FakeRun(subprocess.TimeoutExpired("claude", 1),
                  subprocess.CompletedProcess([], 1, stdout="", stderr="rate limit"),
                  ok({"winner": "B"}))
    assert make(tmp_path, run).ask("pick", SCHEMA) == {"winner": "B"}
    assert len(run.commands) == 3


def test_missing_claude_and_missing_output_raise(tmp_path):
    with pytest.raises(TeacherError):
        make(tmp_path, FakeRun(FileNotFoundError("claude"))).ask("p", SCHEMA)
    nothing = subprocess.CompletedProcess([], 0, stdout=json.dumps({"is_error": False}), stderr="")
    with pytest.raises(TeacherError):
        make(tmp_path, FakeRun(nothing)).ask("p", SCHEMA)


def test_cache_hit_skips_run(tmp_path):
    run = FakeRun(ok({"winner": "A"}, cost=0.5))
    teacher = make(tmp_path, run)
    first = teacher.ask("pick", SCHEMA)
    again = make(tmp_path, run).ask("pick", SCHEMA)  # fresh object, same cache dir
    assert first == again
    assert len(run.commands) == 1
    other = make(tmp_path, FakeRun(ok({"winner": "B"})))
    assert other.ask("a different prompt", SCHEMA) == {"winner": "B"}
    assert teacher.calls == [{"cost_usd": 0.5, "cached": False}]


def test_changed_image_misses_cache(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"one")
    run = FakeRun(ok({"winner": "A"}))
    teacher = make(tmp_path, run)
    teacher.ask("pick", SCHEMA, [img])
    teacher.ask("pick", SCHEMA, [img])
    assert len(run.commands) == 1
    img.write_bytes(b"two")
    teacher.ask("pick", SCHEMA, [img])
    assert len(run.commands) == 2
    assert [c["cached"] for c in teacher.calls] == [False, True, False]


def test_image_paths_are_absolute_in_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel.png").write_bytes(b"x")
    run = FakeRun(ok({"winner": "A"}))
    make(tmp_path, run).ask("pick", SCHEMA, [Path("rel.png")])
    prompt = run.commands[0][2]
    assert prompt == f"pick\n\nRead these image files:\n{(tmp_path / 'rel.png').resolve()}"


def test_bad_answer_is_not_cached_and_is_retried(tmp_path):
    def validate(answer):
        if "winner" not in answer:
            raise ValueError("no winner")
    run, sleeps = FakeRun(ok({"oops": 1}), ok({"winner": "A"})), []
    teacher = make(tmp_path, run, sleeps)
    assert teacher.ask("pick", SCHEMA, validate=validate) == {"winner": "A"}
    assert len(run.commands) == 2 and sleeps == [30]
    # a bad answer that never gets better raises and leaves nothing in the cache
    run2 = FakeRun(ok({"oops": 1}))
    with pytest.raises(TeacherError):
        make(tmp_path, run2).ask("other", SCHEMA, validate=validate)
    assert len(run2.commands) == 3
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_missing_claude_is_not_retried(tmp_path):
    run, sleeps = FakeRun(FileNotFoundError("claude")), []
    with pytest.raises(TeacherError):
        make(tmp_path, run, sleeps).ask("p", SCHEMA)
    assert len(run.commands) == 1 and sleeps == []
