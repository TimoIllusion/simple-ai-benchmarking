import simple_ai_benchmarking.experimental.log_shipper as ls
from simple_ai_benchmarking.experimental.log_shipper import (
    LogShipper,
    post_lines,
    post_run_report,
    _terminal_status_from_stop_file,
    parse_args,
)


def _shipper(tmp_path, **kw):
    log_file = tmp_path / "run.log"
    log_file.write_text("")
    shipper = LogShipper(path=str(log_file), run_id="r1", token="tok", **kw)
    return shipper, log_file


# --------------------------------------------------------------------------- #
# Line splitting / carry handling
# --------------------------------------------------------------------------- #
def test_complete_lines_only_partial_is_carried(tmp_path):
    shipper, _ = _shipper(tmp_path)
    assert shipper._split_complete_lines("hello\nwor", final=False) == ["hello"]
    # The dangling "wor" is held until the rest of the line arrives.
    assert shipper._split_complete_lines("ld\n", final=False) == ["world"]


def test_final_flush_emits_trailing_fragment(tmp_path):
    shipper, _ = _shipper(tmp_path)
    shipper._split_complete_lines("partial line", final=False)
    assert shipper._split_complete_lines("", final=True) == ["partial line"]


def test_final_flush_drops_trailing_empty_after_newline(tmp_path):
    shipper, _ = _shipper(tmp_path)
    assert shipper._split_complete_lines("a\nb\n", final=True) == ["a", "b"]


# --------------------------------------------------------------------------- #
# Reading a growing file
# --------------------------------------------------------------------------- #
def test_read_new_text_advances_offset(tmp_path):
    shipper, log_file = _shipper(tmp_path)
    log_file.write_text("line1\n")
    assert shipper._read_new_text() == "line1\n"
    # Nothing new yet.
    assert shipper._read_new_text() == ""
    log_file.write_text("line1\nline2\n")
    assert shipper._read_new_text() == "line2\n"


def test_read_new_text_handles_truncation(tmp_path):
    shipper, log_file = _shipper(tmp_path)
    log_file.write_text("aaaa\nbbbb\n")
    shipper._read_new_text()
    # File shrinks (rotated/truncated): offset resets and we re-read from the top.
    log_file.write_text("x\n")
    assert shipper._read_new_text() == "x\n"


# --------------------------------------------------------------------------- #
# poll_once: queue + flush + retry semantics
# --------------------------------------------------------------------------- #
def test_poll_once_ships_complete_lines(tmp_path, monkeypatch):
    shipper, log_file = _shipper(tmp_path)
    sent = []
    monkeypatch.setattr(
        ls, "post_lines",
        lambda url, rid, lines, token, stream: sent.append(list(lines)) or True,
    )
    log_file.write_text("a\nb\nhalf")
    shipper.poll_once()
    assert sent == [["a", "b"]]
    # The half line is not shipped until completed / final.
    assert shipper._pending == []


def test_poll_once_keeps_pending_on_failure_then_retries(tmp_path, monkeypatch):
    shipper, log_file = _shipper(tmp_path)
    outcomes = [False, True]
    sent = []

    def fake_post(url, rid, lines, token, stream):
        sent.append(list(lines))
        return outcomes.pop(0)

    monkeypatch.setattr(ls, "post_lines", fake_post)
    log_file.write_text("a\nb\n")
    shipper.poll_once()           # first attempt fails -> lines stay pending
    assert shipper._pending == ["a", "b"]
    shipper.poll_once()           # retries the same lines, succeeds
    assert shipper._pending == []
    assert sent == [["a", "b"], ["a", "b"]]


def test_pending_buffer_is_bounded(tmp_path, monkeypatch):
    shipper, log_file = _shipper(tmp_path)
    monkeypatch.setattr(ls, "post_lines", lambda *a, **k: False)  # always fail
    monkeypatch.setattr(ls, "MAX_PENDING_LINES", 3)
    log_file.write_text("".join(f"line{i}\n" for i in range(10)))
    shipper.poll_once()
    # Only the newest MAX_PENDING_LINES are retained.
    assert shipper._pending == ["line7", "line8", "line9"]


def test_final_flush_retries_tail_before_giving_up(tmp_path, monkeypatch):
    shipper, log_file = _shipper(tmp_path)
    shipper.interval = 0  # no real sleeping between retries
    # Fail the first two attempts, then succeed: the tail must still be delivered
    # instead of being dropped on the single final pass.
    outcomes = [False, False, True]
    sent = []

    def fake_post(url, rid, lines, token, stream):
        ok = outcomes.pop(0)
        if ok:
            sent.append(list(lines))
        return ok

    monkeypatch.setattr(ls, "post_lines", fake_post)
    log_file.write_text("tail line\n")
    shipper._final_flush()
    assert shipper._pending == []
    assert sent == [["tail line"]]


# --------------------------------------------------------------------------- #
# Run lifecycle reporting
# --------------------------------------------------------------------------- #
def test_run_reports_running_then_completed(tmp_path, monkeypatch):
    log_file = tmp_path / "run.log"
    log_file.write_text("done line\n")
    stop_file = tmp_path / "run.done"
    stop_file.write_text("ok")  # present from the start -> one pass then stop

    statuses = []
    monkeypatch.setattr(ls, "post_lines", lambda *a, **k: True)
    monkeypatch.setattr(
        ls, "post_run_report",
        lambda url, token, fields: statuses.append(fields["status"]) or True,
    )
    shipper = LogShipper(
        path=str(log_file), run_id="r1", token="tok",
        stop_file=str(stop_file), report=True, interval=0,
    )
    assert shipper.run() == 0
    assert statuses == ["running", "completed"]


def test_run_without_report_sends_no_status(tmp_path, monkeypatch):
    log_file = tmp_path / "run.log"
    log_file.write_text("x\n")
    stop_file = tmp_path / "run.done"
    stop_file.write_text("ok")
    calls = []
    monkeypatch.setattr(ls, "post_lines", lambda *a, **k: True)
    monkeypatch.setattr(ls, "post_run_report", lambda *a, **k: calls.append(1) or True)
    shipper = LogShipper(
        path=str(log_file), run_id="r1", stop_file=str(stop_file),
        report=False, interval=0,
    )
    shipper.run()
    assert calls == []


def test_terminal_status_from_stop_file(tmp_path):
    ok = tmp_path / "ok.done"
    ok.write_text("0")
    assert _terminal_status_from_stop_file(str(ok)) == "completed"

    bad = tmp_path / "bad.done"
    bad.write_text("137")
    assert _terminal_status_from_stop_file(str(bad)) == "failed"

    assert _terminal_status_from_stop_file(None) == "completed"
    assert _terminal_status_from_stop_file(str(tmp_path / "missing")) == "completed"


# --------------------------------------------------------------------------- #
# HTTP helpers are best-effort (never raise)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_post_lines_returns_true_on_2xx(monkeypatch):
    monkeypatch.setattr(ls.urllib.request, "urlopen", lambda req, timeout=0: _FakeResp(201))
    assert post_lines("http://db", "r1", ["a"], "tok") is True


def test_post_lines_swallows_errors(monkeypatch):
    def boom(req, timeout=0):
        raise OSError("network down")

    monkeypatch.setattr(ls.urllib.request, "urlopen", boom)
    assert post_lines("http://db", "r1", ["a"], "tok") is False


def test_post_run_report_omits_empty_fields(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["body"] = req.data
        return _FakeResp(200)

    monkeypatch.setattr(ls.urllib.request, "urlopen", fake_urlopen)
    ok = post_run_report(
        "http://db", "tok",
        {"run_id": "r1", "status": "running", "host": "", "provider": "runpod"},
    )
    assert ok is True
    import json

    body = json.loads(captured["body"])
    assert "host" not in body  # empty strings are dropped
    assert body["provider"] == "runpod"


def test_parse_args_reads_token_from_env(monkeypatch):
    monkeypatch.setenv("AI_BENCHMARK_DATABASE_TOKEN", "envtok")
    args = parse_args(["/tmp/x.log", "--run-id", "r1"])
    assert args.token == "envtok"
    assert args.run_id == "r1"
