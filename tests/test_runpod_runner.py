import simple_ai_benchmarking.experimental.runpod_runner as runner
from simple_ai_benchmarking.experimental.runpod_runner import (
    DONE_MARKER,
    build_config,
    build_remote_script,
    main,
    parse_args,
)


def _config(argv):
    return build_config(parse_args(argv))


def test_remote_script_cv_publish_includes_register_then_publish():
    cfg = _config(["--dry-run", "--workload", "pt"])
    script = build_remote_script(cfg)

    assert "saib-pt" in script
    # Register must come before publish so unknown-profile rejections are avoided.
    assert script.index("saib-register results_pt.csv") < script.index(
        "saib-pub results_pt.csv"
    )
    assert script.strip().splitlines()[-1] == f'echo "{DONE_MARKER}"'


def test_remote_script_no_publish_skips_upload():
    cfg = _config(["--dry-run", "--workload", "llm", "--no-publish"])
    script = build_remote_script(cfg)

    assert "saib-llm" in script
    assert "saib-pub" not in script
    assert "saib-register" not in script


def test_remote_script_never_contains_the_token():
    # The token is injected over SSH at run time, not baked into the script.
    cfg = _config(["--dry-run", "--workload", "pt", "--db-token", "supersecret"])
    assert "supersecret" not in build_remote_script(cfg)


def test_gpu_fallback_list_is_parsed_in_order():
    cfg = _config(["--dry-run", "--gpu", "A100,RTX 4090 , L40S"])
    assert cfg.gpus == ["A100", "RTX 4090", "L40S"]


def test_dry_run_exits_zero_without_api_key(capsys):
    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "remote script" in out


class _FakeRunpod:
    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times

    def create_pod(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("This machine does not have the resources")
        return {"id": "pod-123"}


def test_capacity_retry_eventually_succeeds(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    cfg = _config(["--dry-run", "--capacity-wait", "60"])
    fake = _FakeRunpod(fail_times=2)

    pod = runner.create_pod_with_fallback(fake, cfg)

    assert pod["id"] == "pod-123"
    assert fake.calls == 3


def test_no_capacity_wait_tries_once_then_exits(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    cfg = _config(["--dry-run", "--gpu", "A,B"])  # capacity_wait defaults to 0
    fake = _FakeRunpod(fail_times=99)

    try:
        runner.create_pod_with_fallback(fake, cfg)
        assert False, "expected SystemExit"
    except SystemExit:
        pass

    # One pass over both GPUs, no retry loop.
    assert fake.calls == 2


def test_ssh_destination_direct_uses_root_and_port():
    direct = runner._ssh_destination({"kind": "direct", "ip": "1.2.3.4", "port": 16593})
    assert direct == ["-p", "16593", "root@1.2.3.4"]


def test_ssh_base_pins_identity_and_batch_mode():
    base = runner._ssh_base("/key")
    assert "BatchMode=yes" in base
    assert "IdentitiesOnly=yes" in base
    assert base[:3] == ["ssh", "-i", "/key"]


def test_check_ssh_key_usable_missing_file(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="not found"):
        runner.check_ssh_key_usable(str(tmp_path / "nope"))


def test_check_ssh_key_usable_unencrypted_passes(tmp_path):
    key = tmp_path / "k"
    import subprocess

    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    runner.check_ssh_key_usable(str(key))  # should not raise


def test_check_ssh_key_usable_encrypted_not_in_agent_errors(tmp_path):
    import pytest
    import subprocess

    key = tmp_path / "enc"
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "secretpass", "-f", str(key)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with pytest.raises(SystemExit, match="ssh-add"):
        runner.check_ssh_key_usable(str(key))


def test_prompt_secret_returns_pasted_value(monkeypatch):
    monkeypatch.setattr(runner.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "  pasted-token  ")
    assert runner._prompt_secret("token") == "pasted-token"


def test_prompt_secret_is_noop_without_tty(monkeypatch):
    monkeypatch.setattr(runner.sys.stdin, "isatty", lambda: False)
    # Must not call getpass (would hang in CI); returns "" so the caller errors.
    monkeypatch.setattr(
        "getpass.getpass",
        lambda prompt="": (_ for _ in ()).throw(AssertionError("getpass called")),
    )
    assert runner._prompt_secret("token") == ""
