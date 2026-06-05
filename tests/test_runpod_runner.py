import simple_ai_benchmarking.experimental.runpod_runner as runner
from simple_ai_benchmarking.experimental.runpod_runner import (
    BLACKWELL_IMAGE,
    DEFAULT_IMAGE,
    DONE_MARKER,
    PIP_BASE,
    PIP_LOWBIT,
    TORCHAO_CU128_INDEX,
    TORCHAO_TORCH28,
    build_config,
    build_container_script,
    build_pod_body,
    is_blackwell,
    main,
    parse_args,
)


def _config(argv):
    return build_config(parse_args(argv))


# --------------------------------------------------------------------------- #
# Generation-aware profile resolution
# --------------------------------------------------------------------------- #
def test_blackwell_detection():
    assert is_blackwell("NVIDIA B200")
    assert is_blackwell("NVIDIA GeForce RTX 5090")
    assert is_blackwell("NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert not is_blackwell("NVIDIA H100 80GB HBM3")
    assert not is_blackwell("NVIDIA RTX A6000")
    assert not is_blackwell("NVIDIA GeForce RTX 4090")


def test_blackwell_gpu_auto_selects_torch28_image_and_lowbit():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA B200"])
    assert cfg.image == BLACKWELL_IMAGE
    assert cfg.pip_spec == PIP_LOWBIT
    # KV decoder and FP8/FP4 are excluded from the default run, even on Blackwell.
    assert cfg.llm_args == "-w 0 1"


def test_non_blackwell_gpu_auto_selects_default_image_no_lowbit():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA RTX A6000"])
    assert cfg.image == DEFAULT_IMAGE
    assert cfg.pip_spec == PIP_BASE
    # Default run excludes the KV decoder and FP8/FP4.
    assert cfg.llm_args == "-w 0 1"


def test_explicit_overrides_win_over_auto():
    # An explicit --llm-args is respected verbatim, e.g. to opt into the FP8 backend.
    cfg = _config(
        ["--dry-run", "--gpu", "NVIDIA H100 80GB HBM3",
         "--image", BLACKWELL_IMAGE, "--pip-spec", PIP_LOWBIT,
         "--llm-args", "--backend huggingface-causal-fp8"]
    )
    assert cfg.image == BLACKWELL_IMAGE
    assert cfg.pip_spec == PIP_LOWBIT
    assert cfg.llm_args == "--backend huggingface-causal-fp8"


def test_first_gpu_in_fallback_list_drives_the_profile():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA B200, NVIDIA H100 80GB HBM3"])
    assert cfg.gpus == ["NVIDIA B200", "NVIDIA H100 80GB HBM3"]
    assert cfg.image == BLACKWELL_IMAGE


# --------------------------------------------------------------------------- #
# Container script
# --------------------------------------------------------------------------- #
def test_script_both_runs_pt_then_llm_and_publishes():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA RTX A6000", "--workload", "both"])
    script = build_container_script(cfg)
    assert script.index("saib-pt") < script.index("saib-llm")
    # Register must precede publish so unknown-profile rejections are avoided.
    assert script.index("saib-register results_pt.csv") < script.index("saib-pub results_pt.csv")
    assert script.index("saib-register llm_results.csv") < script.index("saib-pub-llm llm_results.csv")
    assert 'saib-pt --publish-each --non-interactive --database-url "${URL}"' in script
    assert '--publish-each --non-interactive --database-url "${URL}"' in script
    assert script.strip().splitlines()[-1] == f'echo "{DONE_MARKER}"'


def test_blackwell_script_pins_torchao_to_torch28_compatible_release():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA B200", "--workload", "llm"])
    script = build_container_script(cfg)

    assert (
        f"pip install --extra-index-url {TORCHAO_CU128_INDEX} "
        f'"{TORCHAO_TORCH28}" "{PIP_LOWBIT}"'
    ) in script


def test_non_lowbit_script_does_not_install_torchao():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA RTX A6000"])

    assert "torchao==" not in build_container_script(cfg)


def test_script_self_terminates_and_caps_threads_by_default():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA RTX A6000"])
    script = build_container_script(cfg)
    assert "trap cleanup EXIT" in script
    assert "RUNPOD_POD_ID" in script  # the DELETE teardown
    assert "OMP_NUM_THREADS=8" in script
    assert "timeout -k 60" in script


def test_keep_disables_self_terminate():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA RTX A6000", "--keep"])
    script = build_container_script(cfg)
    assert "trap cleanup EXIT" not in script
    assert "will NOT self-terminate" in script


def test_workload_llm_only_skips_pt():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA RTX A6000", "--workload", "llm"])
    script = build_container_script(cfg)
    assert "saib-llm" in script
    assert "saib-pt" not in script


def test_no_publish_skips_upload():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA RTX A6000", "--no-publish"])
    script = build_container_script(cfg)
    assert "saib-pt" in script
    assert "saib-pub" not in script
    assert "saib-register" not in script
    assert "--publish-each" not in script


def test_script_never_contains_the_token():
    # The token is injected via the pod env, not baked into the script text.
    cfg = _config(["--dry-run", "--gpu", "NVIDIA RTX A6000", "--db-token", "supersecret"])
    assert "supersecret" not in build_container_script(cfg)


# --------------------------------------------------------------------------- #
# Pod body
# --------------------------------------------------------------------------- #
def test_pod_body_no_public_ip_and_token_in_env():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA B200", "--db-token", "tok"])
    cfg.api_key = "key"
    body = build_pod_body(cfg, "echo hi")
    assert body["supportPublicIp"] is False  # no SSH route needed
    assert body["dockerStartCmd"][:2] == ["bash", "-lc"]
    assert body["gpuTypeIds"] == ["NVIDIA B200"]
    assert body["env"]["AI_BENCHMARK_DATABASE_TOKEN"] == "tok"


def test_pod_body_no_publish_omits_token():
    cfg = _config(["--dry-run", "--gpu", "NVIDIA B200", "--no-publish", "--db-token", "tok"])
    cfg.api_key = "key"
    body = build_pod_body(cfg, "echo hi")
    assert "AI_BENCHMARK_DATABASE_TOKEN" not in body["env"]


# --------------------------------------------------------------------------- #
# Capacity retry
# --------------------------------------------------------------------------- #
def test_dry_run_exits_zero_without_api_key(capsys):
    assert main(["--dry-run", "--gpu", "NVIDIA B200"]) == 0
    out = capsys.readouterr().out
    assert "container script" in out


def test_capacity_retry_eventually_succeeds(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def fake_rest(method, path, key, body=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            return 500, "This machine does not have the resources"
        return 201, {"id": "pod-123"}

    monkeypatch.setattr(runner, "rest_call", fake_rest)
    cfg = _config(["--dry-run", "--gpu", "NVIDIA B200", "--capacity-wait", "60"])
    cfg.api_key = "key"
    assert runner.create_pod_with_retry(cfg, "echo hi") == "pod-123"
    assert calls["n"] == 3


def test_no_capacity_wait_tries_once_then_exits(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)

    def fake_rest(method, path, key, body=None):
        return 500, "This machine does not have the resources"

    monkeypatch.setattr(runner, "rest_call", fake_rest)
    cfg = _config(["--dry-run", "--gpu", "NVIDIA B200"])  # capacity_wait defaults to 0
    cfg.api_key = "key"
    try:
        runner.create_pod_with_retry(cfg, "echo hi")
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_prompt_secret_returns_pasted_value(monkeypatch):
    monkeypatch.setattr(runner.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "  pasted-token  ")
    assert runner._prompt_secret("token") == "pasted-token"


def test_prompt_secret_is_noop_without_tty(monkeypatch):
    monkeypatch.setattr(runner.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "getpass.getpass",
        lambda prompt="": (_ for _ in ()).throw(AssertionError("getpass called")),
    )
    assert runner._prompt_secret("token") == ""
