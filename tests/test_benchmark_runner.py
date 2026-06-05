import queue as queue_mod
from types import SimpleNamespace

from simple_ai_benchmarking.benchmark import _extract_repetition_result, process_workloads
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.workloads.llm_workload import PyTorchLocalGeneration


class _FakeQueue:
    """Minimal stand-in for multiprocessing.Queue used to test result handling
    without spawning real processes."""

    def __init__(self, value=None, raises=None):
        self._value = value
        self._raises = raises
        self.get_called = False

    def get(self, timeout=None):
        self.get_called = True
        if self._raises is not None:
            raise self._raises
        return self._value


def test_extract_result_returns_queue_value_on_clean_exit():
    sentinel = object()
    fake_queue = _FakeQueue(value=sentinel)

    assert _extract_repetition_result(0, fake_queue) is sentinel


def test_extract_result_is_runtimeerror_on_nonzero_exit():
    fake_queue = _FakeQueue(value="should-not-be-read")

    result = _extract_repetition_result(1, fake_queue)

    assert isinstance(result, RuntimeError)
    # A crashed child must not cause us to block on (or read from) the queue.
    assert not fake_queue.get_called


def test_extract_result_is_runtimeerror_on_empty_queue():
    fake_queue = _FakeQueue(raises=queue_mod.Empty())

    result = _extract_repetition_result(0, fake_queue)

    assert isinstance(result, RuntimeError)


def test_llm_workload_unifies_on_public_sync_device():
    # The LLM path overrides the public sync_device hook (no private duplicate),
    # so the generic benchmark loop and the internal timing share one mechanism.
    assert PyTorchLocalGeneration.sync_device is not AIWorkload.sync_device
    assert not hasattr(PyTorchLocalGeneration, "_sync_device")


class _Logger:
    def __init__(self):
        self.results = []
        self.exported_sizes = []
        self.excel_exported = False

    def add_benchmark_result_by_averaging_multiple_results(self, results):
        self.results.append(results[0])

    def export_to_csv(self, path):
        self.exported_sizes.append(len(self.results))

    def export_to_excel(self, path):
        self.excel_exported = True

    def pretty_print_summary(self):
        pass


def test_process_workloads_exports_and_calls_back_after_each_workload(monkeypatch):
    result_logger = _Logger()
    callback_sizes = []
    monkeypatch.setattr(
        "simple_ai_benchmarking.benchmark._repeat_benchmark_n_times",
        lambda workload, repetitions: [SimpleNamespace(workload=workload)],
    )

    process_workloads(
        ["a", "b"],
        result_logger=result_logger,
        on_workload_logged=lambda logger: callback_sizes.append(len(logger.results)),
    )

    assert callback_sizes == [1, 2]
    assert result_logger.exported_sizes == [1, 2, 2]
    assert result_logger.excel_exported


def test_process_workloads_swallowing_callback_failure_keeps_running(monkeypatch):
    result_logger = _Logger()
    callback_calls = []
    monkeypatch.setattr(
        "simple_ai_benchmarking.benchmark._repeat_benchmark_n_times",
        lambda workload, repetitions: [SimpleNamespace(workload=workload)],
    )

    def failing_callback(logger):
        callback_calls.append(len(logger.results))
        raise RuntimeError("publish unavailable")

    process_workloads(
        ["a", "b"],
        result_logger=result_logger,
        on_workload_logged=failing_callback,
    )

    assert callback_calls == [1, 2]
    assert result_logger.exported_sizes == [1, 2, 2]
    assert result_logger.excel_exported
