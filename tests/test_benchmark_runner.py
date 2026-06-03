import queue as queue_mod

from simple_ai_benchmarking.benchmark import _extract_repetition_result
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
