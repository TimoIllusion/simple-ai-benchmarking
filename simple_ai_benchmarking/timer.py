import time


class Timer:

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.duration_s = time.perf_counter() - self.start
