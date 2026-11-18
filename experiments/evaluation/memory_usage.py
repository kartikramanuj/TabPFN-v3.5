"""Sample peak process RAM during a model's fit and prediction."""

from threading import Event, Thread
from time import sleep

import psutil


class PeakMemoryTracker:
    """Track resident-set size and change from the start of one model run."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop = Event()
        self._thread: Thread | None = None
        self.baseline_bytes = 0
        self.peak_bytes = 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
            except psutil.Error:
                return
            sleep(self.interval_seconds)

    def __enter__(self):
        self.baseline_bytes = self._process.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._thread = Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
        except psutil.Error:
            pass

    def as_megabytes(self) -> dict[str, float]:
        """Return baseline, peak, and incremental process RAM in MiB."""
        mib = 1024**2
        return {
            "baseline_process_ram_mb": self.baseline_bytes / mib,
            "peak_process_ram_mb": self.peak_bytes / mib,
            "peak_process_ram_delta_mb": max(
                0.0, (self.peak_bytes - self.baseline_bytes) / mib
            ),
        }
