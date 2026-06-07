from __future__ import annotations

import sys


class ProgressBar:
    def __init__(self, *, total: int, desc: str, enabled: bool = True) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.enabled = enabled
        self.current = 0
        self.last_rendered = ""

    def __enter__(self) -> "ProgressBar":
        if self.enabled:
            self._render()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.enabled:
            self.current = self.total
            self._render()
            sys.stderr.write("\n")
            sys.stderr.flush()

    def update(self, step: int = 1, **metrics: float) -> None:
        if not self.enabled:
            return
        self.current = min(self.current + step, self.total)
        self._render(metrics)

    def _render(self, metrics: dict[str, float] | None = None) -> None:
        width = 28
        ratio = self.current / self.total if self.total else 1.0
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        suffix = ""
        if metrics:
            suffix = " " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        message = f"\r{self.desc} [{bar}] {self.current}/{self.total}{suffix}"
        padding = " " * max(len(self.last_rendered) - len(message), 0)
        sys.stderr.write(message + padding)
        sys.stderr.flush()
        self.last_rendered = message
