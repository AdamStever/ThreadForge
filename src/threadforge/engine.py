"""SignalEngine: feed each incoming value to every registered signal at once.

This is the seam that lets the project grow: today it drives a few hand-written
signals; later the same fan-out feeds many signals into a model without
changing how data flows in.

WHY A FAN-OUT ENGINE?
  Without this, the main script would need to know about every signal and call
  each one manually. As signals accumulate that list grows messier. The engine
  gives us one place to send data — register a new signal once and it
  automatically receives every future value.

ANALOGY:
  Think of a power strip. The stream is the wall socket. Each signal is a
  device plugged into the strip. The engine is the strip itself — it passes
  electricity (data) to everything plugged in, without the wall knowing how
  many devices there are.
"""

from threadforge.signals.base import Signal


class SignalEngine:
    def __init__(self) -> None:
        self._signals: dict[str, Signal] = {}

    def register(self, name: str, signal: Signal) -> None:
        """Plug a signal into the engine under a given name."""
        self._signals[name] = signal

    def update(self, value: float) -> dict[str, float | None]:
        """Push one stream value to every registered signal.

        Returns a dict mapping signal name -> current signal output.
        A value of None means that signal's window hasn't filled yet.
        """
        return {name: sig.update(value) for name, sig in self._signals.items()}

    def reset(self) -> None:
        """Clear all signal windows — use when starting a new stream."""
        for sig in self._signals.values():
            sig.reset()
