"""Run anomaly detection on a NAB-style CSV and print detected events.

Usage:
    python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv

Optionally evaluates against NAB label windows if you pass them in code below.
Reads settings from config/default.json.
"""

import json
import sys
from pathlib import Path

from threadforge.data import stream_csv, check_timestamps
from threadforge.engine import SignalEngine
from threadforge.signals import Momentum, Volatility, Entropy, EntropyFine, EntropyCoarse, Sharpness, Acceleration, ZScore
from threadforge.detection import RobustCalibrator, Detector
from threadforge.evaluation import evaluate, print_report

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config" / "default.json") as f:
        return json.load(f)


def load_labels(filename: str) -> list[tuple[str, str]]:
    """Look up anomaly windows for a given filename from the label registry.

    Returns a list of (start, end) pairs, or an empty list if the file has
    no registered labels — in which case evaluation is skipped.
    """
    registry_path = ROOT / "labels" / "windows.json"
    if not registry_path.exists():
        return []
    with open(registry_path) as f:
        registry = json.load(f)
    windows = registry.get(filename, [])
    return [(w[0], w[1]) for w in windows]


def build_engine_and_calibrators(window_size: int, multiplier: float):
    """Register all signals with the engine and create a matching calibrator each."""
    engine = SignalEngine()
    engine.register("momentum",       Momentum(window_size))
    engine.register("volatility",     Volatility(window_size))
    engine.register("entropy",        Entropy(window_size))
    engine.register("entropy_fine",   EntropyFine(window_size))
    engine.register("entropy_coarse", EntropyCoarse(window_size))
    engine.register("zscore",         ZScore(window_size))
    engine.register("acceleration",   Acceleration(window_size))

    calibrators = {name: RobustCalibrator(multiplier) for name in engine._signals}
    return engine, calibrators


def main(csv_path: str) -> None:
    cfg = load_config()

    stream = stream_csv(csv_path)
    if not stream:
        print(f"No data found in {csv_path}")
        return

    warnings = check_timestamps(stream)
    if warnings:
        print(f"Timestamp warnings ({len(warnings)}):")
        for w in warnings:
            if w["type"] == "gap":
                print(f"  gap {w['multiple']}x median after index {w['after_index']} "
                      f"({w['after_timestamp']} -> {w['before_timestamp']}, {w['gap_seconds']:.0f}s)")
            else:
                print(f"  {w['type']}: {w['detail']}")
        print()

    engine, calibrators = build_engine_and_calibrators(
        cfg["window_size"], cfg["threshold_multiplier"]
    )
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        calib_steps=cfg["calibration_steps"],
        gap_steps=cfg["gap_steps"],
    )

    events = detector.run(stream)

    thresholds = {name: f"{cal.threshold:.3f}" for name, cal in calibrators.items()}
    print(f"Points: {len(stream)}  |  thresholds: {thresholds}")
    print(f"Detected {len(events)} anomaly event(s):\n")
    for ev in events:
        p = ev.peak
        print(
            f"  {ev.start} -> {ev.end}  "
            f"(peak {p.signal_name}={p.signal_value:.2f} at {p.timestamp}, "
            f"value={p.value:.1f}, {ev.size} pts)"
        )

    labels = load_labels(Path(csv_path).name)
    if labels:
        report = evaluate(events, labels)
        print("\nEvaluation vs labeled windows:")
        print_report(report)
    else:
        print("\nNo labels registered for this file — skipping evaluation.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/run_detection.py <path-to-csv>")
        raise SystemExit(1)
    main(sys.argv[1])
