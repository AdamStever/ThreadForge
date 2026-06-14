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
from threadforge.signals import Momentum, Volatility, Entropy, Sharpness, Acceleration
from threadforge.detection import Calibrator, Detector
from threadforge.evaluation import evaluate, print_report


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "default.json"
    with open(cfg_path) as f:
        return json.load(f)


def build_engine_and_calibrators(window_size: int, multiplier: float):
    """Register all signals with the engine and create a matching calibrator each."""
    engine = SignalEngine()
    engine.register("momentum",    Momentum(window_size))
    engine.register("volatility",  Volatility(window_size))
    engine.register("entropy",     Entropy(window_size))
    engine.register("sharpness",   Sharpness(window_size))
    engine.register("acceleration", Acceleration(window_size))

    calibrators = {name: Calibrator(multiplier) for name in engine._signals}
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

    # --- Optional: evaluate against NAB labels for this specific file ---
    labels = [
        ("2014-02-18 16:02:00", "2014-02-19 08:42:00"),
        ("2014-02-24 10:17:00", "2014-02-25 02:57:00"),
    ]
    if "5f5533" in Path(csv_path).name:
        report = evaluate(events, labels)
        print("\nEvaluation vs NAB labels:")
        print_report(report)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/run_detection.py <path-to-csv>")
        raise SystemExit(1)
    main(sys.argv[1])
