"""Run anomaly detection on a NAB-style CSV and print detected events.

Usage:
    python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv
    python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv --store threadforge.db

Reads settings from config/default.json.
"""

import json
import sys
from pathlib import Path

from threadforge.data import stream_csv, check_timestamps, FeatureStore
from threadforge.engine import SignalEngine
from threadforge.signals import Momentum, Volatility, Entropy, EntropyFine, EntropyCoarse, Acceleration, ZScore, Autocorrelation, HilbertEnvelope, SpectralFlatness
from threadforge.detection import RobustCalibrator, Detector, Scorer
from threadforge.evaluation import evaluate, print_report

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config" / "default.json") as f:
        return json.load(f)


def load_labels(filename: str) -> list[tuple[str, str]]:
    registry_path = ROOT / "labels" / "windows.json"
    if not registry_path.exists():
        return []
    with open(registry_path) as f:
        registry = json.load(f)
    windows = registry.get(filename, [])
    return [(w[0], w[1]) for w in windows]


def build_engine_and_calibrators(window_size: int, multiplier: float):
    engine = SignalEngine()
    engine.register("momentum",       Momentum(window_size))
    engine.register("volatility",     Volatility(window_size))
    engine.register("entropy",        Entropy(window_size))
    engine.register("entropy_fine",   EntropyFine(window_size))
    engine.register("entropy_coarse", EntropyCoarse(window_size))
    engine.register("zscore",         ZScore(window_size))
    engine.register("acceleration",   Acceleration(window_size))
    engine.register("autocorrelation", Autocorrelation(window_size))
    engine.register("hilbert",        HilbertEnvelope(window_size))
    engine.register("spectral_flatness", SpectralFlatness(window_size))
    calibrators = {name: RobustCalibrator(multiplier) for name in engine._signals}
    return engine, calibrators


def main(csv_path: str, db_path: str | None = None) -> None:
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
    scorer = Scorer(cfg["scorer_weights"], cfg["score_threshold"])

    store_ctx = FeatureStore(db_path) if db_path else None

    detector_holder = {}

    def _run(store=None):
        detector = Detector(
            engine=engine,
            calibrators=calibrators,
            scorer=scorer,
            calib_steps=cfg["calibration_steps"],
            gap_steps=cfg["gap_steps"],
            store=store,
            min_calib_samples=cfg.get("min_calibration_samples", 30),
            gap_seconds=cfg.get("gap_seconds"),
        )
        detector_holder["d"] = detector
        return detector.run(stream)

    if store_ctx is not None:
        with store_ctx:
            run_id = store_ctx.begin_run(Path(csv_path).name)
            events = _run(store_ctx)
        print(f"Scores written to {db_path} (run_id={run_id})")
    else:
        events = _run()

    detector = detector_holder["d"]
    samples = detector.calibration_samples
    eff_min, eff_max = min(samples.values()), max(samples.values())
    print(
        f"Calibration: requested {cfg['calibration_steps']} steps  |  "
        f"effective {eff_min}-{eff_max} points per signal (warm-up trimmed)"
    )

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
    args = sys.argv[1:]
    if not args or len(args) > 3:
        print("usage: python scripts/run_detection.py <path-to-csv> [--store <db-path>]")
        raise SystemExit(1)

    csv_path = args[0]
    db_path = None
    if len(args) == 3 and args[1] == "--store":
        db_path = args[2]

    main(csv_path, db_path)
