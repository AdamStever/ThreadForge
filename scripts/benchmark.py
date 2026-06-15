"""Run detection across all labeled files and report aggregate scores.

Usage:
    python scripts/benchmark.py            # default: peak matching
    python scripts/benchmark.py overlap    # span-overlap matching
    python scripts/benchmark.py peak       # explicit peak matching
"""

import json
import sys
from pathlib import Path

from threadforge.data import stream_csv
from threadforge.engine import SignalEngine
from threadforge.signals import Momentum, Volatility, Entropy, EntropyFine, EntropyCoarse, Acceleration, ZScore, Autocorrelation, HilbertEnvelope, SpectralFlatness
from threadforge.detection import RobustCalibrator, Detector, Scorer
from threadforge.evaluation import evaluate, PEAK, OVERLAP
from threadforge.nab_scoring import score_file, normalized_score

ROOT = Path(__file__).resolve().parent.parent


def event_flags(stream, events) -> list[bool]:
    """Per-row boolean detections from grouped events (for NAB scoring)."""
    flagged = {p.timestamp for ev in events for p in ev.points}
    return [ts in flagged for ts, _ in stream]


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


def main(mode: str = PEAK):
    with open(ROOT / "config" / "default.json") as f:
        cfg = json.load(f)

    with open(ROOT / "labels" / "windows.json") as f:
        all_labels = json.load(f)

    scorer = Scorer(cfg["scorer_weights"], cfg["score_threshold"])

    results = []
    nab_results = []
    print(f"Match mode: {mode}")
    print(f"{'File':<45}  {'Precision':>9}  {'Recall':>6}  {'Events':>6}")
    print("-" * 75)

    for csv_path in sorted((ROOT / "data" / "raw").glob("*.csv")):
        filename = csv_path.name
        if filename not in all_labels or not all_labels[filename]:
            continue

        stream = stream_csv(str(csv_path))
        engine, calibrators = build_engine_and_calibrators(
            cfg["window_size"], cfg["threshold_multiplier"]
        )
        detector = Detector(
            engine=engine,
            calibrators=calibrators,
            scorer=scorer,
            calib_steps=cfg["calibration_steps"],
            gap_steps=cfg["gap_steps"],
            min_calib_samples=cfg.get("min_calibration_samples", 30),
            gap_seconds=cfg.get("gap_seconds"),
        )
        events = detector.run(stream)
        labels = [(w[0], w[1]) for w in all_labels[filename]]
        r = evaluate(events, labels, mode=mode)
        results.append(r)
        nab_results.append(score_file(
            [ts for ts, _ in stream], event_flags(stream, events), labels,
            profile="standard", probation=cfg["calibration_steps"],
        ))
        print(f"{filename:<45}  {r['precision']:>9.3f}  {r['recall']:>6.3f}  {len(events):>6}")

    if not results:
        print("No labeled files found.")
        return

    avg_p = sum(r["precision"] for r in results) / len(results)
    avg_r = sum(r["recall"] for r in results) / len(results)
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) else 0.0

    print("-" * 75)
    print(f"{'Files scored:':<45}  {len(results)}")
    print(f"{'Avg Precision:':<45}  {avg_p:.3f}")
    print(f"{'Avg Recall:':<45}  {avg_r:.3f}")
    print(f"{'Avg F1:':<45}  {f1:.3f}")
    print(f"{'NAB score (standard, 0-100):':<45}  {normalized_score(nab_results):.1f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else PEAK
    if mode not in (PEAK, OVERLAP):
        print(f"usage: python scripts/benchmark.py [{PEAK}|{OVERLAP}]")
        raise SystemExit(1)
    main(mode)
