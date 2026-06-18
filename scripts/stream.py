"""Run the streaming runtime over a replayed feed — a demo of the live service.

Reads a NAB-style CSV one row at a time (as a real feed would arrive) and scores
it online with `OnlineForecastResidualDetector`, printing anomaly events as they
close. With `--rate` it sleeps between rows to play the stream back in pseudo
real time.

    python scripts/stream.py data/raw/ec2_cpu_utilization_5f5533.csv
    python scripts/stream.py data/raw/ec2_cpu_utilization_5f5533.csv --threshold 8 --verbose
    python scripts/stream.py data/raw/ec2_cpu_utilization_5f5533.csv --rate 200   # ~200 rows/sec

Probation is an absolute number of warm-up steps (a live stream has no known
total); the detector scores 0 until then.
"""

import argparse

from threadforge.detection import OnlineForecastResidualDetector
from threadforge.streaming import StreamRuntime, replay_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="NAB-style CSV (columns: timestamp,value)")
    ap.add_argument("--threshold", type=float, default=10.0, help="flag when score >= this. Default 10.")
    ap.add_argument("--probation", type=int, default=750, help="warm-up steps scored 0. Default 750.")
    ap.add_argument("--ewma-alpha", type=float, default=0.2, help="EWMA smoothing. Default 0.2.")
    ap.add_argument("--resid-window", type=int, default=200, help="residual history window. Default 200.")
    ap.add_argument("--gap-steps", type=int, default=20, help="rows that split events. Default 20.")
    ap.add_argument("--rate", type=float, default=None, help="replay rows/sec (default: as fast as possible).")
    ap.add_argument("--verbose", action="store_true", help="print every flagged point, not just events.")
    args = ap.parse_args()

    detector = OnlineForecastResidualDetector(
        ewma_alpha=args.ewma_alpha,
        resid_window=args.resid_window,
        probation=args.probation,
    )

    n_points = 0

    def on_result(r):
        nonlocal n_points
        n_points += 1
        if args.verbose and r.is_anomaly:
            print(f"  flag  @ {r.timestamp}  value={r.value:.3f}  score={r.score:.2f}", flush=True)

    def on_event(ev):
        peak = ev.peak
        print(f"ANOMALY  {ev.start} -> {ev.end}  ({ev.size} pts)  "
              f"peak score {peak.signal_value:.2f} @ {peak.timestamp}", flush=True)

    runtime = StreamRuntime(
        detector, threshold=args.threshold, gap_steps=args.gap_steps,
        on_result=on_result, on_event=on_event,
    )

    print(f"Streaming {args.csv}  (threshold={args.threshold}, probation={args.probation})")
    events = runtime.run(replay_csv(args.csv, rate=args.rate))
    print("-" * 60)
    print(f"{n_points} points streamed -> {len(events)} anomaly event(s)")


if __name__ == "__main__":
    main()
