"""Regression check: fail if faithfulness drops more than a threshold below the baseline.

    python -m eval.regress                    # compare the current run with the baseline
    python -m eval.regress --update-baseline  # record the current run (and the threshold) as the baseline

Uses the fixed RAGAS subset of the regression config (``rerank_tiered``). The threshold is
``noise_multiplier`` times the judge noise measured by judging the same answers twice
(``python -m eval.ragas_run --config rerank_tiered --repeat``), and never below
``min_threshold``, so ordinary judge variation doesn't fail the check.

``make eval`` runs the regression config, judges it and calls this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from eval.report import ragas_item, read_csv
from healthrag.config import get_settings
from healthrag.evaluation.stats import mean

CONFIG = "rerank_tiered"
NOISE_MULTIPLIER = 2.0
MIN_THRESHOLD = 0.05


def judge_noise(first: dict[str, float], second: dict[str, float]) -> float:
    """Mean absolute faithfulness difference between two judgments of the same answers."""
    diffs = [abs(first[i] - second[i]) for i in set(first) & set(second) if first[i] == first[i] and second[i] == second[i]]
    return mean(diffs)


def main(argv: list[str] | None = None) -> int:
    """Exit 1 on a regression, 0 otherwise."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    results = settings.evaluation.results_dir
    current = ragas_item(read_csv(results / f"{CONFIG}_ragas.csv"), "faithfulness")
    if not current:
        print(f"No RAGAS scores for {CONFIG}; run `python -m eval.ragas_run --config {CONFIG}` first.")
        return 1
    score = mean(list(current.values()))
    baseline_file = results / "baseline.json"

    if args.update_baseline:
        repeat = ragas_item(read_csv(results / f"{CONFIG}_ragas_repeat.csv"), "faithfulness")
        noise = judge_noise(current, repeat) if repeat else float("nan")
        threshold = max(MIN_THRESHOLD, NOISE_MULTIPLIER * noise) if noise == noise else MIN_THRESHOLD
        baseline = {
            "config": CONFIG,
            "metric": "faithfulness",
            "mean": round(score, 4),
            "n": len(current),
            "judge_model": settings.llm.judge_model,
            "judge_noise_mean_abs_diff": None if noise != noise else round(noise, 4),
            "threshold": round(threshold, 4),
            "rule": f"threshold = max({MIN_THRESHOLD}, {NOISE_MULTIPLIER} x judge noise)",
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        baseline_file.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"baseline: faithfulness {score:.4f} (n={len(current)}), threshold {threshold:.4f}")
        return 0

    if not baseline_file.exists():
        print("No baseline yet; run `python -m eval.regress --update-baseline` first.")
        return 1
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    drop = baseline["mean"] - score
    status = "FAIL" if drop > baseline["threshold"] else "ok"
    print(f"{status}: faithfulness {score:.4f} vs baseline {baseline['mean']:.4f} "
          f"(drop {drop:.4f}, threshold {baseline['threshold']:.4f}, n={len(current)})")  # fmt: skip
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
