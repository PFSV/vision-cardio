from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import load_state, save_state
from .rppg_baseline import estimate_hr_from_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rPPG inference for a single video clip.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--state", default="ml_state.json")
    parser.add_argument("--recovery-score", type=float, default=0.5)
    parser.add_argument("--trend-bpm", type=float, default=0.0)
    parser.add_argument("--completed-last-session", action="store_true")
    args = parser.parse_args()

    state = load_state(args.state)
    pred = estimate_hr_from_video(args.video)
    state.update_from_estimate(pred.bpm, pred.confidence)
    decision = state.suggest_recommendation(
        current_bpm=pred.bpm,
        recovery_score=args.recovery_score,
        trend_bpm=args.trend_bpm,
        completed_last_session=args.completed_last_session,
        confidence=pred.confidence,
    )
    save_state(args.state, state)
    print(
        json.dumps(
            {
                "prediction": {
                    "bpm": pred.bpm,
                    "confidence": pred.confidence,
                    "method": pred.method,
                    "peak_strength": pred.peak_strength,
                },
                "calibration": {
                    "baseline_bpm": state.baseline_bpm,
                    "confidence_threshold": state.confidence_threshold,
                    "updates_seen": state.updates_seen,
                },
                "decision": decision,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

