#!/usr/bin/env python3
"""Deterministic exercise-coach policy from camera-based heart-rate signals.

The policy stays intentionally small:
- confidence gates noisy frames,
- uses resting-heart-rate trend plus recovery,
- returns one of easy / maintain / push / defer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CoachInput:
    current_hr_bpm: float
    resting_hr_bpm: float
    hr_confidence: float
    recovery_score: float
    rhr_trend_bpm: float
    completed_last_session: bool
    face_visible: bool
    ambient_light_ok: bool


@dataclass(frozen=True)
class CoachOutput:
    recommendation: str
    intensity_step: int
    camera_state: str
    reason: str


def recommend(payload: CoachInput) -> CoachOutput:
    if not payload.face_visible or not payload.ambient_light_ok or payload.hr_confidence < 0.6:
        return CoachOutput(
            recommendation="defer",
            intensity_step=0,
            camera_state="deferred",
            reason="low-quality camera signal or low confidence",
        )

    hr_delta = payload.current_hr_bpm - payload.resting_hr_bpm
    stressed = payload.recovery_score < 0.4 or payload.rhr_trend_bpm >= 5.0 or hr_delta >= 35.0
    stable = payload.recovery_score >= 0.7 and abs(payload.rhr_trend_bpm) <= 2.0 and hr_delta <= 20.0

    if stressed:
        return CoachOutput(
            recommendation="easy",
            intensity_step=-1,
            camera_state="coaching",
            reason="heart-rate load looks elevated relative to baseline or recovery is poor",
        )

    if stable and payload.completed_last_session:
        return CoachOutput(
            recommendation="push",
            intensity_step=1,
            camera_state="coaching",
            reason="signal is stable and recent recovery/completion support a small increase",
        )

    return CoachOutput(
        recommendation="maintain",
        intensity_step=0,
        camera_state="coaching",
        reason="signal is usable but not strong enough to justify a step change",
    )


def parse_args() -> CoachInput:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-hr-bpm", type=float, required=True)
    parser.add_argument("--resting-hr-bpm", type=float, required=True)
    parser.add_argument("--hr-confidence", type=float, default=0.8)
    parser.add_argument("--recovery-score", type=float, default=0.5)
    parser.add_argument("--rhr-trend-bpm", type=float, default=0.0)
    parser.add_argument("--completed-last-session", action="store_true")
    parser.add_argument("--no-face-visible", dest="face_visible", action="store_false")
    parser.add_argument("--no-ambient-light-ok", dest="ambient_light_ok", action="store_false")
    parser.set_defaults(face_visible=True, ambient_light_ok=True)
    args = parser.parse_args()
    return CoachInput(
        current_hr_bpm=args.current_hr_bpm,
        resting_hr_bpm=args.resting_hr_bpm,
        hr_confidence=args.hr_confidence,
        recovery_score=args.recovery_score,
        rhr_trend_bpm=args.rhr_trend_bpm,
        completed_last_session=args.completed_last_session,
        face_visible=args.face_visible,
        ambient_light_ok=args.ambient_light_ok,
    )


def main() -> None:
    payload = parse_args()
    decision = recommend(payload)
    print(json.dumps({"input": asdict(payload), "output": asdict(decision)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
