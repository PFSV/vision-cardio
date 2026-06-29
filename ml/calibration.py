from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CalibrationState:
    version: int = 1
    baseline_bpm: float = 72.0
    confidence_threshold: float = 0.6
    smoothing_alpha: float = 0.15
    updates_seen: int = 0

    def update_from_estimate(self, bpm: float, confidence: float) -> None:
        if confidence < self.confidence_threshold:
            return
        self.baseline_bpm = (1.0 - self.smoothing_alpha) * self.baseline_bpm + self.smoothing_alpha * bpm
        self.updates_seen += 1
        if self.updates_seen >= 10:
            # Small automatic tightening once enough stable examples accumulate.
            self.confidence_threshold = min(0.8, self.confidence_threshold + 0.02)

    def suggest_recommendation(
        self,
        current_bpm: float,
        recovery_score: float,
        trend_bpm: float,
        completed_last_session: bool,
        confidence: float,
    ) -> dict:
        if confidence < self.confidence_threshold:
            return {
                "recommendation": "defer",
                "intensity_step": 0,
                "reason": "low confidence",
            }
        hr_delta = current_bpm - self.baseline_bpm
        stressed = recovery_score < 0.4 or trend_bpm >= 5.0 or hr_delta >= 35.0
        stable = recovery_score >= 0.7 and abs(trend_bpm) <= 2.0 and hr_delta <= 20.0
        if stressed:
            return {
                "recommendation": "easy",
                "intensity_step": -1,
                "reason": "elevated load relative to baseline or poor recovery",
            }
        if stable and completed_last_session:
            return {
                "recommendation": "push",
                "intensity_step": 1,
                "reason": "stable signal and good recent recovery",
            }
        return {
            "recommendation": "maintain",
            "intensity_step": 0,
            "reason": "signal usable but not strong enough for a step change",
        }


def load_state(path: str | Path) -> CalibrationState:
    path = Path(path)
    if not path.exists():
        return CalibrationState()
    return CalibrationState(**json.loads(path.read_text()))


def save_state(path: str | Path, state: CalibrationState) -> None:
    path = Path(path)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))

