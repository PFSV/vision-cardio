#!/usr/bin/env python3
"""Emit the locked phase-1 product spec for the vision_cardio project.

This is intentionally deterministic so later iterations can diff the product
contract against implementation changes.
"""

from __future__ import annotations

import json


SPEC = {
    "objective": "Lock the first shippable direction for a passive, on-device heart-rate app and define the first measurable offline target.",
    "platform": "iOS-first",
    "ui": "SwiftUI",
    "capture": "AVFoundation front camera only",
    "inference": "Core ML and Vision on the critical path",
    "privacy": {
        "consent": "explicit opt-in",
        "revocation": "clear revocation",
        "retention": "minimal retention",
    },
    "product_framing": "wellness monitoring only",
    "offline_targets": {
        "hr_mape_percent_max": 10.0,
        "daily_rhr_mae_bpm_max": 5.0,
        "confidence_gating_required": True,
    },
    "paper_relationship": {
        "status": "reference pattern",
        "note": "Any divergence in capture timing, architecture, gating, population, or metrics must be documented before coding.",
    },
    "risks": [
        "No app code exists yet.",
        "No local dataset is wired up yet.",
    ],
}


def main() -> None:
    print(json.dumps(SPEC, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
