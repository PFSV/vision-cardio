from __future__ import annotations

import numpy as np


def bin_to_bpm(bin_idx: int, min_bpm: int = 40) -> int:
    return min_bpm + bin_idx


def bpm_to_bin(bpm: float, min_bpm: int = 40, max_bpm: int = 180) -> int:
    bpm = int(round(bpm))
    return int(np.clip(bpm, min_bpm, max_bpm) - min_bpm)
