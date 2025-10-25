"""Anomaly scoring utilities for post-processing model energies."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def detect_with_ema(energies: Iterable[float], alpha: float = 0.05, threshold: float = 1.0) -> Tuple[list[bool], list[float]]:
    """Detect anomalies by thresholding an exponential moving average."""

    energies = list(energies)
    if not energies:
        return [], []

    ema_values: list[float] = []
    alarms: list[bool] = []
    ema = energies[0]
    ema_values.append(ema)
    alarms.append(ema > threshold)

    for energy in energies[1:]:
        ema = alpha * energy + (1 - alpha) * ema
        ema_values.append(ema)
        alarms.append(ema > threshold)

    return alarms, ema_values


def detect_with_cusum(
    energies: Iterable[float],
    baseline: float = 0.5,
    h: float = 5.0,
    k: float = 0.1,
) -> Tuple[list[bool], list[float]]:
    """Detect anomalies using a cumulative sum procedure."""

    s = 0.0
    cusum_values: list[float] = []
    alarms: list[bool] = []

    for energy in energies:
        s = max(0.0, s + (energy - baseline - k))
        cusum_values.append(s)
        alarms.append(s > h)

    return alarms, cusum_values
