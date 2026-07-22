"""Normal calibration skorlarindan alarm esigi secimi."""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def pot_threshold(values, *, q: float = 0.01, initial_quantile: float = 0.5,
                  min_exceedances: int = 3) -> float:
    """Peaks-over-threshold/GPD ile ust kuyruk esigi; az veride max fallback."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        raise ValueError("POT icin sonlu calibration skoru yok")
    u = float(np.quantile(vals, initial_quantile))
    exceed = vals[vals > u] - u
    if len(exceed) < min_exceedances:
        return float(vals.max())
    xi, _, sigma = genpareto.fit(exceed, floc=0.0)
    n, nu = len(vals), len(exceed)
    if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
        return float(vals.max())
    if abs(xi) < 1e-6:
        return float(u + sigma * np.log(nu / (q * n)))
    return float(u + (sigma / xi) * ((q * n / nu) ** (-xi) - 1))
