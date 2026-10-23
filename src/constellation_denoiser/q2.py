"""Q-factor in dB for 16-QAM from BER (fraction, not percent).

Q2_dB = 10 * log10( 10 * erfcinv(8 * BER / 3)**2 )   (formula given in the E015-E018 prompt as
eq. (17) of Sidelnikov et al., JLT 2021; the PDF was not available to verify it). NaN for BER <= 0
or BER >= 3/8, where the inverse is undefined.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfcinv


def q2_db(ber) -> float:
    try:
        ber = float(ber)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(ber) or ber <= 0.0 or ber >= 3.0 / 8.0:
        return float("nan")
    return float(10.0 * np.log10(10.0 * erfcinv(8.0 * ber / 3.0) ** 2))


def q2_fields(raw_ber, model_ber, prefix: str = "") -> dict:
    raw, model = q2_db(raw_ber), q2_db(model_ber)
    return {f"{prefix}q2_db_raw": raw, f"{prefix}q2_db_model": model, f"{prefix}q2_gain_db": model - raw}
