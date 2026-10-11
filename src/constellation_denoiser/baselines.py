"""Simple equalizer baselines evaluated with the neural-model split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .experiment import ber_from_iq, load_split
from .paths import E001_CHECKPOINTS, E001_RESULTS


def to_complex(iq: np.ndarray) -> np.ndarray:
    return iq[:, 0] + 1j * iq[:, 1]


def score_linear_ffe(train_data, validation_data, test_data, window: int = 7):
    x_train = train_data[0][:, :, 0] + 1j * train_data[0][:, :, 1]
    y_train = to_complex(train_data[1])
    x_val = validation_data[0][:, :, 0] + 1j * validation_data[0][:, :, 1]
    y_val = to_complex(validation_data[1])
    x_test = test_data[0][:, :, 0] + 1j * test_data[0][:, :, 1]
    y_test = to_complex(test_data[1])

    # Complex linear FIR equalizer plus a bias term, fit by least squares.
    design_train = np.column_stack((x_train, np.ones(len(x_train), dtype=np.complex128)))
    coefficients, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    val_pred = np.column_stack((x_val, np.ones(len(x_val), dtype=np.complex128))) @ coefficients
    test_pred = np.column_stack((x_test, np.ones(len(x_test), dtype=np.complex128))) @ coefficients

    def iq(values):
        return np.column_stack((values.real, values.imag))

    def bit_mse(pred, target):
        return float(np.mean(np.abs(pred - target) ** 2) / 2)

    run_rows = []
    for name in test_data[5]:
        mask = test_data[4] == name
        raw_iq = np.column_stack((test_data[2][mask].real, test_data[2][mask].imag))
        run_rows.append(
            {
                "run": name,
                "raw_ber": ber_from_iq(raw_iq, test_data[3][mask]),
                "ffe_ber": ber_from_iq(iq(test_pred[mask]), test_data[3][mask]),
                "ffe_mse_per_iq": bit_mse(test_pred[mask], y_test[mask]),
            }
        )

    result = {
        "model": "Complex FFE",
        "window": window,
        "parameters_complex": len(coefficients),
        "validation_ber": ber_from_iq(iq(val_pred), validation_data[3]),
        "test_ber": ber_from_iq(iq(test_pred), test_data[3]),
        "test_raw_ber": ber_from_iq(
            np.column_stack((test_data[2].real, test_data[2].imag)), test_data[3]
        ),
        "test_mse_per_iq": bit_mse(test_pred, y_test),
    }
    return result, run_rows, coefficients


def main():
    train_data, validation_data, test_data = load_split(window=7)
    result, per_run, coefficients = score_linear_ffe(train_data, validation_data, test_data)
    output_dir = E001_RESULTS
    pd.DataFrame([result]).to_csv(output_dir / "linear_ffe_baseline.csv", index=False)
    pd.DataFrame(per_run).to_csv(output_dir / "linear_ffe_per_run.csv", index=False)
    np.savez_compressed(E001_CHECKPOINTS / "complex_ffe.npz", coefficients=coefficients)
    print(pd.Series(result).to_string())
    print(pd.DataFrame(per_run).to_string(index=False))


if __name__ == "__main__":
    main()
