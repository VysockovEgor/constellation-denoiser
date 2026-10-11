from itertools import product
import numpy as np
import pandas as pd
from pathlib import Path

from .paths import RAW_DATA_DIR


SQRT10 = np.sqrt(10.0)
LEVELS = np.array([-3 / SQRT10, -1 / SQRT10, 1 / SQRT10, 3 / SQRT10])
AXIS_BITS = {LEVELS[2]: "00", LEVELS[3]: "01", LEVELS[1]: "10", LEVELS[0]: "11"}


def load_channel_data(csv_path):
    path = Path(csv_path)
    if not path.is_absolute():
        if not path.exists():
            path = RAW_DATA_DIR / path
    df = pd.read_csv(path, header=None, names=["tx_I", "tx_Q", "rx_I", "rx_Q"])
    tx = df["tx_I"] + df["tx_Q"] * 1j
    rx = df["rx_I"] + df["rx_Q"] * 1j
    return np.array(tx), np.array(rx)


def bit_constellation():
    points, labels = [], []
    for i, j in product(LEVELS, repeat=2):
        points.append(i + j * 1j)
        labels.append(AXIS_BITS[i] + AXIS_BITS[j])
    return np.array(points), np.array(labels)


CONSTELLATION, BIT_LABELS = bit_constellation()
BIT_TABLE = np.array([list(map(int, label)) for label in BIT_LABELS])


def get_indexes(points):
    distances = np.abs(points[:, None] - CONSTELLATION[None, :])
    return np.argmin(distances, axis=1)


def symbols_to_bits(points):
    return BIT_TABLE[get_indexes(points)].reshape(-1)


def calculate_ber(rx_bits, tx_bits):
    return sum(rx_bits ^ tx_bits) / rx_bits.size


def split(tx, rx):
    n = tx.size
    a, b = int(n * 0.7), int(n * 0.85)
    return (tx[:a], rx[:a]), (tx[a:b], rx[a:b]), (tx[b:], rx[b:])
