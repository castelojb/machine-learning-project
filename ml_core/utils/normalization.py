from typing import Callable

import numpy as np


class Normalization:

    @staticmethod
    def z_score_normalization(arr: np.ndarray) -> tuple[np.array, Callable[[np.ndarray], np.ndarray]]:
        mean = arr.mean()

        std = arr.std()

        out = (arr - mean) / std

        return out, lambda x: (std * x) + mean
