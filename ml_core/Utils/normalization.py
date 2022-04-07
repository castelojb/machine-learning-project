import numpy as np


def z_score_normalization(arr: np.ndarray) -> np.array:

	mean = arr.mean()

	std = arr.std()

	out = (arr - mean) / std

	return out