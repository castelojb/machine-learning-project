from typing import Callable

import numpy as np


def z_score_normalization(arr: np.ndarray,
						  with_denomalized=False
						  ) -> np.ndarray | tuple[np.array, Callable[[np.ndarray], np.ndarray]]:
	mean = arr.mean()

	std = arr.std()

	out = (arr - mean) / std

	if with_denomalized:
		return out, lambda x: (std * x) + mean

	return out


def max_min_normalization(arr: np.ndarray,
						  with_denomalized=False
						  ) -> np.ndarray | tuple[np.array, Callable[[np.ndarray], np.ndarray]]:
	max_ = np.max(arr)

	min_ = np.min(arr)

	out = (arr - min_) / (max_ - min_)

	if with_denomalized:
		return out, lambda x: x * (max_ - min_) + min_

	return out
