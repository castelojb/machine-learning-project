from typing import Dict

import numpy as np


def add_ones_column(x: np.ndarray) -> np.ndarray:
	return np.column_stack([
		np.ones(x.shape[0]),
		x
	])


def reshape_vector(x: np.ndarray) -> np.ndarray:
	return x.reshape([-1, 1])


def generate_polynomial_order(x: np.ndarray, order: int, with_bias=False) -> np.ndarray:
	if with_bias: x = add_ones_column(x)

	pow_arrays = [
		x[:, c] ** i for c in range(1, x.shape[1]) for i in range(2, order)
	]

	return np.column_stack([x, *pow_arrays])


def categorize_data(y: np.ndarray) -> tuple[Dict[str, int], np.ndarray]:
	unique_values = np.unique(y)

	convert_values = {
		value: index for index, value in enumerate(unique_values)
	}

	y_ = np.empty([y.shape[0]], dtype=float)

	for i, v in enumerate(y):
		y_[i] = convert_values[v]

	return convert_values, y_


def get_dummies(y: np.ndarray) -> np.ndarray:
	unique_values = np.unique(y)

	return np.array([
		(y == value) for value in unique_values
	], dtype=int)
