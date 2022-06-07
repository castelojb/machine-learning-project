from typing import Dict, Callable

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


def normalize_data(x: np.ndarray, normalize_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
	x_norm = np.empty_like(x)

	for idx, col in enumerate(x.T):
		norm = normalize_function(col)
		x_norm[:, idx] = norm

	return x_norm


def get_dummies(y: np.ndarray, with_data_orientation=False) -> np.ndarray:
	unique_values = np.unique(y)

	encode = np.array([
		(y == value) for value in unique_values
	], dtype=int)

	one_hot_encode = encode.reshape([y.shape[0], unique_values.shape[0]]) if with_data_orientation else encode

	return one_hot_encode
