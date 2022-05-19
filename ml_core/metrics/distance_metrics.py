import numpy as np


def euclidian(origim: np.ndarray, dest: np.ndarray) -> float:
	return np.sqrt(
		np.sum((dest - origim) ** 2)
	)


def mahalanobis(origim: np.ndarray, dest: np.ndarray) -> float:
	arr = np.c_[origim, dest]

	cov = np.cov(arr)

	inv_cov = np.linalg.pinv(cov)

	return np.sqrt(
		(origim - dest).T @ inv_cov @ (origim - dest)
	)
