import numpy as np


def euclidian(origim: np.ndarray, dest: np.ndarray) -> float:
	return np.sqrt(
		np.sum((dest - origim) ** 2)
	)


class Mahalanobis:

	def __init__(self, cov_matrix: np.ndarray):
		self.inv_cov = np.linalg.pinv(cov_matrix)

	def __call__(self, origim: np.ndarray, dest: np.ndarray) -> float:

		return np.sqrt(
			(origim - dest).T @ self.inv_cov @ (origim - dest)
		)
