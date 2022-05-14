from typing import Callable

import numpy as np

from ml_core.classification.non_parametric.base_class import NonParametricModel


class Knn(NonParametricModel):

	def __init__(self, k: int, dist_function: Callable[[np.ndarray, np.ndarray], np.ndarray]):
		self.k = k
		self.dist_function = dist_function

	def _calculate_dist_one_2_many(self, rown: np.ndarray, x: np.ndarray) -> np.ndarray:
		return np.apply_along_axis(
			lambda x_rown: self.dist_function(rown, x_rown),
			0,
			x
		)

	def _calculate_dist_many_2_many(self, origim: np.ndarray, dest: np.ndarray) -> np.ndarray:
		return np.fromiter(map(
			lambda rown: self._calculate_dist_one_2_many(rown, dest), origim
		), dtype=float)

	def _take_k_nearest(self, dists: np.ndarray) -> np.ndarray:
		temp = np.argpartition(-dists, self.k)

		result_idx = temp[:self.k]

		return result_idx

	def predict(self, x: np.ndarray) -> np.ndarray:
		pass

	def is_close(self, model: 'NonParametricModel', **kwargs) -> bool:
		pass

	def __copy__(self):
		pass

	def __str__(self):
		pass
