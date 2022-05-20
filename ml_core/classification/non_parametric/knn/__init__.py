from typing import Callable

import numpy as np

from ml_core.classification.non_parametric.base_class import NonParametricModel


class Knn(NonParametricModel):

	def __init__(self, x: np.ndarray, y: np.ndarray, k: int, dist_function: Callable[[np.ndarray, np.ndarray], float]):
		super().__init__(x, y)
		self.k = k
		self.dist_function = dist_function

	def _calculate_dist_one_2_many(self, rown: np.ndarray, x: np.ndarray) -> np.ndarray:
		return np.apply_along_axis(
			lambda x_rown: self.dist_function(rown, x_rown),
			1,
			x
		)

	def _take_k_nearest(self, dists: np.ndarray) -> np.ndarray:
		temp = np.argpartition(dists, self.k)

		result_idx = temp[:self.k]

		return result_idx

	def _predict_row(self, rown: np.ndarray) -> np.ndarray:

		dists_to_other_points = self._calculate_dist_one_2_many(rown, self.x)

		k_nearests_idx = self._take_k_nearest(dists_to_other_points)

		class_of_points = self.y[k_nearests_idx]

		unique_class, counts = np.unique(class_of_points, return_counts=True)

		most_freq_class_idx = np.argmax(counts)

		return unique_class[most_freq_class_idx]

	def predict(self, x: np.ndarray) -> np.ndarray:
		return np.apply_along_axis(
			self._predict_row, 1, x
		).reshape([-1, 1])

	def is_close(self, model: 'NonParametricModel', **kwargs) -> bool:
		pass

	def __copy__(self):
		return Knn(
			self.x.copy(),
			self.y.copy(),
			self.k,
			self.dist_function
		)

	def __str__(self):
		pass
