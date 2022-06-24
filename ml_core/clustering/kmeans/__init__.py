from copy import deepcopy
from typing import Callable

import numpy as np
from tqdm.notebook import tqdm

from ml_core._abstract_models import MlModel, MlAlgoritm
from ml_core.metrics.distance_metrics import euclidian


class KmeansModel(MlModel):
	
	def __init__(self, centroids: list[np.ndarray], dist_function: Callable[[np.ndarray, np.ndarray], float]):
		self.centroids = centroids
		self.dist_function = dist_function

	@staticmethod
	def first_model(n_centroids: int, n_features: int, dist_function: Callable[[np.ndarray, np.ndarray], float] = euclidian, **kwargs) -> 'KmeansModel':

		centroids = [
			np.random.random(n_features) for _ in range(n_centroids)
		]

		return KmeansModel(
			centroids,
			dist_function
		)

	def predict_only(self, row: np.ndarray) -> int:

		dists = [
			self.dist_function(row, centroid) for centroid in self.centroids
		]

		return np.argmin(dists)

	def predict(self, x: np.ndarray) -> np.ndarray:

		return np.apply_along_axis(
			self.predict_only, 1, x
		)

	def update(self, centroids: list[np.ndarray], **kwargs):

		return KmeansModel(
			deepcopy(centroids), self.dist_function
		)

	def is_close(self, model: 'KmeansModel', **kwargs) -> bool:

		for self_centroid in self.centroids:

			prox = False

			for model_centroid in model.centroids:

				if all(np.isclose(self_centroid, model_centroid)):
					prox = True

			if not prox:
				return False

	def __copy__(self):

		return KmeansModel(
			deepcopy(self.centroids), self.dist_function
		)

	def __str__(self):
		return '\n'.join(map(str, self.centroids))


class Kmeans(MlAlgoritm):

	def __init__(self, ephocs: int = 100, with_history=False, first_model: KmeansModel = None, n_centroids=3, dist_function: Callable[[np.ndarray, np.ndarray], float] = euclidian):
		self.with_history = with_history
		self.dist_function = dist_function
		self.n_centroids = n_centroids
		self.first_model = first_model
		self.ephocs = ephocs

	def _train_loop(self, x: np.ndarray, model: KmeansModel):

		for ephoc in range(self.ephocs):

			clusters = model.predict(x)

			new_clusters = [
				np.mean(
					x[clusters == n_cluster], axis=1
				) for n_cluster in range(len(model.centroids))
			]

			old_model = model.__copy__()
			model.update(new_clusters)

			if model.is_close(old_model): break

			yield model.__copy__()

	def fit(self, x: np.ndarray, **kwargs) -> MlModel | list[MlModel]:

		first_model = self.first_model.__copy__() if self.first_model else KmeansModel.first_model(self.n_centroids, x.shape[1], self.dist_function)

		models = self._train_loop(x, first_model)

		if self.with_history:
			return list(models)

		final_model = first_model
		for model in models:
			final_model = model

		return final_model
		