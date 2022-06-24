from abc import ABC
from copy import deepcopy
from typing import Callable

import numpy as np
from tqdm.notebook import tqdm

from ml_core._abstract_models import MlModel, MlAlgoritm
from ml_core.metrics.distance_metrics import euclidian


class PCAModel(MlModel, ABC):

	def __init__(self, weights: np.ndarray, explained_variation: np.ndarray):
		self.explained_variation = explained_variation
		self.weights = weights

	def predict(self, x: np.ndarray) -> np.ndarray:
		return (self.weights @ x.T).T

	def __copy__(self):
		return PCAModel(self.weights.copy(), self.explained_variation.copy())

	def __str__(self):
		return self.weights.__str__()


class PCA(MlAlgoritm):

	def __init__(self, n_components: int):
		self.n_components = n_components

	def fit(self, x: np.ndarray, **kwargs) -> MlModel | list[MlModel]:

		cov = np.cov(x.T)

		U, S, _ = np.linalg.svd(cov)

		explained_variation = np.sum(S[:self.n_components] / np.sum(S))

		weights = U[:, :self.n_components].T

		return PCAModel(weights, explained_variation)
