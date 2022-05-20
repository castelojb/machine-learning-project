from abc import abstractmethod
from typing import Type

import numpy as np

from ml_core._abstract_models import MlModel, MlAlgoritm


class NonParametricModel(MlModel):
	name: str

	def __init__(self, x: np.ndarray, y: np.ndarray, **kwargs):
		self.x = x
		self.y = y

	def first_model(self, **kwargs) -> 'NonParametricModel':
		pass

	@abstractmethod
	def predict(self, x: np.ndarray) -> np.ndarray:
		pass

	def update(self, x: np.ndarray, y: np.ndarray, **kwargs):
		self.y = y
		self.x = x

	@abstractmethod
	def is_close(self, model: 'NonParametricModel', **kwargs) -> bool:
		pass

	@abstractmethod
	def __copy__(self):
		pass

	@abstractmethod
	def __str__(self):
		pass


class NonParametricAlg(MlAlgoritm):

	def __init__(self, model: Type[NonParametricModel], **kwargs):
		self.model = model
		self.kwargs = kwargs

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> MlModel:

		return self.model(x, y, **self.kwargs)
