from abc import abstractmethod
from typing import Type

import numpy as np

from ml_core._abstract_models import MlModel, MlAlgoritm


class NonParametricModel(MlModel):
	name: str
	x: np.ndarray
	y: np.ndarray

	def set_data(self, x: np.ndarray, y: np.ndarray):
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

	def __init__(self, model: NonParametricModel):
		self.model = model

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> MlModel:
		final_model = self.model.set_data(
			x, y
		)

		return final_model
