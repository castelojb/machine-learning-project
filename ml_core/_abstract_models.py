from abc import abstractmethod

import numpy as np


class MlModel:

	@abstractmethod
	def first_model(self, **kwargs) -> 'MlModel': pass

	@abstractmethod
	def predict(self, x: np.ndarray) -> np.ndarray: pass

	@abstractmethod
	def update(self, **kwargs): pass

	@abstractmethod
	def is_close(self, model: 'MlModel', **kwargs) -> bool: pass

	@abstractmethod
	def __copy__(self): pass

	@abstractmethod
	def __str__(self): pass


class MlAlgoritm:

	@abstractmethod
	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> MlModel | list[MlModel]: pass
