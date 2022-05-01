from typing import Callable

import numpy as np

from ml_core.classification.logistic_functions import sigmoid
from ml_core.regression.linear_regression import LinearModel


class LogisticBinaryModel(LinearModel):

	def __init__(self, w: np.ndarray, logistic_function: Callable[[np.ndarray], np.ndarray]):
		super().__init__(w)
		self.logistic_function = logistic_function

	@staticmethod
	def first_model(lenght: int, fill_value: float) -> 'LogisticBinaryModel':
		w = np.full(shape=lenght, fill_value=fill_value).reshape([-1, 1])
		return LogisticBinaryModel(w, sigmoid)

	@staticmethod
	def __ajust_binary_output(x: np.ndarray) -> np.ndarray:
		return np.around(x)


	def predict(self, x: np.ndarray) -> np.ndarray:
		return self.logistic_function(x @ self.w).reshape([-1, 1])

	def predict_ajust(self, x: np.ndarray) -> np.ndarray:
		return self.__ajust_binary_output(self.predict(x))

	def __copy__(self):
		return LogisticBinaryModel(
			self.w.reshape([-1, 1]),
			self.logistic_function
		)
