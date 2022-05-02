from copy import copy
from typing import Callable

import numpy as np
from tqdm.notebook import tqdm

from ml_core.classification.logistic_functions import sigmoid
from ml_core.regression.linear_regression import LinearModel, LinearAlgoritm
from ml_core._abstract_models import MlAlgoritm, MlModel


class LogisticBinaryModel(LinearModel):

	def __init__(self, w: np.ndarray, logistic_function: Callable[[np.ndarray], np.ndarray]):
		super().__init__(w)
		self.logistic_function = logistic_function

	@staticmethod
	def first_model(lenght: int, fill_value: float, logistic_function: Callable[[np.ndarray], np.ndarray]=sigmoid) -> 'LogisticBinaryModel':
		w = np.full(shape=lenght, fill_value=fill_value).reshape([-1, 1])
		return LogisticBinaryModel(w, logistic_function)

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

class LogisticMulticlassModel(MlModel):
	
	def __init__(self, binary_models: list[LogisticBinaryModel]):
		self.binary_models = binary_models

	def first_model(self, lenght: int, fill_value: float, n_classes: int, **kwargs) -> 'LogisticMulticlassModel':

		return LogisticMulticlassModel(
			[
				LogisticBinaryModel.first_model(lenght, fill_value) for _ in range(n_classes)
			]
		)

	def predict(self, x: np.ndarray) -> np.ndarray:

		preds = np.array([
			model.predict(x) for model in self.binary_models
		])

		return preds.argmax(axis=0).reshape([-1,1])


	def update(self, binary_models: list[LogisticBinaryModel], **kwargs):

		self.binary_models = copy(binary_models)


	def is_close(self, model: 'LogisticMulticlassModel', **kwargs) -> bool:

		return all([
			model1.is_close(model2) for model1, model2 in zip(self.binary_models, model.binary_models)
		])

	def __copy__(self):
		return LogisticMulticlassModel(self.binary_models)

	def __str__(self):

		return '\n'.join([
			model.__str__() for model in self.binary_models
		])


class LinearRegressionMulticlass(MlAlgoritm):

	def __init__(self, linear_alg: LinearAlgoritm, initial_w_values: float=1):
		self.initial_w_values = initial_w_values
		self.linear_alg = linear_alg

	def fit(self, x: np.ndarray, y: np.ndarray, first_model: LogisticBinaryModel = None, **kwargs) -> MlModel | list[MlModel]:

		if not first_model:
			first_model = LogisticBinaryModel.first_model(
				lenght=x.shape[1],
				fill_value=self.initial_w_values
			)


		binary_models = [
			self.linear_alg.fit(x, y_class.reshape([-1, 1]), first_model=first_model.__copy__()) for y_class in tqdm(y)
		]

		return LogisticMulticlassModel(binary_models)
		

