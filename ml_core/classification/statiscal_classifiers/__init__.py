from abc import abstractmethod

import numpy as np

from ml_core._abstract_models import MlModel, MlAlgoritm


class StatisticalModel(MlModel):

	name: str

	def __init__(self, covs: np.ndarray, means: np.ndarray, probs: np.ndarray, std: np.ndarray, noise=10**-5):
		self.std = std
		self.probs = probs
		self.means = means
		self.covs = covs

		self.__preprocess_data(noise)

	def first_model(self, **kwargs) -> 'StatisticalModel':
		pass

	@abstractmethod
	def predict(self, x: np.ndarray) -> np.ndarray:
		pass

	def update(self, covs: np.ndarray, means: np.ndarray, probs: np.ndarray, **kwargs):
		self.probs = probs
		self.means = means
		self.covs = covs

	def is_close(self, model: 'StatisticalModel', **kwargs) -> bool:
		pass

	def __copy__(self):
		return StatisticalModel(
			self.covs,
			self.means,
			self.probs,
			self.std
		)

	def __str__(self):

		header = f'-------------{self.name}-------------\n'

		covs_str = f'-------------Covariance Matrix-------------\n{self.covs.__str__()}'

		means_str = f'-------------Means Matrix-------------\n{self.means.__str__()}'

		probs_str = f'-------------Probs Class-------------\n{self.probs.__str__()}'

		return '\n'.join([
			header,
			covs_str,
			means_str,
			probs_str
		])

	def __preprocess_data(self, noise):
		self.cov_inv = np.linalg.pinv(self.covs)
		self.cov_det = np.linalg.det(self.covs)

		if np.linalg.det(self.covs) == 0: self.cov_det += noise


class MulticlassStatisticalModel(MlModel):

	def __init__(self, statiscal_models: list[StatisticalModel]):
		self.statiscal_models = statiscal_models

	def first_model(self, **kwargs) -> 'MlModel':
		pass

	def predict(self, x: np.ndarray) -> np.ndarray:

		preds = np.array([
			model.predict(x) for model in self.statiscal_models
		])
		return preds
		return preds.argmax(axis=0).reshape([-1, 1])

	def update(self, **kwargs):
		pass

	def is_close(self, model: 'MlModel', **kwargs) -> bool:
		pass

	def __copy__(self):
		return MulticlassStatisticalModel(self.statiscal_models)

	def __str__(self):

		return '\n'.join([
			model.__str__() for model in self.statiscal_models
		])


class StatisticalClassifiers(MlAlgoritm):

	def __init__(self, model: StatisticalModel):
		self.model = model

	@staticmethod
	def __get_metrics_from_data(x: np.ndarray, y: np.ndarray, model: StatisticalModel) -> list[StatisticalModel]:

		models = [
			model(
				covs=np.cov(x[y_class].T),
				means=x[y_class].mean(axis=0),
				probs=x[y_class].shape[0] / x.shape[0],
				std=x[y_class].std(axis=0)
			) for y_class in y
		]

		return models

	@abstractmethod
	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> MlModel | list[MlModel]:

		models = self.__get_metrics_from_data(
			x, y, self.model
		)

		return MulticlassStatisticalModel(models)