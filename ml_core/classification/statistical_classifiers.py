from abc import abstractmethod

import numpy as np

from ml_core._abstract_models import MlModel


class StatisticalModel(MlModel):

	name: str

	def __init__(self, covs: np.ndarray, means: np.ndarray, probs: np.ndarray):
		self.probs = probs
		self.means = means
		self.covs = covs

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
			self.probs
		)

	def __str__(self):

		header = f'-------------{self.name}-------------\n'

		covs_str = f'-------------Covariance Matrix-------------\n{self.covs.__str__()}'

		means_str = f'-------------Means Matrix-------------\n{self.means.__str__()}'

		probs_str = f'-------------Probs Matrix-------------\n{self.probs.__str__()}'

		return '\n'.join([
			header,
			covs_str,
			means_str,
			probs_str
		])

class GaussianDiscriminant(StatisticalModel):

	name = 'GaussianDiscriminant'

	def predict(self, x: np.ndarray) -> np.ndarray:
		pass