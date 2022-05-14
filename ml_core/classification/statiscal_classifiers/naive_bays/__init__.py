import numpy as np

from ml_core.classification.statiscal_classifiers.base_class import StatisticalModel


class NaiveBayesGaussian(StatisticalModel):
	name = 'NaiveBayesGaussian'

	def predict_row(self, rown: np.ndarray) -> np.ndarray:
		term1 = 0.5 * np.sum(
			np.log(
				2 * np.pi * self.std ** 2
			)
		)

		term2 = 0.5 * np.sum(
			((rown - self.means) ** 2) / (self.std ** 2)
		)

		preds = np.log(self.probs) - term1 - term2

		return preds

	def predict(self, x: np.ndarray) -> np.ndarray:
		return np.apply_along_axis(
			self.predict_row, 1, x
		)
