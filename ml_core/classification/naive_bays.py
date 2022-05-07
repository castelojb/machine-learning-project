import numpy as np

from ml_core.classification.statiscal_classifiers import StatisticalModel


class NaiveBayesGaussian(StatisticalModel):

	name = 'NaiveBayesGaussian'

	def predict_row(self, rown: np.ndarray) -> np.ndarray:

		preds = np.log(self.probs) - 0.5 * np.log(2*np.pi * self.std**2).sum() - 0.5 * (((rown - self.means) ** 2) / self.std**2).sum()

		return preds


	def predict(self, x: np.ndarray) -> np.ndarray:

		return np.apply_along_axis(
			self.predict_row, 1, x
		)