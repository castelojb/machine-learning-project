import numpy as np

from ml_core.classification.statiscal_classifiers.base_class import StatisticalModel


class GaussianDiscriminant(StatisticalModel):
	name = 'GaussianDiscriminant'

	def predict_row(self, rown: np.ndarray) -> np.ndarray:
		preds = np.log(self.probs) - 0.5 * np.log(np.abs(self.cov_det)) - 0.5 * (rown - self.means).T @ self.cov_inv @ (
					rown - self.means)

		return preds

	def predict(self, x: np.ndarray) -> np.ndarray:
		return np.apply_along_axis(
			self.predict_row, 1, x
		)
