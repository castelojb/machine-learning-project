import numpy as np

from ml_core.regression.linear_regression import LinearAlgoritm, LinearModel


class OrdinaryLeastSquares(LinearAlgoritm):

	def _step_training(self, x: np.ndarray, y: np.ndarray, model: LinearModel):
		pass

	def _training_loop(self, x: np.ndarray, y: np.ndarray, model: LinearModel):
		pass

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel:

		l2_reg_matrix = np.eye(x.shape[1]) * self.l2_regulazation

		w = np.linalg.inv( (x.T @ x) + l2_reg_matrix ) @ x.T @ y

		return LinearModel(w)
