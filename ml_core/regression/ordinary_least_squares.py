import numpy as np

from ml_core.regression._linear_regression import LinearAlgoritm, LinearModel


class OrdinaryLeastSquares(LinearAlgoritm):

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel:

		l2_reg_matrix = np.eye(x.shape[1]) * self.l2_regulazation

		w = np.linalg.inv( (x.T @ x) + l2_reg_matrix ) @ x.T @ y

		return LinearModel(w)
