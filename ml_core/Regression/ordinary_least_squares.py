import numpy as np

from ml_core.Regression.linear_regression import LinearAlgoritm, LinearModel


class OrdinaryLeastSquares(LinearAlgoritm):

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel:
		w = np.linalg.inv(x.T @ x) @ x.T @ y
		return LinearModel(w)
