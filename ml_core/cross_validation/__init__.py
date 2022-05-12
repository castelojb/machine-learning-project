from typing import Callable

import numpy as np

from ml_core._abstract_models import MlAlgoritm, MlModel

class GridSearch:

	def __init__(self,
				 ml_alg: MlAlgoritm,
				 otimization_metric: Callable[[np.ndarray, np.ndarray], float],
				 search_values: dict[str, list[float]]):

		self.search_values = search_values
		self.otimization_metric = otimization_metric
		self.ml_alg = ml_alg

	def __call__(self,
				 x_trn: np.ndarray,
				 y_trn: np.ndarray,
				 x_tst: np.ndarray,
				 y_tst: np.ndarray) -> dict[str, float | MlModel | dict[str, float]]:

		options = self.search_values.values()
		paramns = self.search_values.keys()

		grid_values = np.array(np.meshgrid(options)).T.reshape([-1, len(options)])

		grid_formated = map(lambda rown: {key: value for key, value in zip(paramns, rown)}, grid_values)

		grid_max_result = {
			'result': 0,
			'model': None,
			'paramns': None
		}
		for param_combination in grid_formated:
			alg_with_paramns = self.ml_alg(**param_combination)

			model = alg_with_paramns.fit(x_trn, y_trn)

			preds = model.predict(x_tst)

			result = self.otimization_metric(y_tst, preds)

			if result > grid_max_result['result']:

				grid_max_result = {
					'result': result,
					'model': model,
					'paramns': param_combination
				}

		return grid_max_result

class Kfold:

	def __init__(self, k: int, metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]]):
		self.metrics = metrics
		self.k = k

	def __call__(self, ml_alg: MlAlgoritm, x: np.ndarray, y: np.ndarray):

		idx = np.arange(x.shape[0])

		folds = np.split(idx, self.k)

		models = np.empty(self.k)

		for k, fold in enumerate(folds):

			x_fold = x[fold]
			y_fold = y[fold]
			
			x_rest = x[idx != fold]
			y_rest = y[idx != fold]

			model = ml_alg.fit(x_rest, y_rest)
			preds = model.predict(x_fold)

			models[k] = {key: function(y_fold, preds) for key, function in self.metrics.items()}

		return models
