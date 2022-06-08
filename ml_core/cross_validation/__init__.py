from copy import copy
from typing import Callable, Iterable, Type

import numpy as np
from tqdm.notebook import tqdm

from ml_core._abstract_models import MlAlgoritm, MlModel
import itertools
from joblib import Parallel, delayed


class GridSearch:

	def __init__(self,
				 ml_alg: Type[MlAlgoritm],
				 otimization_metric: Callable[[np.ndarray, np.ndarray], float],
				 search_values: dict[str, list[float]]):

		self.search_values = search_values
		self.otimization_metric = otimization_metric
		self.ml_alg = ml_alg

	def _execute_alg_by_parameters(
			self,
			x_trn: np.ndarray,
			y_trn: np.ndarray,
			x_tst: np.ndarray,
			y_tst: np.ndarray,
			parameters: dict):

		alg_with_paramns = self.ml_alg(**parameters)

		model = alg_with_paramns.fit(x_trn, y_trn)

		preds = model.predict(x_tst)

		result = self.otimization_metric(y_tst, preds)

		return {
			'result': result,
			'model': model,
			'paramns': parameters
		}

	def __call__(self,
				 x_trn: np.ndarray,
				 y_trn: np.ndarray,
				 x_tst: np.ndarray,
				 y_tst: np.ndarray) -> dict[str, float | MlModel | dict[str, float]]:

		options = self.search_values.values()
		paramns = self.search_values.keys()

		grid_values = [element for element in itertools.product(*options)]

		grid_formated = map(lambda rown: {key: value for key, value in zip(paramns, rown)}, grid_values)

		results = Parallel(n_jobs=-1)(delayed(self._execute_alg_by_parameters)(
			x_trn,
			y_trn,
			x_tst,
			y_tst,
			param_combination
		) for param_combination in tqdm(grid_formated))

		return max(results, key=lambda x: x['result'])


class Kfold:

	def __init__(self, k: int, metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]], verbose=False):
		self.verbose = verbose
		self.metrics = metrics
		self.k = k

	def generate_folds(self,
					   x: np.ndarray,
					   y: np.ndarray,
					   y_1d: np.ndarray = None) -> Iterable[dict[str, np.ndarray]]:

		idx = np.arange(x.shape[0])

		folds = np.array_split(idx, self.k)

		if y_1d is None:
			y_1d = y

		if y.shape[0] > y.shape[1]:
			select_function = lambda arr, idx: arr[idx]
		else:
			select_function = lambda arr, idx: arr[:, idx]

		for k, fold in enumerate(folds):
			x_fold = x[fold]
			y_fold = y_1d[fold]

			not_in_fold = np.isin(idx, fold, invert=True)

			x_rest = x[not_in_fold]
			y_rest = select_function(y, not_in_fold)

			yield {
				'x_trn': x_rest.copy(),
				'y_trn': y_rest.copy(),
				'x_tst': x_fold.copy(),
				'y_tst': y_fold.copy()
			}

	@staticmethod
	def apply_alg_in_payload(ml_alg: MlAlgoritm, data: dict[str, np.ndarray]) -> MlModel:

		model = ml_alg.fit(
			data['x_trn'],
			data['y_trn']
		)

		return model

	def __call__(self,
				 ml_alg: MlAlgoritm,
				 x: np.ndarray,
				 y: np.ndarray,
				 y_1d: np.ndarray = None,
				 apply_alg_function: Callable[[MlAlgoritm, dict[str, np.ndarray]], MlModel] = apply_alg_in_payload,
				 **kwargs):

		payload_data = self.generate_folds(x, y, y_1d=y_1d)

		fold_results = np.empty(self.k, dtype=dict)

		for k, data in enumerate(payload_data):
			model = apply_alg_function(ml_alg, data)

			preds = model.predict(
				data['x_tst']
			)

			fold_results[k] = {key: fun_(data['y_tst'], preds) for key, fun_ in self.metrics.items()}

		if self.verbose:

			print('--------------REPORTANDO OS RESULTADOS OBTIDOS--------------')

			for metric in self.metrics.keys():
				all_metric_results = np.fromiter(
					map(lambda result: result[metric], fold_results),
					dtype=float
				)

				print(f'--------------{metric.upper()}--------------')
				print(f'Média: {all_metric_results.mean()}')
				print(f'Desvio Padrão: {all_metric_results.std()}')

		return fold_results
