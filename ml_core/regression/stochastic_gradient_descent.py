from typing import Any

import numpy as np
from tqdm.notebook import trange

from ml_core.metrics import ErrorMetrics
from ml_core.regression._linear_regression import LinearAlgoritm, LinearModel


class StochasticGradientDescent(LinearAlgoritm):

	def __step_training(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

		predicted = model.predict(x)

		error = (y - predicted)

		w = model.w + (self.alpha * error * x.T)

		model.update(w)

		return model

	def __training_loop(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

		pbar = trange(self.ephocs)

		for ephoc in pbar:

			for observation, correct_value in zip(x, y):

				observation, correct_value = observation.reshape([1, -1]), correct_value[0]

				model = self.__step_training(observation, correct_value, model)

				predicted = model.predict(x)

				error = ErrorMetrics.rmse(y, predicted)

				yield {
					'ephoc': ephoc,
					'model': model.__copy__(),
					'rmse_error': error
				}

			pbar.set_description(f"RMSE: {error} \n")

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel | list[dict[str, Any]]:

		first_model = LinearModel.first_model(
			lenght=x.shape[1],
			fill_value=self.initial_w_values
		)

		idx = np.arange(x.shape[0])
		np.random.shuffle(idx)

		x_shuffle = x[idx]
		y_shuffle = y[idx]

		history = list(self.__training_loop(x_shuffle, y_shuffle, first_model))

		if self.with_history_predictions:
			return history

		final_model = history[-1]['model']

		return final_model