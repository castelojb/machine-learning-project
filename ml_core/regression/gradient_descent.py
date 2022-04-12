from typing import Any

import numpy as np
from tqdm.notebook import trange

from ml_core.metrics import ErrorMetrics
from ml_core.regression.linear_regression import LinearAlgoritm, LinearModel


class GradientDescent(LinearAlgoritm):

	def __step_training(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

		predicted = model.predict(x)

		error = y - predicted

		w = model.w + (self.alpha / x.shape[0]) * (x.T @ error)

		model.update(w)

		return model

	def __training_without_histoty(self, x: np.ndarray, y: np.ndarray, model: LinearModel, desnormalize_function) -> LinearModel:

		pbar = trange(self.ephocs)

		for _ in pbar:
			model = self.__step_training(x, y, model)

			normalized_predicted = model.predict(x)

			denormalized_predicted = desnormalize_function(normalized_predicted)

			error = ErrorMetrics.rmse(y, denormalized_predicted)

			pbar.set_description(f"RMSE: {error} \n")

		return model

	def __training_with_history(
			self,
			x: np.ndarray,
			y: np.ndarray,
			model: LinearModel,
			desnormalize_function
	) -> list[dict[str, Any]]:

		history = list()

		pbar = trange(self.ephocs)

		for ephoc in pbar:

			model = self.__step_training(x, y, model)

			normalized_predicted = model.predict(x)

			#denormalized_predicted = desnormalize_function(normalized_predicted)

			error = ErrorMetrics.rmse(y, normalized_predicted)

			history.append({
				'ephoc': ephoc,
				'model': model.__copy__(),
				'rmse_error': error
			})

			pbar.set_description(f"RMSE: {error} \n")

		return history

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel | list[dict[str, Any]]:

		x_ = np.column_stack([
			np.ones(x.shape[0]),
			x
		])

		model = LinearModel.first_model(
			lenght=x_.shape[1],
			fill_value=self.initial_w_values
		)

		if not self.with_history_predictions:

			final_model = self.__training_without_histoty(x_, y, model, kwargs['denormalize_function'])

			return final_model

		else:

			history = self.__training_with_history(x_, y, model, kwargs['denormalize_function'])

			return history
