import numpy as np
from tqdm.notebook import trange

from ml_core.ErrorMetrics import rmse
from ml_core.Regression.linear_regression import LinearAlgoritm, LinearModel


class GradientDescent(LinearAlgoritm):

	def __step_training(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

		predicted = model.predict(x)

		error = y - predicted

		w = model.w + (self.alpha / x.shape[0]) * (x.T @ error)

		model.update(w)

		return model

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel | tuple[LinearModel, list]:

		x_ = np.column_stack([
			np.ones(x.shape[0]),
			x
		])

		model = LinearModel.first_model(
			lenght=x_.shape[1],
			fill_value=self.initial_w_values
		)

		pbar = trange(self.ephocs)

		if not self.with_history_predictions:

			for _ in pbar:

				model = self.__step_training(x_, y, model)

				normalized_predicted = model.predict(x_)

				desnormalized_predicted = kwargs['desnormalize_function'](normalized_predicted)

				pbar.set_description(f"RMSE: {rmse(y, desnormalized_predicted)} \n")

			return model
		else:

			history = list()

			for ephoc in pbar:

				model = self.__step_training(x_, y, model)

				normalized_predicted = model.predict(x_)

				desnormalized_predicted = kwargs['desnormalize_function'](normalized_predicted)

				error = rmse(y, desnormalized_predicted)

				history.append({
					'ephoc': ephoc,
					'prediction': desnormalized_predicted,
					'rmse_error': error
				})
				pbar.set_description(f"RMSE: {error} \n")

			return model, history
