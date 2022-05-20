import numpy as np
from tqdm.notebook import trange

from ml_core.regression.base_class import LinearAlgoritm, LinearModel


class StochasticGradientDescent(LinearAlgoritm):

	def _step_training(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

		predicted = model.predict(x)

		error = (y - predicted)

		w = model.w + (self.alpha * (error * x.T - (self.l2_regulazation * model.w)))

		model.update(w)

		return model

	def _training_loop(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

		idx = np.arange(x.shape[0])
		np.random.seed(self.seed)

		for _ in trange(self.ephocs):

			np.random.shuffle(idx)
			x_shuffle = x[idx]
			y_shuffle = y[idx]

			for observation, correct_value in zip(x_shuffle, y_shuffle):
				observation, correct_value = observation.reshape([1, -1]), correct_value[0]

				model = self._step_training(observation, correct_value, model)

				yield model.__copy__()
