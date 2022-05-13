from typing import Any

import numpy as np
from tqdm.notebook import trange

from ml_core.regression.linear_regression import LinearAlgoritm, LinearModel


class GradientDescent(LinearAlgoritm):

    def _step_training(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

        predicted = model.predict(x)

        error = y - predicted

        w = model.w + self.alpha * ( (1 / x.shape[0]) * (x.T @ error) - self.l2_regulazation*model.w)

        model.update(w)

        return model

    def _training_loop(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

        for _ in trange(self.ephocs):

            model = self._step_training(x, y, model)

            yield model.__copy__()
