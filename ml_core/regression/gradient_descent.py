from typing import Any

import numpy as np
from tqdm.notebook import trange

from ml_core.metrics import ErrorMetrics
from ml_core.regression._linear_regression import LinearAlgoritm, LinearModel


class GradientDescent(LinearAlgoritm):

    def __step_training(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

        predicted = model.predict(x)

        error = y - predicted

        w = model.w + (self.alpha / x.shape[0]) * (x.T @ error)

        model.update(w)

        return model

    def __training_loop(self, x: np.ndarray, y: np.ndarray, model: LinearModel):

        pbar = trange(self.ephocs)

        for ephoc in pbar:

            model = self.__step_training(x, y, model)

            predicted = model.predict(x)

            error = ErrorMetrics.rmse(y, predicted)

            pbar.set_description(f"RMSE: {error} \n")

            yield {
                'ephoc': ephoc,
                'model': model.__copy__(),
                'rmse_error': error
            }

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel | list[dict[str, Any]]:

        first_model = LinearModel.first_model(
            lenght=x.shape[1],
            fill_value=self.initial_w_values
        )

        history = list(self.__training_loop(x, y, first_model))

        if self.with_history_predictions:
            return history

        final_model = history[-1]['model']

        return final_model
