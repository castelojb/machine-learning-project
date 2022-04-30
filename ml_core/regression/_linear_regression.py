from abc import abstractmethod
import numpy as np

from ml_core._abstract_models import MlModel, MlAlgoritm


class LinearModel(MlModel):

    w: np.ndarray

    def __init__(self, w: np.ndarray):
        self.w = w.reshape([-1, 1])

    @staticmethod
    def first_model(lenght: int, fill_value: float) -> 'LinearModel':
        w = np.full(shape=lenght,fill_value=fill_value).reshape([-1, 1])
        return LinearModel(w)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w

    def update(self, w: np.ndarray):
        self.w = w.reshape([-1, 1])

    def is_close(self, model: 'LinearModel', atol=0.001, **kwargs) -> bool:

        return np.isclose(
            self.w,
            model.w,
            atol=atol
        ).all()

    def __copy__(self):
        return LinearModel(self.w.reshape([-1, 1]))

    def __str__(self):
        return self.w.__str__()


class LinearAlgoritm(MlAlgoritm):

    initial_w_values: float
    ephocs: int
    alpha: float
    with_history_predictions: bool
    seed: int
    l2_regulazation: float

    def __init__(self,
                 alpha=0.01,
                 ephocs=100,
                 initial_w_values=1,
                 l2_regulazation=0,
                 with_history_predictions=False,
                 seed=42,
                 atol=0.00001):

        self.atol = atol
        self.l2_regulazation = l2_regulazation
        self.seed = seed
        self.with_history_predictions = with_history_predictions
        self.initial_w_values = initial_w_values
        self.ephocs = ephocs
        self.alpha = alpha

    @abstractmethod
    def _step_training(self, x: np.ndarray, y: np.ndarray, model: LinearModel): pass

    @abstractmethod
    def _training_loop(self, x: np.ndarray, y: np.ndarray, model: LinearModel): pass

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel | list[LinearModel]:

        first_model = LinearModel.first_model(
            lenght=x.shape[1],
            fill_value=self.initial_w_values
        )

        model_generator = self._training_loop(x, y, first_model)

        previos_model = first_model.__copy__()
        final_model = first_model.__copy__()

        history = []

        for model in model_generator:

            if model.is_close(previos_model, atol=self.atol):
                final_model = model
                break

            if self.with_history_predictions:
                history.append(
                    model.__copy__()
                )

            previos_model = model.__copy__()


        if self.with_history_predictions:
            return history

        return final_model
