from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LinearModel:

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

    def __copy__(self):
        return LinearModel(self.w.reshape([-1, 1]))

    def __str__(self):

        return self.w.__str__()


class LinearAlgoritm(ABC):

    initial_w_values: float
    ephocs: int
    alpha: float
    with_history_predictions: bool
    seed: int
    l2_regulazation: float

    def __init__(self, alpha=0.01, ephocs=100, initial_w_values=1, l2_regulazation=0, with_history_predictions=False, seed=1234):

        self.l2_regulazation = l2_regulazation
        self.seed = seed
        self.with_history_predictions = with_history_predictions
        self.initial_w_values = initial_w_values
        self.ephocs = ephocs
        self.alpha = alpha

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel | list[dict[str, Any]]: pass
