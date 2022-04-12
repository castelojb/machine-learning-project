from abc import ABC, abstractmethod
import numpy as np


class LinearModel:

    w: np.ndarray

    def __init__(self, w: np.ndarray):
        self.w = w.copy()

    @staticmethod
    def first_model(lenght: int, fill_value: float) -> 'LinearModel':
        w = np.full(shape=lenght,fill_value=fill_value).reshape([-1, 1])
        return LinearModel(w)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w

    def update(self, w: np.ndarray):
        self.w = w.copy()

    def __copy__(self):
        return LinearModel(self.w)


class LinearAlgoritm(ABC):

    with_regulazation: bool
    initial_w_values: float
    ephocs: int
    alpha: float
    with_history_predictions: bool

    def __init__(self, alpha=0.01, ephocs=100, initial_w_values=1, with_regulazation=False, with_history_predictions=False):
        self.with_history_predictions = with_history_predictions
        self.with_regulazation = with_regulazation
        self.initial_w_values = initial_w_values
        self.ephocs = ephocs
        self.alpha = alpha

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> LinearModel: pass
