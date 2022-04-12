import numpy as np


class DataProcess:

    @staticmethod
    def add_ones_column(x: np.ndarray) -> np.ndarray:

         return np.column_stack([
                np.ones(x.shape[0]),
                x
            ])

    @staticmethod
    def reshape_vector(x: np.ndarray) -> np.ndarray: return x.reshape([-1, 1])