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

    @staticmethod
    def generate_polynomial_order(x: np.ndarray, order: int, with_bias=False) -> np.ndarray:

        if with_bias: x = DataProcess.add_ones_column(x)

        pow_arrays = [
            x[:, c] ** i for c in range(1, x.shape[1]) for i in range(2, order)
        ]

        return np.column_stack([x, *pow_arrays])