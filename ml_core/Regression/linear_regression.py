import numpy as np
from tqdm.notebook import trange

from ml_core.ErrorMetrics import rmse


def gradient_descent(X: np.ndarray, y: np.ndarray, alpha=0.01, ephocs=100, initial_w_values=1, with_history_predictions=False):

    X_ = np.column_stack([
        np.ones(X.shape[0]),
        X
    ])

    w = np.full(
        shape=X_.shape[1],
        fill_value=initial_w_values
    ).reshape([-1, 1])

    if with_history_predictions:
        history = []

    pbar = trange(ephocs)
    for _ in pbar:

        predicted = X_ @ w

        error = y - predicted

        w = w + (alpha / X.shape[0]) * (X_.T @ error)

        pbar.set_description(f"RMSE: {rmse(y, predicted)} \n")

        if with_history_predictions:
            history.append(predicted)

    if with_history_predictions:
        return w, history

    return w


def ordinary_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:

    return np.linalg.inv(X.T @ X) @ X.T @ y
