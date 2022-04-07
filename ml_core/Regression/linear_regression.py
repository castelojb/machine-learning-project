import numpy as np
from tqdm.notebook import trange


def gradient_descent(X, y, alpha, ephocs=100, initial_w_values=1):

    X_ = np.column_stack(
        np.ones(X.shpa[0]),
        X
    )

    w = np.full(
        shape=X_.shape[1],
        fill_value=initial_w_values
    )

    pbar = trange(ephocs)
    for _ in pbar:

        predicted = w*X_

        error = y - predicted

        w =

        pbar.set_description(f"Error: {error.sum()} \n")

    return w0, w1
