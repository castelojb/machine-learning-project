import numpy as np

def rmse(real_arr: np.ndarray, predicted_arr: np.ndarray) -> float:
    diff = (real_arr - predicted_arr) ** 2

    return np.sqrt(diff.mean())

def acuracy(real_arr: np.ndarray, predicted_arr: np.ndarray) -> float:

    return  (real_arr == np.around(predicted_arr)).sum() / real_arr.shape[0]