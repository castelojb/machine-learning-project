import numpy as np

def rmse(real_arr: np.ndarray, predicted_arr: np.ndarray) -> float:
    diff = (real_arr - predicted_arr) ** 2

    return np.sqrt(diff.mean())

def acuracy(real_arr: np.ndarray, predicted_arr: np.ndarray) -> float:

    return  (real_arr[:, 0] == predicted_arr[:, 0]).sum() / real_arr.shape[0]

def precision(real_arr: np.ndarray, predicted_arr: np.ndarray) -> float:

    positives = real_arr == 1

    negatives = not positives

    true_positives = predicted_arr[positives].sum()

    true_negatives = (not predicted_arr[negatives]).sum()

    return true_positives / (true_positives + true_negatives)

def recall(real_arr: np.ndarray, predicted_arr: np.ndarray) -> float:

    positives = real_arr == 1

    true_positives = predicted_arr[positives].sum()

    false_positives = (not predicted_arr[positives]).sum()

    return true_positives / (true_positives + false_positives)

def f1_score(real_arr: np.ndarray, predicted_arr: np.ndarray) -> float:

    recall_ = recall(real_arr, predicted_arr)

    precision_ = precision(real_arr, predicted_arr)

    return 2 * (recall_ * precision_) / (recall_ + precision_)