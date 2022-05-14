from typing import List, Callable

import numpy as np

from ml_core._abstract_models import MlModel


def error_metric_in_history_models(x,
								   y,
								   history_models: List[MlModel],
								   error_metric: Callable[[np.ndarray, np.ndarray], float],
								   denomalize_function: Callable[[np.ndarray], np.ndarray] = lambda x: x) -> List[
	float]:
	erros = []
	for model in history_models:
		preds = model.predict(x)

		preds_denomalized = denomalize_function(preds)

		erros.append(
			error_metric(y, preds_denomalized)
		)

	return erros
