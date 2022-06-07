from abc import abstractmethod, ABC
from typing import Type

import numpy as np
import copy
from ml_core._abstract_models import MlModel, MlAlgoritm
from ml_core.data_process import add_ones_column
from ml_core.neural_networks.activate_functions import Relu


class Layer(MlModel, ABC):

	@abstractmethod
	def update_neurons_weights(self, **kwargs):
		pass


class Network(MlModel):

	def __init__(self, layers: np.ndarray):
		self.layers = layers

	@staticmethod
	def _first_model(
			n_layers: int,
			n_neurons: int,
			n_features: int,
			momentum: float,
			layer_type: Type[Layer],
			bias=0,
			activation_function=Relu) -> np.ndarray:

		layers = np.empty(n_layers, dtype=Layer)
		layers[0] = layer_type.first_model(
			n_neurons,
			n_features,
			momentum,
			bias=bias,
			activation_function=activation_function,
		)
		for idx in range(1, n_layers-1):
			layers[idx] = layer_type.first_model(
				n_neurons,
				n_neurons - 1,
				momentum,
				bias=bias,
				activation_function=activation_function,
			)

		return layers

	def update_output_layer_w(self, input: np.ndarray, output: np.ndarray, y_true: np.ndarray, alpha: float,
							  regularization=1):

		theta = self.layers[-1].update_output_layer_w(input, output, y_true, alpha, regularization=regularization)

		return theta

	def _predict(self, x: np.ndarray) -> np.ndarray:
		u = x

		for layer in self.layers:
			y = layer.predict(u)
			u = layer.activation_function.function(y)

		return u

	@abstractmethod
	def predict(self, x: np.ndarray) -> np.ndarray:
		pass

	def update(self, **kwargs):
		pass

	def is_close(self, model: 'MlModel', **kwargs) -> bool:
		pass

	def __copy__(self):
		return Network(
			self.layers.copy()
		)

	def __str__(self):
		return '\n\n'.join([
			layer.__str__() for layer in self.layers
		])


class RegressionNetwork(Network):

	def __copy__(self):
		return RegressionNetwork(
			copy.deepcopy(self.layers)
		)

	@staticmethod
	def first_model(
			n_layers: int,
			n_neurons: int,
			n_features: int,
			momentum: float,
			layer_type: Type[Layer],
			bias=0,
			activation_function=Relu,
			**kwargs) -> 'MlModel':

		layers = Network._first_model(
			n_layers,
			n_neurons,
			n_features,
			momentum,
			layer_type,
			bias=bias,
			activation_function=activation_function
		)

		layers[-1] = layer_type.first_model(
					1,
					n_neurons - 1,
					momentum,
					bias=bias,
					activation_function=activation_function,
			)

		return RegressionNetwork(
			layers
		)

	def predict(self, x: np.ndarray) -> np.ndarray:
		preds = self._predict(x)
		return preds.reshape(1, -1)[0]


class ClassificationNetwork(Network):

	def __copy__(self):
		return ClassificationNetwork(
			self.layers.copy()
		)

	@staticmethod
	def first_model(
			n_layers: int,
			n_neurons: int,
			n_features: int,
			n_class: int,
			momentum: float,
			layer_type: Type[Layer],
			bias=0,
			activation_function=Relu,
			**kwargs) -> 'MlModel':

		layers = Network._first_model(
			n_layers,
			n_neurons,
			n_features,
			momentum,
			layer_type,
			bias=bias,
			activation_function=activation_function
		)

		layers[-1] = layer_type.first_model(
					n_class,
					n_neurons - 1,
					momentum,
					bias=bias,
					activation_function=activation_function,
			)

		return ClassificationNetwork(
			layers
		)

	def predict(self, x: np.ndarray) -> np.ndarray:
		preds = self._predict(x)
		return preds.argmax(axis=1).reshape([-1, 1])


class NetworkTrainer(MlAlgoritm):

	def __init__(
			self,
			first_model: Network = None,
			ephocs=100,
			batch_size=40,
			alpha=0.01,
			regularization=1,
			seed=42,
			with_history=False):

		self.with_history = with_history
		self.seed = seed
		self.regularization = regularization
		self.alpha = alpha
		self.batch_size = batch_size
		self.ephocs = ephocs
		self.first_model = first_model

	@staticmethod
	def _forward(x_batch: np.ndarray, model: Network) -> tuple[list[np.ndarray], list[np.ndarray]]:

		first_value = x_batch

		after_activate = [first_value]
		before_activate = [first_value]

		for layer in model.layers:
			layer_pred = layer.predict(after_activate[-1])

			layer_prec_activated = layer.activation_function.function(layer_pred)

			before_activate.append(layer_pred)
			after_activate.append(layer_prec_activated)

		return before_activate, after_activate

	def _backward(self, model: Network, before_activate: list[np.ndarray], after_activate: list[np.ndarray],
				  theta: np.ndarray):

		for idx_layer in range(model.layers.shape[0] - 1, 0, -1):
			current_layer = model.layers[idx_layer - 1]
			previos_layer = model.layers[idx_layer]

			theta = current_layer.update_hidden_layer_w(
				output=before_activate[idx_layer],
				input=after_activate[idx_layer - 1],
				back_layer=previos_layer,
				previos_theta=theta,
				alpha=self.alpha,
				regularization=self.regularization
			)

		return theta

	def _loop_training(self, x: np.ndarray, y: np.ndarray, model: Network) -> list[Network]:

		data_idx = np.arange(x.shape[0])
		# np.random.seed(self.seed)

		batch_idx = np.random.choice(data_idx, [self.ephocs, self.batch_size])

		for ephoc in range(self.ephocs):

			idx = batch_idx[ephoc]

			x_batch = x[idx, :]
			y_batch = y[idx, :]

			before_activate, after_activate = self._forward(x_batch, model)

			theta_forward = model.update_output_layer_w(
				input=before_activate[-2],
				output=after_activate[-1],
				y_true=y_batch,
				alpha=self.alpha,
				regularization=self.regularization
			)

			theta_backward = self._backward(
				model,
				before_activate,
				after_activate,
				theta_forward
			)
			print(f'------------------{ephoc}------------------')
			print(model)
			yield model.__copy__()

	def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> MlModel | list[MlModel]:

		model = self.first_model.__copy__()

		model_generator = self._loop_training(x, y, model.__copy__())

		if self.with_history:
			return list(model_generator)

		final_model = model.__copy__()

		for model in model_generator:
			final_model = model

		return final_model.__copy__()
