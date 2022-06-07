from typing import Type

import numpy as np

from ml_core._abstract_models import MlModel
from ml_core.neural_networks.activate_functions import ActivationFunction, Relu
from ml_core.neural_networks.base_class import Layer


class PerceptronLayer(Layer):

	def __init__(self, n_features: int, momentum: float, activation_function: Type[ActivationFunction],
				 neurons: np.ndarray, delta: np.ndarray):
		self.delta = delta
		self.activation_function = activation_function
		self.momentum = momentum
		self.neurons = neurons
		self.n_features = n_features

	@staticmethod
	def first_model(
			n_neurons: int,
			n_features: int,
			momentum: float,
			bias=0.2,
			activation_function=Relu,
			seed=42,
			**kwargs) -> 'PerceptronLayer':

		# np.random.seed(seed)
		w = np.sqrt(2 / n_neurons) * np.random.normal(0, 1, (n_neurons, n_features))

		neurons = np.c_[np.zeros(n_neurons) + bias, w]

		delta = np.zeros_like(neurons)

		return PerceptronLayer(
			n_features,
			momentum,
			activation_function,
			neurons,
			delta
		)

	def predict(self, x: np.ndarray) -> np.ndarray:
		return x @ self.neurons.T

	def update(self, momentum: float, **kwargs):
		self.momentum = momentum

	def update_neurons_weights(self, gradient: np.ndarray):
		self.delta = self.momentum * self.delta - gradient
		self.neurons = self.neurons + self.delta

	def update_output_layer_w(self, input: np.ndarray, output: np.ndarray, y_true: np.ndarray, alpha: float,
							  regularization=1) -> np.ndarray:
		error = y_true - self.activation_function.function(output)
		theta = -error
		gradient = theta.T @ input

		update_term = alpha * (gradient + regularization * self.neurons)

		self.update_neurons_weights(update_term)

		return theta

	def update_hidden_layer_w(
			self,
			output: np.ndarray,
			input: np.ndarray,
			back_layer: 'PerceptronLayer',
			previos_theta: np.ndarray,
			alpha: float,
			regularization=1) -> np.ndarray:
		theta = self.activation_function.derivation_function(output) * (previos_theta @ back_layer.neurons)

		gradient = theta.T @ input

		update_term = alpha * (gradient + regularization * self.neurons)

		self.update_neurons_weights(update_term)

		return theta

	def is_close(self, model: 'MlModel', **kwargs) -> bool:
		pass

	def __copy__(self):
		return PerceptronLayer(
			self.n_features,
			self.momentum,
			self.activation_function,
			self.neurons.copy(),
			self.delta.copy()
		)

	def __str__(self):
		return self.neurons.__str__()
