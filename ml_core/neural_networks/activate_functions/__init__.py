from abc import abstractmethod
import numpy as np


class ActivationFunction:

	@staticmethod
	@abstractmethod
	def function(x: np.ndarray) -> np.ndarray: pass

	@staticmethod
	@abstractmethod
	def derivation_function(x: np.ndarray) -> np.ndarray: pass


class Relu(ActivationFunction):

	@staticmethod
	def function(x: np.ndarray) -> np.ndarray:

		x_ = x.copy()
		for idx_x in range(x.shape[0]):
			for idx_y in range(x.shape[1]):
				x_[idx_x, idx_y] = max(0, x_[idx_x, idx_y])
		return x_

	@staticmethod
	def derivation_function(x: np.ndarray) -> np.ndarray:
		x[x <= 0] = 0
		x[x > 0] = 1
		return x


class Sigmoid(ActivationFunction):

	@staticmethod
	def function(x: np.ndarray) -> np.ndarray:
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def derivation_function(x: np.ndarray) -> np.ndarray:
		return Sigmoid.function(x) - np.square(Sigmoid.function(x))


class Tanh(ActivationFunction):

	@staticmethod
	def function(x: np.ndarray) -> np.ndarray:
		return np.tanh(x)

	@staticmethod
	def derivation_function(x: np.ndarray) -> np.ndarray:
		return 1 - np.square(np.tanh(x))


class Softmax(ActivationFunction):

	@staticmethod
	def function(x: np.ndarray) -> np.ndarray:
		return np.exp(x) / np.exp(x).sum()

	@staticmethod
	def derivation_function(x: np.ndarray) -> np.ndarray:
		return 1 - Softmax.function(x)
