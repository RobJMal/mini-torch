import numpy as np

RNG_SEED = 137
rng = np.random.default_rng(RNG_SEED)

class Linear():
    def __init__(self, in_features: int, out_features: int) -> None:
        # Need to initialize weights randomly so network can learn (symmetric init causes issues)
        self._weights = rng.standard_normal((out_features, in_features))
        self._bias = np.zeros((out_features, 1))

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self._check_shape(input_data, self._bias)
        self._input_data = input_data
        return self._weights @ input_data + self.bias

    def backward(self, prev_layer_grad: np.ndarray) -> np.ndarray:
        self._grad_weights = np.outer(prev_layer_grad, self._input_data.T) 
        self._grad_bias = prev_layer_grad
        return self._weights.T @ prev_layer_grad

    def update_weights(self, value) -> None:
        self._check_shape(value, self._weights)
        self._weights = value

    def update_bias(self, value: np.ndarray) -> None:
        self._check_shape(value, self._bias)
        self._bias = value

    def _check_shape(self, data: np.ndarray, expected: np.ndarray) -> None:
        """Checks shape of inputs to ensure valid multiplication
        """
        if expected.shape != data.shape:
            raise ValueError(f"new weight updates is invalid size. Got {data.shape}, expected {expected.shape}")

class ReLU():
    def __init__(self) -> None:
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self._output = np.fmax(0, input_data)
        return np.fmax(0, input_data)

    def backward(self, prev_layer_grad: np.ndarray) -> np.ndarray:
        grad = self._output.copy()
        grad[grad > 0.0] = 1.0
        grad = grad * prev_layer_grad   
        return grad

class MSELoss():
    def __init__(self) -> None:
        pass

    def forward(self, input_data: np.ndarray, target_data: np.ndarray) -> float:
        self._diff_data = input_data - target_data
        return np.square(self._diff_data).mean()

    def backward(self) -> np.ndarray:
        return self._diff_data * 2 / len(self._diff_data)