import numpy as np

class Linear():
    def __init__(self, in_features: int, out_features: int) -> None:
        self._weights = np.zeros((out_features, in_features))
        self._bias = np.zeros((out_features, 1))

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    def update_weights(self, value) -> None:
        self._check_shape(value, self._weights)
        self._weights = value

    def update_bias(self, value: np.ndarray) -> None:
        self._check_shape(value, self._bias)
        self._bias = value

    def output_data(self, input_data: np.ndarray) -> np.ndarray:
        self._check_shape(input_data, self._bias)
        return self._weights @ input_data + self.bias

    def _check_shape(self, data: np.ndarray, expected: np.ndarray) -> None:
        """Checks shape of inputs to ensure valid multiplication
        """
        if expected.shape != data.shape:
            raise ValueError(f"new weight updates is invalid size. Got {data.shape}, expected {expected.shape}")
