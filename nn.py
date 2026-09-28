import numpy as np

class Linear():
    def __init__(self, in_features: int, out_features: int) -> None:
        self._weights = np.zeros((in_features, out_features))
        self._bias = np.zeros((1, in_features))

    @property
    def weights(self) -> np.ndarray:
        return self._weights
