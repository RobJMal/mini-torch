import numpy as np
from nn import Linear, ReLU

class MLP():
    def __init__(self, seed=137) -> None:
        self._layers = [
            Linear(2, 2),
            ReLU(),
        ]

        self._layers[0].update_weights(np.array([[1, 2], [3, 1]]))
        self._layers[0].update_bias(np.array([[1], [-1]]))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data
        for i, layer in enumerate(self._layers):
            output = layer.forward(output)

        return output


def main():
    mlp = MLP()

    input_data = np.array([[-2], [4]])
    output_test1 = mlp.forward(input_data)
    print(f"output: {output_test1}")
    print("")


if __name__ == "__main__":
    main()
