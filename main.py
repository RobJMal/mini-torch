import numpy as np
from nn import Linear, ReLU, MSELoss

class MLP():
    def __init__(self, seed=137) -> None:
        self._layers = [
            Linear(2, 2),
            ReLU(),
            Linear(2, 2),
        ]

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

    mse_loss = MSELoss()
    input_data = np.array([[1], [0.5]])
    target_data = np.array([[0.5], [2]])

    print(f"MSE forward: {mse_loss.forward(input_data, target_data)}")
    print(f"MSE backward: {mse_loss.backward()}")

if __name__ == "__main__":
    main()
