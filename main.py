import numpy as np
from nn import Linear, ReLU, MSELoss

LR = 0.001

class MLP():
    def __init__(self) -> None:
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

    def backward(self, grad_input: np.ndarray) -> None:
        layer_grad = grad_input
        for layer in reversed(self._layers):
            layer_grad = layer.backward(layer_grad)

    def update_linear_layers(self) -> None:
        for layer in self._layers:
            if type(layer) is Linear:
                layer.weights = layer.weights - LR * layer.weights_grad
                layer.bias = layer.bias - LR * layer.bias_grad

def main():
    input_data = np.array([[1], [0.5]])
    target_data = np.array([[0.5], [2]])

    mlp = MLP()
    output_test1 = mlp.forward(input_data)
    print(f"output: {output_test1}")
    print("")

    loss_fn = MSELoss()
    loss = loss_fn.forward(output_test1, target_data)
    grad = loss_fn.backward()
    mlp.backward(grad)
    mlp.update_linear_layers()

if __name__ == "__main__":
    main()
