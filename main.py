import os
import numpy as np
import matplotlib.pyplot as plt
from nn import Linear, ReLU, MSELoss

LR = 0.001

class MLP():
    def __init__(self) -> None:
        self._layers = [
            Linear(2, 8),
            ReLU(),
            Linear(8, 1),
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
    os.makedirs("results", exist_ok=True)

    dataset = [
        (np.array([[0],[0]]), np.array([0])),
        (np.array([[0],[1]]), np.array([1])),
        (np.array([[1],[0]]), np.array([1])),
        (np.array([[1],[1]]), np.array([0])),
    ]

    mlp = MLP()
    loss_fn = MSELoss()

    epoch_losses = []

    for step in range(1000):
        step_losses = []
        for input_data, target_data in dataset:
            prediction = mlp.forward(input_data)
            loss = loss_fn.forward(prediction, target_data)
            grad = loss_fn.backward()
            mlp.backward(grad)
            mlp.update_linear_layers()
            step_losses.append(loss)

        epoch_avg_loss = np.mean(step_losses)
        epoch_losses.append(epoch_avg_loss)

        if step % 50 == 0:
            print(epoch_avg_loss)

    for input_data, target_data in dataset:
        prediction = mlp.forward(input_data)
        predicted_class = 0 if prediction[0, 0] < 0.5 else 1
        print(f"Prediction: {predicted_class} | Target: {target_data}")

    # Loss curve
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Training loss over time")
    plt.savefig("results/loss_curve.png")

    # Decision boundary
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)

    grid_predictions = np.zeros_like(xx1)
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            point = np.array([[xx1[i, j]], [xx2[i, j]]])
            output = mlp.forward(point)
            grid_predictions[i, j] = 1 if output[0, 0] > 0.5 else 0

    plt.figure()
    plt.contourf(xx1, xx2, grid_predictions, levels=[-0.5, 0.5, 1.5],
                 colors=["lightblue", "salmon"], alpha=0.6)

    for input_data, target_data in dataset:
        x1, x2 = input_data[0, 0], input_data[1, 0]
        color = "blue" if target_data[0] == 0 else "red"
        plt.scatter(x1, x2, c=color, s=100, edgecolors="black")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("XOR decision boundary")
    plt.savefig("results/xor_decision_boundary.png")


if __name__ == "__main__":
    main()
