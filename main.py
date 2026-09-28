import numpy as np
from nn import Linear, ReLU

def main():
    print("Hello from mini-torch!")
    test1 = Linear(2, 2)
    print(f"weights: {test1.weights}")
    print(f"bias: {test1.bias}")
    print("")

    test1.update_weights(np.array([[1, 2], [3, 1]]))
    test1.update_bias(np.array([[1], [-1]]))

    print(f"updated weights: {test1.weights}")
    print(f"updated bias: {test1.bias}")
    print("")

    input_data = np.array([[-2], [4]])
    output_test1 = test1.forward(input_data)
    print(f"output: {output_test1}")
    print("")

    relu1 = ReLU()
    hlayer1 = relu1.forward(output_test1)
    print(f"hidden layer1 output: {hlayer1}")


if __name__ == "__main__":
    main()
