import numpy as np
from nn import Linear

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
    print(f"output: {test1.output_data(input_data)}")


if __name__ == "__main__":
    main()
