from nn import Linear

def main():
    print("Hello from mini-torch!")
    test1 = Linear(10, 10)
    print(type(test1.weights))
    print(test1.weights)


if __name__ == "__main__":
    main()
