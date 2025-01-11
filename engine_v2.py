from typing import Union, List, Tuple

import numpy as np  # type: ignore


class Tensor:
    def __init__(self, data, children=()):
        self.data = np.array(data) if hasattr(data, "__len__") else np.array([data])
        self.shape = self.data.shape
        self.grad = 0.0
        self.children = children 
        self._backward = lambda: None
        self._current_index = 0

    def __getitem__(self, indices: Union[List, int, Tuple, slice]):
        return self.data[indices]

    def __add__(self, other):
        assert self.shape == other.shape, "Different shape"
        output = Tensor(self.data + other.data, (self, other))

        def backward():
            self.grad += 1 * output.grad
            other.grad += 1 * output.grad

        output._backward = backward
        return output

    def __mul__(self, other):
        print(self.shape, other.shape)
        output = Tensor(np.matmul(self.data, other.data), (self, other))

        def backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = backward

        return output

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad}, shape={self.shape})"

    def sum(self):
        output = Tensor(self.data.sum(), children=(self,))
        def backward():
            self.grad += 1 * output.grad

        output._backward = backward

        return output
    


    def backward(self):
        graph = []
        visited = set()

        def create_graph(root: Tensor):
            if root not in visited:
                visited.add(root)
                for child in root.children:
                    create_graph(child)
                graph.append(root)

        create_graph(self)
        self.grad = 1.0

        for child in reversed(graph):
            child._backward()


if __name__ == "__main__":
    a = Tensor([[1, 2, 3], [4, 5, 7]])
    b = Tensor([[4, 2, 3], [8, 3, 2]])
    c = a + b
    d = c.sum()

    # print(d)

    d.backward()

    print(a, b)
    # print(d)
