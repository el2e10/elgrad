from typing import Union, List, Tuple

import numpy as np  # type: ignore

class BroadcastError(Exception):

    def __init__(self, shape_1, shape_2):
        self.message = f"Cannot broadcaset items of shape {shape_1} and {shape_2}"
        super().__init__(self.message)


class Tensor:
    def __init__(self, data, children=(), require_grad=False):
        self.data = np.array(data) if hasattr(data, "__len__") else np.array([data])
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        self.require_grad = require_grad
        self.grad = Tensor.zeros(self.shape) if require_grad else 0.0
        self.children = children 
        self._backward = lambda: None
        self._current_index = 0


    @staticmethod
    def ones(shape: Union[int, Tuple, List[int]]):
        return Tensor(np.ones(shape))

    @staticmethod
    def zeros(shape: Union[int, Tuple, List[int]]):
        return Tensor(np.zeros(shape))


    @staticmethod
    def broadcast(x, y):
        """Return the shape of the broadcast array if it's possible or raise a BroadcastError

        x -- Tensor
        y -- Tensor
        """
        x_shape = x.shape
        y_shape = y.shape

        x_rank = len(x_shape)
        y_rank = len(y_shape)

        max_rank = max(x_rank, y_rank)

        shape = [0] * max_rank
        i = max_rank -1

        while(i >= 0):
            n1 = x_rank - max_rank + i
            d1 = x_shape[n1] if(n1 >= 0) else 1

            n2 = y_rank - max_rank + i
            d2 = y_shape[n2] if(n2 >= 0) else 1

            if(d1 == 1):
                shape[i] = d2
            elif(d2 == 1):
                shape[i] = d1
            elif(d1 == d2):
                shape[i] = d1
            else:
                raise BroadcastError(x_shape, y_shape)

            i -= 1

        return tuple(shape)

    def __setitem__(self, indices, value):
        self.data[indices] = value

    def __getitem__(self, indices: Union[List, int, Tuple, slice]):
        return self.data[indices]

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # The broadcast method will raise a broadcast exception if it's not broadcastable
        Tensor.broadcast(self.data, other.data)

        output = Tensor(self.data + other.data, (self, other))

        def backward():
            self.grad += 1 * output.grad
            other.grad += 1 * output.grad

        output._backward = backward
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Checking if the mat mul can be performed
        # Refer for more info https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html#array_api.matmul
        assert self.shape[-1] == other.shape[-min(other.ndim, 2)], "Invalid dimension cannot do dot product"
        
        
        print(self.data.shape, other.data.shape)
        output = Tensor(np.matmul(self.data, other.data), children=(self, other))

        def backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = backward

        return output

    def __repr__(self) -> str:
        grad = self.grad.data if(isinstance(self.grad, Tensor)) else self.grad
        return f"Tensor(data={self.data}, grad={grad}, shape={self.shape})\n"

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

    a = Tensor([[1, 2, 3], [4, 5, 7]], require_grad=True)
    b = Tensor([[4, 2, 3], [8, 3, 2], [1, 2, 4]], require_grad=True)

    c = a * b

    # d = c.sum()
    #
    # d.backward()
    #
    # print(a, b)


