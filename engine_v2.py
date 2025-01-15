from typing import Union, List, Tuple
from functools import reduce

import numpy as np  # type: ignore

class BroadcastError(Exception):

    def __init__(self, shape_1, shape_2):
        self.message = f"Cannot broadcaset items of shape {shape_1} and {shape_2}"
        super().__init__(self.message)


class Tensor:
    def __init__(self, data, children=(), require_grad=False, label=""):
        self.data = np.array(data) if hasattr(data, "__len__") else np.array([data])
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        self.require_grad = require_grad
        self.grad = Tensor.zeros(self.shape) if require_grad else 0.0
        self.children = children 
        self.label = label
        self._backward = lambda: None
        self._current_index = 0


    @staticmethod
    def ones(shape: Union[int, Tuple, List[int]]):
        return Tensor(np.ones(shape))

    @staticmethod
    def zeros(shape: Union[int, Tuple, List[int]]):
        return Tensor(np.zeros(shape))


    @staticmethod
    def can_broadcast(x, y):
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

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # The broadcast method will raise a broadcast exception if it's not broadcastable
        Tensor.can_broadcast(self.data, other.data)

        output = Tensor(self.data + other.data, (self, other))

        def backward():
            self.grad += Tensor._sum_if_broadcasting_occured(output.grad, self.grad) 
            other.grad += Tensor._sum_if_broadcasting_occured(output.grad, other.grad)

        output._backward = backward
        return output


    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Checking if the mat mul can be performed
        # Refer for more info https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html#array_api.matmul
        assert self.shape[-1] == other.shape[-min(other.ndim, 2)], "Invalid dimension cannot do dot product"
        
        output = Tensor(np.matmul(self.data, other.data), children=(self, other))

        def backward():
            tmp = other.data * output.grad
            self.grad = Tensor.fill_empty(self.grad, tmp.shape) + tmp
            tmp = other.data * output.grad
            other.grad = Tensor.fill_empty(other.grad, tmp.shape) + tmp

        output._backward = backward

        return output

    def __mul__(self, other):
        other = Tensor.fill_empty(Tensor(other), self.shape, fill_value=other) if isinstance(other, (int, float)) else other

        # Check if we the two tensors can be broadcasted. Raise a BroadcastError exception if not possible
        Tensor.can_broadcast(self, other)

        output = Tensor(self.data * other.data, label="M.out")

        def backward():
            
            self.grad += Tensor._sum_if_broadcasting_occured(other * output.grad, self.grad)
            other.grad += Tensor._sum_if_broadcasting_occured(self * output.grad, other.grad)

        output._backward = backward

        return output

    def __repr__(self) -> str:
        grad = self.grad.data if(isinstance(self.grad, Tensor)) else self.grad
        return f"\nTensor(data={self.data}, grad={grad}, shape={self.shape}, label={self.label})"

    def sum(self, axis=None):
        output = Tensor(self.data.sum(axis), children=(self,))
        def backward():
            padd_shape = reduce(lambda x, y: x * y, self.shape) if axis is None else self.shape
            result = Tensor.fill_empty(self.grad, padd_shape) + Tensor.ones(padd_shape)
            result = Tensor.reshape(result, self.shape)
            # print(result)
            self.grad = result

        output._backward = backward

        return output

    @staticmethod
    def reshape(tensor, shape):
        data = tensor.data
        return Tensor(np.reshape(data, shape))

    @staticmethod
    def _sum_if_broadcasting_occured(x, y):
        x_shape, y_shape = x.shape, y.shape
        x_dim, y_dim = len(x.shape), len(y.shape)
        difference = abs(x_dim - y_dim)
        broadcast_axis = []

        if(x_dim > y_dim):
            y_shape = (1,) * difference + y_shape
        elif(x_dim < y_dim):
            x_shape = (1,) * difference + x_shape
        else:
            return x

        # broadcast_axis = tuple([abs(x - y) for x, y in zip(x_shape, y_shape)])
        i = 0
        for k, j in zip(x_shape, y_shape):
            if(k != j):
                broadcast_axis.append(i)
            i += 1
        return x.sum(axis=tuple(broadcast_axis))

    @staticmethod
    def fill_empty(tensor, target_shape, fill_value:Union[int, float]=0.0):
        tensor = Tensor(tensor) if not isinstance(tensor, Tensor) else tensor
        if(isinstance(target_shape, int)):
            pad = [0,  target_shape - tensor.shape[0]]
        else:
            assert tensor.ndim == len(target_shape), "Can't pad this tensor"
            pad = [(0, abs(x - y)) for x, y in zip(tensor.shape, target_shape)] #type: ignore
        
        padded_array = np.pad(tensor.data, pad, constant_values=fill_value)
        return Tensor(padded_array)

    
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


    a = Tensor([[2, 3, 4], [1, 2, 3]], require_grad=True, label="A")
    b = Tensor([4, 2, 3], require_grad=True, label="B")

    c = a + b
    d = c.sum()

    d.backward()
    print(a, b)

