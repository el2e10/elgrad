from typing import Union, List, Tuple
from functools import reduce

import numpy as np  # type: ignore

class BroadcastError(Exception):

    def __init__(self, shape_1, shape_2):
        self.message = f"Cannot broadcaset items of shape {shape_1} and {shape_2}"
        super().__init__(self.message)

class DotProductError(Exception):

    def __init__(self):
        self.message = "Cannot do matrix multiplication"
        super().__init__(self.message)

class Tensor:
    def __init__(self, data, children=(), require_grad=False, label=""):
        # print("The data is ", data, np.array(data))
        self.data = np.array(data, dtype=float) if hasattr(data, "__len__") else np.array([data], dtype=float)
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        self.require_grad = require_grad or any([child.require_grad for child in children])
        self.grad = Tensor.zeros(self.shape) if self.require_grad else None
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

    def T(self):
        return Tensor(np.transpose(self.data))


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
        # print("In add", other)
        other = other if isinstance(other, Tensor) else Tensor(other)
        # The broadcast method will raise a broadcast exception if it's not broadcastable
        Tensor.can_broadcast(self.data, other.data)

        # print("Add is", self, other)
        output = Tensor(self.data + other.data, (self, other), label="Add")

        # print("Result", self.data, other.data, output,"End")

        def backward():
            # print("first add grad")
            if(self.require_grad):
                self.grad += Tensor._sum_if_broadcasting_occured(output.grad, self.grad) 
            # print("second add grad", output, other, self)
            if(other.require_grad):
                other.grad += Tensor._sum_if_broadcasting_occured(output.grad, other.grad)

        output._backward = backward
        return output

    def __rsub__(self, other):
        return self.__sub__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(np.power(self.data, other.data), children=(self, ), label="pow")

        def backward():
            self.grad += other.data * (self.data**(other.data - 1)) * output.grad.data #type: ignore

        output._backward = backward
        return output

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Checking if the mat mul can be performed
        # Refer for more info https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html#array_api.matmul
        if(self.ndim == 1 and other.ndim > 1 and self.shape[-1] == other.shape[-2]):
            pass
        elif(self.shape[-1] == other.shape[-min(other.ndim, 2)]):
            pass
        else:
            raise DotProductError
        
        output = Tensor(np.matmul(self.data, other.data), children=(self, other), label="matmul")

        def backward():

            if(len(output.grad.shape) == 1): #type: ignore
                output_grad = Tensor.expand_dims(output.grad, 1)
            else:
                output_grad = output.grad
            
            if(self.require_grad):
                tmp = Tensor._broadcast_for_gradient(output.grad, self, other)
                # print(f"Tmp is {tmp}, {tmp.T()}, {output_grad}, {output_grad @ tmp.T()}\n")
                self.grad += Tensor._sum_if_broadcasting_occured(output_grad @ tmp.T(), self.grad)
            if(other.require_grad):
                tmp = Tensor._broadcast_for_gradient(output.grad, other, self)
                # print("Tmp is ", tmp)
                other.grad += Tensor._sum_if_broadcasting_occured(tmp.T() @ output_grad, other.grad)

        output._backward = backward

        return output

    @staticmethod
    def compare_input_and_gradient(input, grad):
        pass

    @staticmethod
    def expand_dims(tensor, axis):
        return Tensor(np.expand_dims(tensor.data, axis=axis))

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        return self * (other**-1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        other = Tensor.full(self.shape, other) if isinstance(other, (int, float)) else other
        # print("Other is ",other)

        # Check if we the two tensors can be broadcasted. Raise a BroadcastError exception if not possible
        # print("THe other is", self, other)
        Tensor.can_broadcast(self, other)

        output = Tensor(self.data * other.data, label="M.out", children=(self, other))

        def backward():
            
            if(self.require_grad):
                self.grad += Tensor._sum_if_broadcasting_occured(other * output.grad, self.grad)
            if(other.require_grad):
                other.grad += Tensor._sum_if_broadcasting_occured(self * output.grad, other.grad)

        output._backward = backward

        return output

    def __repr__(self) -> str:
        grad = self.grad.data if(isinstance(self.grad, Tensor)) else self.grad
        return f"\n| Tensor(data={self.data}, grad={grad}, shape={self.shape}, label={self.label}, require_grad={self.require_grad})"

    def sum(self, axis=None):
        output = Tensor(self.data.sum(axis), children=(self,))
        def backward():
            # Sum function always assing 1 as gradient to each element
            # print("first sum grad")
            if(self.require_grad):
                self.grad += Tensor.ones(self.shape)

        output._backward = backward

        return output

    @staticmethod
    def _broadcast_for_gradient(prev_gradient, current, other):
        try:
            Tensor.can_broadcast(prev_gradient, other)
            return prev_gradient * other
        except BroadcastError:
            # print("Cannot broadcast. Will require manual broadcasting", prev_gradient, other.data)
            pass

        g_shape, c_shape, o_shape = prev_gradient.shape, current.shape, other.shape
        broadcast_dimension = Tensor.can_broadcast(current, other)
        # print("\n \t Broadcast details")
        # print(g_shape, c_shape, o_shape)
        # print("Output is ",other)
        # print("Input is", current)
        # print("Broadcast dimension ", Tensor.can_broadcast(current, other))
        # print("\n")
        if(broadcast_dimension == c_shape):
            result = Tensor(np.broadcast_to(other.data, c_shape))
        else:
            result = other 
        return result

    @staticmethod
    def reshape(tensor, shape):
        data = tensor.data
        return Tensor(np.reshape(data, shape))

    @staticmethod
    def _sum_if_broadcasting_occured(x, y):
        x_shape, y_shape = x.shape if hasattr(x, "shape") else (1, ), y.shape if hasattr(y, "shape") else (1, ) 
        x_dim, y_dim = len(x_shape), len(y_shape)
        difference = abs(x_dim - y_dim)
        broadcast_axis = []

        if(x_dim > y_dim):
            y_shape = (1,) * difference + y_shape
        elif(x_dim < y_dim):
            x_shape = (1,) * difference + x_shape
        else:
            return x

        i = 0
        for k, j in zip(x_shape, y_shape):
            if(k != j):
                broadcast_axis.append(i)
            i += 1
        return x.sum(axis=tuple(broadcast_axis))

    @staticmethod
    def full(shape, fill_value):
        return Tensor(np.full(shape, fill_value)) 


    @staticmethod
    def fill_empty(tensor, target_shape, fill_value:Union[int, float]=0.0):
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
            if(child.require_grad):
                child._backward()


if __name__ == "__main__":


    a = Tensor([[1, 2, 3], [4, 5,6], [7, 8, 9]], require_grad=True, label="A")
    b = Tensor([[9, 8, 7], [6, 5, 4], [3, 2, 1]], require_grad=True, label="B")
    # b = Tensor([9, 8, 7], require_grad=True, label="B")

    c = a @ b
    d = c.sum()
    d.backward()
    print("Final output is", a, b)
