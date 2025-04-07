from math import e, prod, floor
from typing import Union, List, Tuple

import numpy as np  # type: ignore

from helper import check_conv2d_stride_shape


class BroadcastError(Exception):
    def __init__(self, shape_1, shape_2):
        self.message = f"Cannot broadcast items of shape {shape_1} and {shape_2}"
        super().__init__(self.message)


class DotProductError(Exception):
    def __init__(self):
        self.message = "Cannot do matrix multiplication"
        super().__init__(self.message)


class Tensor:
    def __init__(self, data, children=(), require_grad=None, label=""):
        self.data = np.array(data, dtype=float) if hasattr(data, "__len__") else np.array([data], dtype=float)
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        if require_grad or require_grad is None:
            self.require_grad = require_grad or any([child.require_grad for child in children])
        else:
            self.require_grad = False
        self.grad = Tensor.zeros(self.shape, False, label=f"{label} grad") if self.require_grad else None
        self.children = children
        self.label = label
        self._backward = lambda: None
        self._current_index = 0

    def zero_grad(self):
        self.grad = Tensor.zeros(self.grad.shape, require_grad=False) if self.require_grad else None  # type: ignore

    @staticmethod
    def ones(shape: Union[int, Tuple, List[int]]):
        return Tensor(np.ones(shape))

    @staticmethod
    def zeros(shape: Union[int, Tuple, List[int]], require_grad, label=""):
        return Tensor(np.zeros(shape), require_grad=require_grad, label=label)

    def T(self):
        # data = np.atleast_2d(self.data) if(len(self.shape) == 1) else self.data
        axes = tuple(range(self.data.ndim)[:-2]) + (-1, -2) if self.data.ndim > 2 else None
        data = np.transpose(self.data, axes=axes)
        output = Tensor(data, children=(self,), label=f"{self.label}.T")

        def backward():
            grad = output.grad.data  # type: ignore
            self.grad = Tensor(np.transpose(grad))

        output._backward = backward
        return output

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
        i = max_rank - 1

        while i >= 0:
            n1 = x_rank - max_rank + i
            d1 = x_shape[n1] if (n1 >= 0) else 1

            n2 = y_rank - max_rank + i
            d2 = y_shape[n2] if (n2 >= 0) else 1

            if d1 == 1:
                shape[i] = d2
            elif d2 == 1:
                shape[i] = d1
            elif d1 == d2:
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

    def log(self):
        output = Tensor(np.log(self.data), children=(self,), label="log")

        def backward():
            self.grad += Tensor._sum_if_broadcasting_occured(output.grad * Tensor(1 / self.data), output.grad)

        output._backward = backward

        return output

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # The broadcast method will raise a broadcast exception if it's not broadcastable
        Tensor.can_broadcast(self.data, other.data)

        output = Tensor(self.data + other.data, (self, other), label="Add")

        def backward():
            if self.require_grad:
                self.grad += Tensor._sum_if_broadcasting_occured(output.grad, self.grad)
            if other.require_grad:
                other.grad += Tensor._sum_if_broadcasting_occured(output.grad, other.grad)

        output._backward = backward
        return output

    def __rsub__(self, other):
        return self.__sub__(other)

    def __sub__(self, other):
        # return self.__add__(-1 * other)
        other = other if isinstance(other, Tensor) else Tensor(other)
        # The broadcast method will raise a broadcast exception if it's not broadcastable
        Tensor.can_broadcast(self.data, other.data)

        output = Tensor(self.data - other.data, (self, other), label="Add")

        def backward():
            if self.require_grad:
                self.grad += Tensor._sum_if_broadcasting_occured(output.grad, self.grad)
            if other.require_grad:
                other.grad += Tensor._sum_if_broadcasting_occured(-output.grad, other.grad)  # type: ignore

        output._backward = backward
        return output

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # The broadcast method will raise a broadcast exception if it's not broadcastable
        Tensor.can_broadcast(self.data, other.data)

        output = Tensor(np.power(self.data, other.data), children=(self, other), label="pow")

        def backward():
            if self.require_grad:
                self.grad += Tensor._sum_if_broadcasting_occured(
                    other.data * (self.data ** (other.data - 1)) * output.grad.data,
                    self.grad,
                )  # type: ignore
            if other.require_grad:
                other.grad += Tensor._sum_if_broadcasting_occured(
                    np.log(self.data) * (self.data**other.data) * output.grad.data,
                    other.grad,
                )  # type: ignore

        output._backward = backward
        return output

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        original_self, original_other = self, other

        # Refer for more info https://pytorch.org/docs/stable/generated/torch.matmul.html
        if self.ndim == 1 and other.ndim >= 2:
            self = Tensor.expand_dims(self, 0)
        elif self.ndim >= 2 and other.ndim == 1:
            other = Tensor.expand_dims(other, 1)

        # Checking if the mat mul can be performed
        # Refer for more info https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html#array_api.matmul
        if self.ndim == 1 and other.ndim > 1 and self.shape[-1] == other.shape[-2]:
            pass
        elif self.shape[-1] == other.shape[-min(other.ndim, 2)]:
            pass
        else:
            raise DotProductError

        output = Tensor(np.matmul(self.data, other.data), children=(self, other), label="matmul")

        self, other = original_self, original_other

        def backward():
            output_grad = Tensor.expand_dims(output.grad, 1) if (output.grad.ndim == 1) else output.grad  # type: ignore
            ne_other = Tensor.reshape(other, (other.shape[0], 1)) if (other.ndim == 1) else other
            self_data = Tensor.expand_dims(self.data, 0) if (self.data.ndim == 1) else self

            if self.require_grad:
                self.grad += Tensor._sum_if_broadcasting_occured(output_grad @ ne_other.T(), self.grad).reshape(self.grad.shape)  # type: ignore
            if other.require_grad:
                ne_other.grad += Tensor._sum_if_broadcasting_occured(self_data.T() @ output_grad, ne_other.grad)
                other.grad = ne_other.grad.reshape(other.grad.shape)  # type: ignore

        output._backward = backward

        return output

    @staticmethod
    def expand_dims(tensor, axis):
        # grad = None if(tensor.grad is None) else
        tensor = tensor if (isinstance(tensor, Tensor)) else Tensor(tensor)
        return Tensor(
            np.expand_dims(tensor.data, axis=axis),
            label=tensor.label,
            require_grad=tensor.require_grad,
            children=tensor.children,
        )

    def __rtruediv__(self, other):
        other = Tensor(other)
        return other.__truediv__(self)

    def __truediv__(self, other):
        other = Tensor.full(self.shape, other) if isinstance(other, (int, float)) else other

        # Check if we the two tensors can be broadcasted. Raise a BroadcastError exception if not possible
        Tensor.can_broadcast(self, other)

        output = Tensor(self.data / other.data, label="Div", children=(self, other))

        def backward():
            if self.require_grad:
                self.grad += Tensor._sum_if_broadcasting_occured((1 / other) * output.grad, self.grad)
            if other.require_grad:
                other.grad += Tensor._sum_if_broadcasting_occured((-self / other**2) * output.grad, other.grad)

        output._backward = backward

        return output
        # return self * (other**-1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        other = Tensor.full(self.shape, other) if isinstance(other, (int, float)) else other

        # Check if we the two tensors can be broadcasted. Raise a BroadcastError exception if not possible
        Tensor.can_broadcast(self, other)

        output = Tensor(self.data * other.data, label="Mul", children=(self, other))

        def backward():
            if self.require_grad:
                self.grad += Tensor._sum_if_broadcasting_occured(other * output.grad, self.grad)
            if other.require_grad:
                other.grad += Tensor._sum_if_broadcasting_occured(self * output.grad, other.grad)

        output._backward = backward

        return output

    def __repr__(self) -> str:
        grad = self.grad.data if (isinstance(self.grad, Tensor)) else self.grad
        return f"\n| Tensor(data={self.data}, grad={grad}, shape={self.shape}, label={self.label}, require_grad={self.require_grad})"

    def sum(self, axis=None, keepdims=False):
        output = Tensor(self.data.sum(axis, keepdims=keepdims), children=(self,), label="Sum")

        def backward():
            # Sum function always assing 1 as gradient to each element
            if self.require_grad:
                output_grad = Tensor.expand_dims(output.grad, axis) if (not keepdims and axis is not None) else output.grad
                self.grad += Tensor.ones(self.shape) * output_grad

        output._backward = backward

        return output

    def mean(self, axis=None, keepdims=False):
        output = Tensor(self.data.mean(axis, keepdims=keepdims), children=(self,), label="Mean")

        def backward():
            if self.require_grad:
                v = prod(self.shape) if axis is None else self.shape[axis]
                output_grad = Tensor.expand_dims(output.grad, axis) if (axis is not None and not keepdims) else output.grad
                self.grad += Tensor.full(self.shape, 1 / v) * output_grad

        output._backward = backward

        return output

    def exp(self):
        output = Tensor(np.exp(self.data), children=(self,), label="Exp")

        def backward():
            if self.require_grad:
                self.grad += Tensor(output.data) * output.grad

        output._backward = backward

        return output

    def softmax(self, dim=0):
        numerator = self.exp()
        output = numerator / numerator.sum(dim, keepdims=True)
        output.label = "Softmax"
        return output

    def relu(self):
        _relu_out = np.maximum(self.data, 0)
        output = Tensor(_relu_out, children=(self,), label="ReLU")

        def backward():
            _grad_out = (output.data > 0).astype(float)
            self.grad += Tensor(_grad_out) * output.grad

        output._backward = backward

        return output

    def reshape(self, shape):
        data = self.data
        output = Tensor(np.reshape(data, shape), children=(self,), label="reshape")
        original_shape = self.shape

        def backward():
            if self.require_grad:
                self.grad += output.grad.reshape(original_shape)  # type: ignore

        output._backward = backward

        return output

    def flatten(self):
        data = self.data
        output = Tensor(data.flatten(), children=(self,), label="flatten")
        original_shape = self.shape

        def backward():
            if self.require_grad:
                self.grad += output.grad.data.flatten(original_shape)  # type: ignore

        output._backward = backward

        return output

    def conv2d(self, filter, stride=1, padding=0):
        status, message = check_conv2d_stride_shape(self, filter, stride)
        assert status, message

        stride = tuple([stride, stride]) if isinstance(stride, int) else stride

        filter = Tensor(filter) if (not isinstance(filter, Tensor)) else filter
        i_h, i_w, f_h, f_w = (*self.shape, *filter.shape)  # type: ignore
        new_filter = [(slice(i, j), slice(k, l)) for i, j in zip(range(0, i_h - f_h + 1, stride[0]), range(f_h, i_h + 1, stride[0])) for k, l in zip(range(0, i_w - f_w + 1, stride[1]), range(f_w, i_w + 1, stride[1]))]
        index = np.r_[tuple(new_filter)]
        # result = np.array([(self.data[index[i], index[i + 1]]).flatten() for i in range(0, len(index), 2)])
        result = Tensor(
            [(self[index[i], index[i + 1]]).flatten() for i in range(0, len(index), 2)],
            require_grad=self.require_grad,
            label="conv inter",
            children=(self,),
        )

        o_h, o_w = (
            floor((i_h + (2 * padding) - f_h) / stride[0] + 1),
            floor((i_w + (2 * padding) - f_w) / stride[1] + 1),
        )
        # print(o_h,o_w)

        return (result @ filter.flatten()).reshape((o_h, o_w))

    @staticmethod
    def _sum_if_broadcasting_occured(x, y):
        # If broadcasting occured between two tensors during an operation we have to sum the gradient along the axis where broadcasting occured
        x_shape, y_shape = (
            x.shape if hasattr(x, "shape") else (1,),
            y.shape if hasattr(y, "shape") else (1,),
        )
        x_dim, y_dim = len(x_shape), len(y_shape)
        difference = abs(x_dim - y_dim)
        broadcast_axis = []

        if x_dim > y_dim:
            y_shape = (1,) * difference + y_shape
        elif x_dim < y_dim:
            x_shape = (1,) * difference + x_shape

        broadcasted_shape = Tensor.can_broadcast(x, y)
        i = 0
        for k, j, m in zip(x_shape, y_shape, broadcasted_shape):
            if len(set([k, j, m])) != 1:
                broadcast_axis.append(i)
            i += 1

        # Need to reshape because the grad should have the same shape as vector
        return x.sum(axis=tuple(broadcast_axis)).reshape(y.shape) if (len(broadcast_axis) > 0) else x

    @staticmethod
    def full(shape, fill_value):
        return Tensor(np.full(shape, fill_value))

    @staticmethod
    def fill_empty(tensor, target_shape, fill_value: Union[int, float] = 0.0):
        if isinstance(target_shape, int):
            pad = [0, target_shape - tensor.shape[0]]
        else:
            assert tensor.ndim == len(target_shape), "Can't pad this tensor"
            pad = [(0, abs(x - y)) for x, y in zip(tensor.shape, target_shape)]  # type: ignore

        padded_array = np.pad(tensor.data, pad, constant_values=fill_value)
        return Tensor(padded_array)

    @staticmethod
    def array_equal(first, second) -> bool:
        if not isinstance(second, (float, int, list, tuple)):
            second = second.data

        if not isinstance(first, (float, int, list, tuple)):
            first = first.data
        # Rounding the value to 4 decimals so that it is easier to compare the gradient values during testing
        return np.array_equal(np.round(first, 4), np.round(second, 4))

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
            if child.require_grad:
                child._backward()


if __name__ == "__main__":
    img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], require_grad=True, label="Img")
    filter = Tensor([[1, 2], [3, 4]], require_grad=True, label="Filter")
    result = img.conv2d(filter)
    sum = result.sum()
    sum.backward()
    print(img, filter, sum)
