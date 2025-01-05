import math
import random
from typing import List


class Value:
    def __init__(self, data, children=(), label="") -> None:
        self.data = data
        self.grad = 0.0
        self.children = set(children)
        self._backward = lambda: None
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # print("Gradient add", output.grad)
            self.grad += 1 * output.grad
            other.grad += 1 * output.grad

        output._backward = _backward
        return output

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # print("Gradient mult", output.grad * other.data)
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only support int and float"
        output = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad = other * (self.data ** (other - 1)) * output.grad

        output._backward = _backward

        return output

    def __exp__(self):
        output = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += output.grad * output.data

        output._backward = _backward

        return output

    def tanh(self):
        x = self.data
        tanh_value = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Value(tanh_value, (self,), "tanh")

        def _backward():
            self.grad += (1 - tanh_value**2) * output.grad

        output._backward = _backward

        return output

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def backward(self):
        graph = []
        visited = set()

        def create_graph(root: Value):
            if root not in visited:
                visited.add(root)
                for child in root.children:
                    create_graph(child)
                graph.append(root)

        create_graph(self)
        self.grad = 1.0

        for child in reversed(graph):
            child._backward()


class Neuron:
    def __init__(self, n):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n)]
        print(self.w)
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        output = sum((xi * wi for xi, wi in zip(x, self.w)), self.b)
        return output.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        self.b.grad = 0.0
        for i in range(len(self.w)):
            self.w[i].grad = 0.0 

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, inputs):
        output = [neuron(inputs) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def zero_grad(self):
        for i in range(len(self.neurons)):
            self.neurons[i].zero_grad()


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(ins, outs) for ins, outs in zip(sizes, sizes[1:])]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for i in range(len(self.layers)):
            self.layers[i].zero_grad()


if __name__ == "__main__":
    TRAINING_LOOP = 25

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    nn = MLP(3, [4, 4, 1])

    for i in range(TRAINING_LOOP):
        output = [nn(x) for x in xs]
        loss: Value = sum([(ypred - yout) ** 2 for ypred, yout in zip(output, ys)])
        print(f"Loss at {i}th iteration is {loss.data}")

        loss.backward()

        # Update weights
        for params in nn.parameters():
            # print(params, params.data, params.grad)
            params.data += -0.05 * params.grad

        nn.zero_grad()
