import numpy as np #type: ignore

from .tensor import Tensor #type: ignore

random_num_generator = np.random.default_rng()

class Linear:

    def __init__(self, inputs, outputs, label=""):
        a = np.sqrt(6/(inputs + outputs))
        self.label = label
        self.w = Tensor(random_num_generator.uniform(-a, a, size=(outputs, inputs)), require_grad=True, label=f"{label}-w")
        self.b = Tensor(random_num_generator.uniform(-a, a, 1), require_grad=True, label=f"{label}-b")

    def __call__(self, input: Tensor) -> Tensor:
        input = Tensor(input) if not isinstance(input, Tensor) else input
        output = (input @ self.w.T()) + self.b
        output.label = f"{self.label} - Linear"
        return output

    def learn(self, lr):
        self.w.data = self.w.data + -lr * self.w.grad.data #type: ignore
        self.b.data = self.b.data + -lr * self.b.grad.data #type: ignore

    def zero_grad(self):
        self.w.zero_grad()
        self.b.zero_grad()
