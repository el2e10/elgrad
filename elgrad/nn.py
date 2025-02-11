import numpy as np #type: ignore

from .tensor import Tensor #type: ignore

random_num_generator = np.random.default_rng()

class Linear:

    def __init__(self, inputs, outputs):
        a = np.sqrt(6/(inputs + outputs))
        self.w = Tensor(random_num_generator.uniform(-a, a, size=(outputs, inputs)), require_grad=True)
        self.b = Tensor(random_num_generator.uniform(-1, 1, 1), require_grad=True)

    def __call__(self, input: Tensor) -> Tensor:
        input = Tensor(input) if not isinstance(input, Tensor) else input
        return (input @ self.w.T()) + self.b

    def learn(self, lr):
        self.w += -(lr) * self.w.grad
        self.b += -(lr) * self.b.grad

    def zero_grad(self):
        self.w.zero_grad()
        self.b.zero_grad()
