import os
import sys

sys.path.append(os.getcwd())

import numpy as np  # type: ignore

from elgrad import Tensor

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]
y = Tensor(ys, require_grad=False, label="y")

TRAINING_LOOP = 25

random_num_generator = np.random.default_rng()

x1 = Tensor(xs, require_grad=False, label="x1")
w1 = Tensor(
    random_num_generator.uniform(-1, 1, size=(4, 3)), require_grad=True, label="w1"
)
b1 = Tensor(random_num_generator.uniform(-1, 1, 1), require_grad=True, label="b1")

x2 = x1 @ w1.T() + b1
# print(x1.shape, w1.shape, x2.shape)
x2.label = "x2"
w2 = Tensor(random_num_generator.uniform(-1, 1, size = (4, 4)), require_grad=True, label="w2")
b2 = Tensor(random_num_generator.uniform(-1, 1, size = 1), require_grad=True, label="b2")

x3 = x2 @ w2.T() + b2
x3.label = "x3"
w3 = Tensor(random_num_generator.uniform(-1, 1, size=(1, 4)), require_grad=True, label="w3")
b3 = Tensor(random_num_generator.uniform(-1, 1, size=1), require_grad=True, label="b3")

x4 = x3 @ w3.T() + b3
# print(x4.shape) 
loss = ((x4 - y.reshape((4, 1))) ** 2).sum()
loss.label = "loss"
print(w1.grad, b1.grad, w2.grad, b2.grad, w3.grad, b3.grad)
loss.backward()
print(w1.grad, b1.grad, w2.grad, b2.grad, w3.grad, b3.grad)

print(f"The output is {x4}\n {loss}")
