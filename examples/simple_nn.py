import os
import sys

sys.path.append(os.getcwd())

import numpy as np  # type: ignore

from elgrad import Tensor, Linear

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

w2 = Tensor(random_num_generator.uniform(-1, 1, size = (4, 4)), require_grad=True, label="w2")
b2 = Tensor(random_num_generator.uniform(-1, 1, size = 1), require_grad=True, label="b2")


w3 = Tensor(random_num_generator.uniform(-1, 1, size=(1, 4)), require_grad=True, label="w3")
b3 = Tensor(random_num_generator.uniform(-1, 1, size=1), require_grad=True, label="b3")

for i in range(TRAINING_LOOP):
    LEARNING_RATE = 0.01

    x2 = x1 @ w1.T() + b1
    x2.label = "x2"

    x3 = x2 @ w2.T() + b2
    x3.label = "x3"

    x4 = x3 @ w3.T() + b3

    loss = ((x4 - y.reshape((4, 1))) ** 2).sum()
    loss.label = "loss"

    w1.zero_grad()
    w2.zero_grad()
    w3.zero_grad()
    b1.zero_grad()
    b2.zero_grad()
    b3.zero_grad()

    loss.backward()
    print(f"Loss at {i} is {w1} {loss.data}")

    w1 = w1 - LEARNING_RATE * w1.grad #type: ignore
    b1 = b1 - LEARNING_RATE * b1.grad #type: ignore
    w2 = w2 - LEARNING_RATE * w2.grad #type: ignore
    b2 = b2 - LEARNING_RATE * b2.grad #type: ignore
    w3 = w3 - LEARNING_RATE * w3.grad #type: ignore
    b3 = b3 - LEARNING_RATE * b3.grad #type: ignore

    




