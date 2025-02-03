import os
import sys

sys.path.append(os.getcwd())

from elgrad import Tensor, Linear

TRAINING_LOOP = 25

y = Tensor([1.0, -1.0, -1.0, 1.0], require_grad=False, label="y")
x1 = Tensor([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
], require_grad=False, label="x1")

l1 = Linear(3, 4)
l2 = Linear(4, 4)
l3 = Linear(4, 1)

for i in range(TRAINING_LOOP):
    LEARNING_RATE = 0.01

    x2 = l1(x1)
    x2.label = "x2"
    
    x3 = l2(x2)
    x3.label = "x3"

    x4 = l3(x3)
    
    loss = ((x4 - y.reshape((4, 1))) ** 2).sum()
    loss.label = "loss"

    l1.zero_grad()
    l2.zero_grad()
    l3.zero_grad()

    loss.backward()

    print(f"Loss at {i} is {loss.data}")

    l1.learn(LEARNING_RATE)
    l2.learn(LEARNING_RATE)
    l3.learn(LEARNING_RATE)
