import sys
from csv import DictReader
from typing import List
sys.path.append("/Users/eldhoittangeorge/Business/ML/Projects/elgrad/")

import numpy as np #type: ignore

from elgrad import Tensor
from elgrad import Linear


EPOCHS = 25
data = []
with open("/Users/eldhoittangeorge/Business/ML/Projects/elgrad/examples/iris/data.csv", 'r') as fp:
    csv_fp = DictReader(fp)
    for row in csv_fp:
        data.append(row)

def convert_label_to_one_hot_encoding(dataset: List[dict], label_colname:str):
    unique_labels: list[str] = [] 
    result: list[list[int]] = []
    for row in dataset:
        if(row[label_colname] not in unique_labels):
            unique_labels.append(row[label_colname])

    length = len(unique_labels)
    for row in dataset:
        position = unique_labels.index(row[label_colname])
        result.append([0 if position == x else 1 for x in range(length)])

    return result

def extract_inputs(dataset: List[dict]):
    result: List[list[float]] = []
    for row in dataset:
        result.append([row["SepalLengthCm"], row["SepalWidthCm"], row["PetalLengthCm"], row["PetalWidthCm"]])

    return result

def normalize_inputs(t: List[List]):
    t = np.array(t, dtype="float64")
    mean = np.mean(t, axis=0)
    std = np.std(t, axis=0)
    result = (t - mean)/(std + 1e-8)
    return result


label = convert_label_to_one_hot_encoding(data, "Species") 
y = Tensor(label, label="Y")
normalized_data = normalize_inputs(extract_inputs(data))
x = Tensor(normalized_data, require_grad=True)


layer1 = Linear(4, 10, label="Layer 1")
layer2 = Linear(10, 20, label="Layer 2")
layer3 = Linear(20, 3, label="Layer 3")


LEARNING_RATE = 0.1
for i in range(EPOCHS):
    x1 = layer1(x)
    z1 = x1.relu()

    x2 = layer2(z1)
    z2 = x2.relu()

    x3 = layer3(z2)
    z3 = x3.softmax()

    # z4.require_grad = True

    loss = (y * x3.log()).sum()
    print(f"Loss at {i}th step is {loss}")

    layer1.zero_grad()
    layer2.zero_grad()
    layer3.zero_grad()

    loss.backward()

    layer1.learn(LEARNING_RATE)
    layer2.learn(LEARNING_RATE)
    layer3.learn(LEARNING_RATE)


