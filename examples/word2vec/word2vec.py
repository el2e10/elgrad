import os
import sys
from csv import reader
from typing import Dict

import numpy as np #type: ignore

sys.path.append(os.getcwd())

from elgrad import Tensor, Linear


def create_one_hot_encoding() -> Dict[str, list[int]]:
    vocab:list[str] = []
    with open("examples/word2vec/data/vocab.csv", 'r', newline='') as fp:
        rdr = reader(fp, delimiter=',')
        for word in rdr:
            vocab.append(word[0])

    result: Dict[str, list[int]] = {}
    vocab_length = len(vocab)
    for index, word in enumerate(vocab): #type: ignore
        result[word] = [1 if index == x else 0 for x in range(vocab_length)] #type: ignore

    return result 

def get_embedding(word: str) -> list[int]:
    return vocabulary[word]

def load_data():
    result = []
    with open("examples/word2vec/data/train.csv", 'r', newline='') as fp:
        rdr = reader(fp, delimiter=',')
        for row in rdr:
            embeddings = [get_embedding(word) for word in row]
            result.append(embeddings)

    array = np.array(result, dtype="int")
    return array[:,:4,:], array[:, -1, :]

vocabulary: Dict[str, list[int]] = create_one_hot_encoding()
x, y = load_data()

x = Tensor(x, label="X")
y = Tensor(y, label="y")

layer1 = Linear(140, 40, label="Layer 1")
layer2 = Linear(40, 140, label="Layer 2")

LEARNING_RATE = 0.03
EPOCHS = 25

for i in range(EPOCHS):
    for sample in x:
        x1 = layer1(sample)
        e1 = x1.sum(axis=0, keepdims=True)

        x2 = layer2(e1)
        z2 = x2.softmax(dim=1)

        print(z2)
        print("row -> ", sample.shape, "w1 ->", layer1.w.shape, "x1 -> ", x1.shape, "e1 -> ", e1.shape, "x2 -> ", x2.shape, "z2 -> ",z2.shape)
        break
    break




