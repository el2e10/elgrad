import os
import sys
from csv import reader
from typing import Dict

import numpy as np #type: ignore

sys.path.append(os.getcwd())
# sys.path.append("/Users/eldhoittangeorge/Personal/ML/Projects/elgrad")

from elgrad import Tensor, Linear

vocab: list[str] = []
def create_one_hot_encoding() -> Dict[str, list[int]]:
    global vocab
    with open("examples/word2vec/data/vocab_2.csv", 'r', newline='') as fp:
    # with open("data/vocab_2.csv", 'r', newline='') as fp:
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
    with open("examples/word2vec/data/train_2.csv", 'r', newline='') as fp:
        rdr = reader(fp, delimiter=',')
        for index, row in enumerate(rdr):
            if(len(row) != 5):
                print(index)
                break
            embeddings = [get_embedding(word) for word in row]
            result.append(embeddings)

    array = np.array(result, dtype="int")
    print(array.shape)
    return array[:,:4,:], array[:, -1, :]

vocabulary: Dict[str, list[int]] = create_one_hot_encoding()
x, y = load_data()

x = Tensor(x, label="X")
y = Tensor(y, label="y")

layer1 = Linear(111, 40, label="Layer 1")
layer2 = Linear(40, 111, label="Layer 2")

LEARNING_RATE = 0.5
EPOCHS = 1 

def get_prediction(output: Tensor):
    index = np.argmax(output.data, axis=1)
    print(vocab[index[0]])

for i in range(EPOCHS):
    x1 = layer1(x)
    e1 = x1.mean(axis=1, keepdims=False)

    x2 = layer2(e1)
    z2 = x2.softmax(dim=1)
    # get_prediction(z2)

    loss = -((y * z2.log()).sum()/328.0)
    loss.label = "Loss"
    print(f"Loss at {i}th iteration is {loss.data}")
    print(f"x1 -> {x1.shape} e1 -> {e1.shape} x2 -> {x2.shape} z2 -> {z2.shape}")

    layer1.zero_grad()
    layer2.zero_grad()

    loss.backward()
    
    layer1.learn(LEARNING_RATE)
    layer2.learn(LEARNING_RATE)

print(layer1.w)


