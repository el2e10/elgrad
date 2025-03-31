import os
import sys
from csv import reader
from typing import Dict

import numpy as np #type: ignore
from numpy.linalg import norm #type: ignore

sys.path.append(os.getcwd())
# sys.path.append("/Users/eldhoittangeorge/Personal/ML/Projects/elgrad")

from elgrad import Tensor, Linear

vocab: list[str] = []
def create_one_hot_encoding() -> Dict[str, list[int]]:
    global vocab
    with open("examples/word2vec/data/vocab_4.csv", 'r', newline='') as fp:
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
    return one_hot_embedding[word]


def load_data():
    result = []
    with open("examples/word2vec/data/train_4.csv", 'r', newline='') as fp:
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

def train(x, y):

    layer1 = Linear(92, 40, label="Layer 1")
    layer2 = Linear(40, 92, label="Layer 2")

    LEARNING_RATE = 0.5
    EPOCHS = 1000 

    # def get_prediction(output: Tensor):
    #     index = np.argmax(output.data, axis=1)
    #     print(vocab[index[0]])

    for i in range(EPOCHS):
        x1 = layer1(x)
        e1 = x1.mean(axis=1, keepdims=False)

        x2 = layer2(e1)
        z2 = x2.softmax(dim=1)
        # get_prediction(z2)

        loss = -((y * z2.log()).sum()/3495.0)
        loss.label = "Loss"

        if(i % 10 == 0):
            print(f"Loss at {i}th iteration is {loss.data}")

        layer1.zero_grad()
        layer2.zero_grad()

        loss.backward()
        
        layer1.learn(LEARNING_RATE)
        layer2.learn(LEARNING_RATE)

    with open("examples/word2vec/embedding.npy", 'wb') as fp:
        np.save(fp, layer1.w.data.T)

def load_embedding():
    file_name = "examples/word2vec/embedding.npy"
    embedding = np.load(file_name)
    return embedding

def load_vocbulary():
    data = []
    with open("examples/word2vec/data/vocab_4.csv", 'r') as fp:
        rdr = reader(fp, delimiter=',')
        for row in rdr:
            data.append(row[0])
    return data

def get_similarity(word_1, word_2):
    global data
    global vocabulary

    index_1 = vocabulary.index(word_1)
    index_2 = vocabulary.index(word_2)

    cos_sim = (data[index_1] @ data[index_2].T) / (norm(data[index_1] * norm(data[index_2])))
    print(f"Cosine similarity between {vocabulary[index_1]} and {vocabulary[index_2]} is {cos_sim}")

if __name__ == '__main__':
    one_hot_embedding: Dict[str, list[int]] = create_one_hot_encoding()

    x, y = load_data()
    x = Tensor(x, label="X")
    y = Tensor(y, label="y")

    train(x, y)
    
    vocabulary = load_vocbulary()
    data = load_embedding()







































