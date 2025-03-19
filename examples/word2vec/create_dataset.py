from csv import writer
from typing import List 
from collections import Counter
from itertools import chain 

MAIN_FILE_PATH = "data"

def load_dataset() -> List[str]:
    result: List[str] =[]
    with open(f"{MAIN_FILE_PATH}/data_dump.txt", "r") as fp:
        result = fp.readlines()

    result = [line.strip() for line in result]
    return result

def create_cbow_dataset(sentence: str) -> List[List[str]]:
    result: List[List[str]] = []
    sentence_arr = sentence.strip().split(' ')
    length = 1
    if(len(sentence_arr) < (2 * length + 1)):
        return result

    for index in range(length, len(sentence_arr) - length):
        result.append(sentence_arr[index-2: index] + sentence_arr[index+1: index + 3] + [sentence_arr[index]])

    for index in range(0, len(sentence_arr) - (length*2)):
        result.append(sentence_arr[index: index + 5])
    return result


def create_vocabulary(data_dump: List[str])-> list[tuple[str, int]]:
    dataset_str = list(chain.from_iterable([x.split(' ') for x in data_dump]))
    c = Counter(dataset_str)  
    words = sorted(c.items(), key=lambda item: item[1], reverse=True)
    with open("data/vocab.csv", "w", newline='') as fp:
        data_writer = writer(fp, delimiter=',') 
        data_writer.writerows(words)

    return words
    

if __name__ == '__main__':
    data_dump = load_dataset()
    vocabulary = create_vocabulary(data_dump)
    print(vocabulary, len(vocabulary))
    dataset: List[List[str]] = []
    for sentence in data_dump:
        dataset += create_cbow_dataset(sentence)

    with open("data/train.csv", 'w', newline='') as fp:
        data_writer = writer(fp, delimiter=',')
        data_writer.writerows(dataset)
