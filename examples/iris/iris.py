import sys
from csv import DictReader
from typing import List
sys.path.append("../../")

from elgrad import Tensor


data = []
with open("data.csv", 'r') as fp:
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

label = convert_label_to_one_hot_encoding(data, "Species")
y = Tensor(label, label="Y")
x = Tensor(extract_inputs(data))


print(x)
