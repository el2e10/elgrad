import re
from typing import List

def split_data(data: str) -> str:
    pattern = re.compile("^[A-Za-z]+ ?[A-Za-z]+:$")
    data = pattern.sub('',data)
    return data
    
def remove_non_word(data: List[str]) -> List[str]:
    result: List[str] = []

    for d in data:
        d = d.lower()
        d = re.sub(r"[^ \w\s]", " ", d)
        d = re.sub(r"\d", "", d)
        d = re.sub(" +", " ", d)
        result.append(d)

    return result


if __name__ == "__main__":
    FILE_PATH = "data/shakespeare.txt"
    with open(FILE_PATH, "r") as fp:
        text = fp.readlines()

    result: List[str] = []
    for line in text:
        line = split_data(line).strip()
        result.append(line) if line else None

    clean_data = remove_non_word(result)
    print(clean_data[:10])
     
    with open("data/data_dump.txt", 'w') as fp:
        for line in clean_data:
            print(line, file=fp)
