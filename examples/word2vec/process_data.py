import re
from typing import List

STOP_WORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

def remove_stopwords(sentence: str) -> str:
    sentence_arr: List[str] = sentence.split(' ')
    # return ' '.join(sentence_arr)
    result_arr: List[str] = []
    for word in sentence_arr:
        if(word in STOP_WORDS):
            continue
        result_arr.append(word)

    return ' '.join(result_arr)



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
    FILE_PATH = "examples/word2vec/data/data4.txt"
    with open(FILE_PATH, "r") as fp:
        text = fp.readlines()

    result: List[str] = []
    for line in text:
        line = split_data(line).strip()
        result.append(line) if line else None

    clean_data = remove_non_word(result)

    data_without_stopwords: List[str] = []
    for line in clean_data:
        clean_line = remove_stopwords(line)
        if(len(clean_line) >= 5):
            data_without_stopwords.append(clean_line)

     
    with open("examples/word2vec/data/data_dump_4.txt", 'w') as fp:
        for line in data_without_stopwords:
            print(line, file=fp)
