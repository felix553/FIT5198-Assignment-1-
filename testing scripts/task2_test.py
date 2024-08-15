import pandas as pd
import itertools

groupnum = input("Please input your group number:")

with open("{}_vocab.txt".format(groupnum.zfill(3)), "r",encoding='utf-8') as file:
    vocab = file.readlines()
try:
    vocab = [each.strip().split(":") for each in vocab]
    assert all(len(each)==2 for each in vocab),'check your vocab file tokens and index'
    try:
        [int(each[1]) for each in vocab]
    except:
        raise ValueError("The index part of your vocab file is not numerical!")
except:
    raise ValueError("Vocab file structured incorrectly!")

print("Task 2 vocab file passed!")

groupnum="1"
with open("{}_countvec.txt".format(groupnum.zfill(3)), "r",encoding='utf-8') as file:
    countvec = file.readlines()

countvec = [each.strip().split(",") for each in countvec]

try:
    allcounts = list(itertools.chain.from_iterable([each[1:] for each in countvec]))
    ind_counts = [each.split(":") for each in allcounts]
    # testing whether the ind:count can be parsed as numerical values
    [(int(each[0]), int(each[1])) for each in ind_counts]
except:
    raise ValueError("The ind:count part of your countvec doesnt look right!")

print("Task 2 countvec file passed!")


