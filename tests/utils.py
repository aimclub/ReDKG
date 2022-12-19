import pandas as pd


def read_test_data():
    train = pd.read_csv(r"data/train.csv")
    test = pd.read_csv(r"data/test.csv")
    valid = pd.read_csv(r"data/valid.csv")

    for i in [train, test, valid]:
        i["neg_head"] = [eval(l) for l in i["neg_head"]]
        i["neg_tail"] = [eval(l) for l in i["neg_tail"]]
    return train, test, valid
