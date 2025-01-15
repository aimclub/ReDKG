import pathlib

import pandas as pd


def read_test_data():
    path = pathlib.Path(__file__).parent.resolve().joinpath("data")
    train = pd.read_csv(path.joinpath("train.csv"))
    test = pd.read_csv(path.joinpath("test.csv"))
    valid = pd.read_csv(path.joinpath("valid.csv"))

    for i in [train, test, valid]:
        i["neg_head"] = [eval(l_) for l_ in i["neg_head"]]
        i["neg_tail"] = [eval(l_) for l_ in i["neg_tail"]]
    return train, test, valid
