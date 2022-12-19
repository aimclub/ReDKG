from tests.utils import read_test_data

train, test, valid = read_test_data()


def test_columns():
    assert set(train.columns) == {"head", "tail", "relation", "neg_head", "neg_tail", "subsampling_weight"}

    assert set(test.columns) == {"head", "tail", "relation", "neg_head", "neg_tail"}
    assert set(valid.columns) == {"head", "tail", "relation", "neg_head", "neg_tail"}


def test_values(self):
    heads = set(self.train["head"]) | set(self.test["head"]) | set(self.valid["head"])
    tails = set(self.train["tail"]) | set(self.test["tail"]) | set(self.valid["tail"])
    entities = heads | tails

    assert set(entities) == set(range(max(entities) + 1))

    relations = set(self.train["relation"]) | set(self.test["relation"]) | set(self.valid["relation"])
    assert set(relations) == set(range(max(relations) + 1))

    assert (max(self.train["subsampling_weight"]) <= 1) == (min(self.train["subsampling_weight"]) >= 0)

    neg_heads = (
        set(sum(self.train["neg_head"], []))
        | set(sum(self.test["neg_head"], []))
        | set(sum(self.valid["neg_head"], []))
    )
    neg_tails = (
        set(sum(self.train["neg_tail"], []))
        | set(sum(self.test["neg_tail"], []))
        | set(sum(self.valid["neg_tail"], []))
    )
    assert neg_heads == heads
    assert neg_tails == tails
