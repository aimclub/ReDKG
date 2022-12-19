from tests.utils import read_test_data

train, test, valid = read_test_data()


def test_columns():
    assert set(train.columns) == {"head", "tail", "relation", "neg_head", "neg_tail", "subsampling_weight"}

    assert set(test.columns) == {"head", "tail", "relation", "neg_head", "neg_tail"}
    assert set(valid.columns) == {"head", "tail", "relation", "neg_head", "neg_tail"}


def test_values():
    heads = set(train["head"]) | set(test["head"]) | set(valid["head"])
    tails = set(train["tail"]) | set(test["tail"]) | set(valid["tail"])
    entities = heads | tails

    assert set(entities) == set(range(max(entities) + 1))

    relations = set(train["relation"]) | set(test["relation"]) | set(valid["relation"])
    assert set(relations) == set(range(max(relations) + 1))

    assert (max(train["subsampling_weight"]) <= 1) == (min(train["subsampling_weight"]) >= 0)

    neg_heads = (
        set(sum(train["neg_head"], []))
        | set(sum(test["neg_head"], []))
        | set(sum(valid["neg_head"], []))
    )
    neg_tails = (
        set(sum(train["neg_tail"], []))
        | set(sum(test["neg_tail"], []))
        | set(sum(valid["neg_tail"], []))
    )
    assert neg_heads == heads
    assert neg_tails == tails
