import unittest
import pandas as pd

class TestData(unittest.TestCase):

    def setUp(self):
        self.train = pd.read_csv(r'data/train.csv')
        self.test = pd.read_csv(r'data/test.csv')
        self.valid = pd.read_csv(r'data/valid.csv')

        for i in [self.train, self.test, self.valid]:
             i['neg_head'] = [eval(l) for l in i['neg_head']]
             i['neg_tail'] = [eval(l) for l in i['neg_tail']]

    def test_columns(self):
        self.assertEqual((set(self.train.columns)), (set(['head', 'tail', 'relation', 'neg_head', 'neg_tail', 'subsampling_weight'])), 
                            "Column names for train dataset do not match requirements ['head', 'tail', 'relation', 'neg_head', 'neg_tail', 'subsampling_weight']")
        self.assertEqual((set(self.test.columns)), (set(['head', 'tail', 'relation', 'neg_head', 'neg_tail'])),
                            "Column names for test dataset do not match requirements ['head', 'tail', 'relation', 'neg_head', 'neg_tail', 'subsampling_weight']")
        self.assertEqual((set(self.valid.columns)), (set(['head', 'tail', 'relation', 'neg_head', 'neg_tail'])),
                            "Column names for valid dataset do not match requirements ['head', 'tail', 'relation', 'neg_head', 'neg_tail', 'subsampling_weight']")
    
    def test_values(self):
        heads = set (self.train['head']) | set(self.test['head']) | set(self.valid['head'])
        tails = set (self.train['tail']) | set(self.test['tail']) | set(self.valid['tail'])
        entities = heads | tails

        self.assertEqual((set(entities)), (set(range(max(entities)+1))), 
                            "Entities are not reindexed from 0 to len(entities)")

        relations = set(self.train['relation']) | set(self.test['relation']) | set(self.valid['relation'])
        self.assertEqual((set(relations)), (set(range(max(relations)+1))), 
                            "Relations are not reindexed from 0 to len(relations)")

        self.assertEqual((max(self.train['subsampling_weight']) <= 1), (min(self.train['subsampling_weight']) >= 0), 
                            "The variable subsampling_weight is not in the range [0,1]")
        
        neg_heads = set(sum(self.train['neg_head'], [])) | set(sum(self.test['neg_head'], [])) | set(sum(self.valid['neg_head'], []))
        neg_tails = set(sum(self.train['neg_tail'], [])) | set(sum(self.test['neg_tail'], [])) | set(sum(self.valid['neg_tail'], []))
        self.assertEqual((neg_heads), (heads), 
                            "There is an index in negative head samples that is not in heads")
        self.assertEqual((neg_tails), (tails), 
                            "There is an index in negative tail samples that is not in tails")

if __name__ == '__main__':
    unittest.main()
