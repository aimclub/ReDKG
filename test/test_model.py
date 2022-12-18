import unittest
import pandas as pd
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import sys
sys.path.append('../')

from redkg.kge import Evaluator
from kge_dataset import InfoData, KGEDataset

class TestModel(unittest.TestCase):

    def setUp(self):
        self.train = pd.read_csv(r'data/train.csv') 
        self.test = pd.read_csv(r'data/test.csv')
        self.valid = pd.read_csv(r'data/valid.csv')

        for i in [self.train, self.test, self.valid]:
             i['neg_head'] = [eval(l) for l in i['neg_head']]
             i['neg_tail'] = [eval(l) for l in i['neg_tail']]

    def test_data_info(self):
        nentity,nrelation, v_tr, v_val, v_test, info = InfoData(self.train,self.test,self.valid).get_data()
        self.assertEqual( (nentity), (20), "Incorrect count of the number of entities")
        self.assertEqual( (nrelation), (2), "Incorrect count of the number of relations")
        self.assertEqual( (v_tr), (71), "Incorrect count of the volume of train")
        self.assertEqual( (v_test), (14), "Incorrect count of the volume of test")
        self.assertEqual( (v_val), (14), "Incorrect count of the volume of valid")

    def test_dataset(self):
        train_set = KGEDataset(self.train)
        test_set = KGEDataset(self.test, mode='test')
        valid_set = KGEDataset(self.valid, mode='test')

        self.assertEqual( (len(train_set)), (100), "Incorrect length of loaded train dataset")
        self.assertEqual( (len(test_set)), (20), "Incorrect length of loaded test dataset")
        self.assertEqual( (len(valid_set)), (20), "Incorrect length of loaded valid dataset")

        train_set_first = ( torch.cat( [i.view(1) for i in train_set[0][0:3]+(train_set[0][5],)] + list(train_set[0][3:5]) ) ).tolist()
        test_set_first = ( torch.cat( [i.view(1) for i in test_set[0][0:3]] + list(test_set[0][3:]) ) ).tolist()
        valid_set_first = ( torch.cat( [i.view(1) for i in valid_set[0][0:3]] + list(valid_set[0][3:]) ) ).tolist()
        self.assertEqual( (round(sum(train_set_first))), (186), "Incorrect reading of train elements")
        self.assertEqual( (round(sum(test_set_first))), (210), "Incorrect reading of test elements")
        self.assertEqual( (round(sum(valid_set_first))), (221), "Incorrect reading of valid elements")

    def test_evaluator(self):
        evaluator = Evaluator() 
        in_dict = {'y_pred_pos': torch.tensor([0.1]), 'y_pred_neg': torch.tensor([[0.04,0.11,0.05,0.03,0.15,0.07]])}
        metrics = evaluator.eval(in_dict)
        self.assertEqual( (round(sum(torch.cat(metrics).tolist()),3)), (2.333), "Metrics calculated incorrectly")

    def test_model(self):
        evaluator = Evaluator() 
        kge_model = KGEModel(model_name = 'TransE', nentity=20, nrelation=2,
                    embedding_size=2, gamma=12, evaluator=evaluator)
        self.assertEqual( (list(kge_model.entity_embedding.shape)), ([20,2]), "Incorrect entity embedding size")
        self.assertEqual( (list(kge_model.relation_embedding.shape)), ([2,2]), "Incorrect relation embedding size")
        positive_score, negative_tail_score, negative_head_score = kge_model(torch.tensor(1), torch.tensor(11), torch.tensor(0), torch.tensor([[0,2,3]]), torch.tensor([[12,13,14]]))
        self.assertEqual( (round(sum([x for l in positive_score.tolist() + negative_tail_score.tolist() + negative_head_score.tolist() for x in l]),3)), (10.778), "Model not working correctly")

if __name__ == '__main__':
    unittest.main()
