from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class KGEDataset(Dataset):

  def __init__(self,table,mode='train'):

    self.mode = mode
    self.head = torch.tensor(np.array(table['head']))
    self.tail = torch.tensor(np.array(table['tail']))
    self.relation = torch.tensor(np.array(table['relation']))
    self.neg_head = torch.tensor(np.array(list(table['neg_head'])))
    self.neg_tail = torch.tensor(np.array(list(table['neg_tail'])))

    if mode=='train':
      self.subsampling_weight = torch.tensor(np.array(table['subsampling_weight']))

  def __len__(self):
    return len(self.head)
  
  def __getitem__(self,idx):
    if self.mode == 'train':
      return self.head[idx],self.tail[idx],self.relation[idx], self.neg_head[idx],self.neg_tail[idx], self.subsampling_weight[idx]
    else:
      return self.head[idx],self.tail[idx],self.relation[idx], self.neg_head[idx],self.neg_tail[idx]


class InfoData():

  def __init__(self,train,test,valid):

    self.nentity = len(pd.concat([train['head'], valid['head'], test['head']]).unique()) + len(pd.concat([train['tail'], valid['tail'], test['tail']]).unique())
    self.nrelation = len(pd.concat([train['relation'], valid['relation'], test['relation']]).unique())
    self.volume_train = round(len(train)/(len(train)+len(test)+len(valid))*100)
    self.volume_valid = round(len(valid)/(len(train)+len(test)+len(valid))*100)
    self.volume_test = round(len(test)/(len(train)+len(test)+len(valid))*100)
    self.info = f' NUMBER OF ENTITY: {self.nentity} \n NUMBER OF RELETION: {self.nrelation} \n TRAIN: {self.volume_train}% \n VALID: {self.volume_valid}% \n TEST: {self.volume_test}%'
  
  def get_data(self):
    return self.nentity, self.nrelation, self.volume_train, self.volume_valid, self.volume_test, self.info
