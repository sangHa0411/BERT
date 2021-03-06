import random
import collections
import torch
from enum import IntEnum
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Token(IntEnum) :
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3
    CLS = 4
    SEP = 5
    MASK = 6
    IGNORE_INDEX = -100

class BertDataset(Dataset) :
  def __init__(self, masked_ids, type_ids, label_ids, order_label) :
    super(BertDataset, self).__init__()
    data_size = len(masked_ids)
    assert data_size == len(type_ids) and data_size == len(label_ids) and data_size == len(order_label)
    self.masked_ids = masked_ids
    self.type_ids = type_ids
    self.label_ids = label_ids
    self.order_label = order_label

  def __len__(self) :
    return len(self.masked_ids)

  def __getitem__(self, idx) :
    return {'input_ids' : self.masked_ids[idx],
    'type_ids' : self.type_ids[idx],
    'label_ids' : self.label_ids[idx],
    'sop' : self.order_label[idx]
    }


class BertCollator:
    def __init__(self, len_data, batch_size, size_gap = 10):
        self.len_data = len_data
        self.batch_size = batch_size
        self.size_gap = size_gap
        self.data_size = len(len_data)
        
    def sample(self) :
        batch_map = collections.defaultdict(list)
        idx_list = []
        batch_index = []
    
        for idx in range(self.data_size) :
            len_idx = self.len_data[idx]
            len_group = len_idx // self.size_gap
            batch_map[len_group].append(idx)
            
        batch_key = list(batch_map.keys())
        batch_key = sorted(batch_key, key=lambda x : x, reverse=True) 
        for key in batch_key :
            idx_list.extend(batch_map[key])
    
        for i in range(0, self.data_size, self.batch_size) :
            batch_index.append(idx_list[i:i+self.batch_size])
    
        random.shuffle(batch_index)
        return batch_index
    
    def __call__(self, batch_samples):   
        batch_input_ids = []
        batch_pos_ids = []
        batch_type_ids = []
        batch_label_ids = []
        batch_sop = []

        for data in batch_samples:
            input_ids = data['input_ids']
            type_ids = data['type_ids']
            label_ids = data['label_ids']
            sop = data['sop']

            batch_input_ids.append(torch.tensor(input_ids))
            batch_pos_ids.append(torch.arange(1,len(input_ids)+1))
            batch_type_ids.append(torch.tensor(type_ids))
            batch_label_ids.append(torch.tensor(label_ids))
            batch_sop.append(sop)

        batch_input_tensor = pad_sequence(batch_input_ids, batch_first=True, padding_value=Token.PAD)
        batch_pos_tensor = pad_sequence(batch_pos_ids, batch_first=True, padding_value=Token.PAD)
        batch_type_tensor = pad_sequence(batch_type_ids, batch_first=True, padding_value=Token.PAD)
        batch_label_tensor = pad_sequence(batch_label_ids, batch_first=True, padding_value=Token.IGNORE_INDEX)
        batch_sop_tensor = torch.tensor(batch_sop)
        
        return {'input_ids' : batch_input_tensor,
            'pos_ids' : batch_pos_tensor,
            'type_ids' : batch_type_tensor,
            'label_ids' : batch_label_tensor,
            'sop' : batch_sop_tensor
        }
