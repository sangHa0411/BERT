import numpy as np
import copy
from dataset import Token

class Masking :
    def __init__(self, vocab_size, ignore_index=-100) :
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def __call__(self, idx_list) :
        idx_list = idx_list
        idx_size = len(idx_list)

        input_list = copy.deepcopy(idx_list)
        label_list = [self.ignore_index] * idx_size

        mask_size = int(idx_size * 0.15)
        mask_prob = np.random.rand(mask_size)

        for i in range(mask_size) :
            pos = np.random.randint(idx_size)
            prob = mask_prob[i]

            if idx_list[pos] == Token.CLS or idx_list[pos] == Token.SEP :
                continue

            if prob <= 0.8 :
                label_list[pos] = idx_list[pos]
                input_list[pos] = Token.MASK
            elif prob <= 0.9 :
                random_id = np.random.randint(self.vocab_size)
                input_list[pos] = random_id
            else :
                continue

        return input_list, label_list
