from dataset import Token

class Encoder :
    def __init__(self, tokenizer, max_size) :
        self.tokenizer = tokenizer
        self.max_size = max_size

    def __call__(self, text_data) :
        assert isinstance(text_data, tuple)
        prev_sen = text_data[0]
        cur_sen = text_data[1]

        prev_idx = self.tokenizer.encode_as_ids(prev_sen)
        cur_idx = self.tokenizer.encode_as_ids(cur_sen)

        idx_list = [Token.CLS] + prev_idx + [Token.SEP] + cur_idx
        idx_list = idx_list[-self.max_size:]
        type_list = [1] * (len(prev_idx) + 2) + [2] * len(cur_idx) 
        type_list = type_list[-self.max_size:]

        return idx_list, type_list
