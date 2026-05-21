import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample, neg_sample_dns_unique






class SRDataset(Dataset):   # 为了方便计算各个长度的指标，在dataset 中加入了 original_input_ids 表示原有序列的长度

    def __init__(self, args, user_seq, test_neg_items=None):
        self.args = args
        self.user_seq = user_seq
        self.max_len = args.max_seq_length
        self.test_neg_items = test_neg_items

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        input_ids = items[:-1]
        original_input_length = len(input_ids)
        answer = [items[-1]]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = items[-1]
        target_neg = (neg_sample(answer, self.args.item_size))


        input_ids = input_ids[-self.max_len:]


        assert len(input_ids) == self.max_len



        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(original_input_length, dtype=torch.long),
        )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
    



class DNSDataset(Dataset):   # 为了方便计算各个长度的指标，在dataset 中加入了 original_input_ids 表示原有序列的长度

    def __init__(self, args, user_seq, test_neg_items=None):
        self.args = args
        self.user_seq = user_seq
        self.max_len = args.max_seq_length
        self.test_neg_items = test_neg_items
        self.N = args.N

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        input_ids = items[:-1]
        original_input_length = len(input_ids)
        answer = [items[-1]]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = items[-1]
        target_neg = neg_sample_dns_unique(answer, input_ids, self.args.item_size, self.N)


        input_ids = input_ids[-self.max_len:]


        assert len(input_ids) == self.max_len



        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(original_input_length, dtype=torch.long),
        )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)