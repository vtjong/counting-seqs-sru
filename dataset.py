from torch.utils.data import Dataset
import torch

class CountingDataset(Dataset):
    def __init__(self, count_len):
        self.count_len = count_len

    def __len__(self):
        return self.count_len - 3

    def __getitem__(self, index):
        seq = [[index, index + 1, index + 2]]
        label = [[index + 3]]

        return {
            'data': torch.Tensor(seq),
            'label': torch.Tensor(label),
        }