from torch.utils.data import Dataset
import torch
import numpy as np

def encode(sequence, dict_size, seq_len):
    # Create an array of zeros with shape (seq_len, dict_size)
    features = np.zeros((seq_len, dict_size), dtype=np.float32)
        
    # numpy.arange creates an array [0, 1, ..., seq_len-1]
    # sequence is used to index dict_size in each row
    features[np.arange(seq_len), sequence] = 1
    return features

class CountingDataset(Dataset):
    def __init__(self, count_len):
        self.count_len = count_len

    def __len__(self):
        return self.count_len - 3

    def __getitem__(self, index):
        seq = [[index, index + 1, index + 2]]
        label = [[index + 3]]

        return {
            'raw inputs': torch.Tensor(seq),
            'raw labels': torch.Tensor(label),
        }

class CountingDatasetEmbeddings(Dataset):
    def __init__(self, count_len):
        self.count_len = count_len

    def __len__(self):
        return self.count_len - 3

    def __getitem__(self, index):
        seq = [index, index + 1, index + 2]
        label = [index + 1, index + 2, index + 3]

        encoded_input = encode(seq, self.count_len, len(seq))
        encoded_output = encode(label, self.count_len, len(label))

        return {
            'raw inputs': torch.Tensor(seq),
            'raw labels': torch.Tensor(label),
            'encoded inputs': torch.from_numpy(encoded_input),
            'encoded labels': torch.from_numpy(encoded_output)
        }
