from torch.utils import data
import torch
import numpy as np

class DataSet(data.Dataset):

    def __init__(self, dataset_path, train = True):
        npArray = np.load(dataset_path)
        self.label, self.data = 0, 0
        if (train):
            self.label = npArray['trainLabel']
            self.data = npArray['trainData']
        else:
            self.label = npArray['testLabel']
            self.data = npArray['testData']

    def __getitem__(self, idx):
        label, data = self.label[idx], self.data[idx]
        label, data = torch.tensor(label), torch.tensor(data, dtype=torch.float)
        data = data.reshape(1, 28, 28)
        return data, label

    def __len__(self):
        return len(self.label)