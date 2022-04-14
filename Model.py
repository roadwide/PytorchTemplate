import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(28*28, 500), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(500, 128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, 10), nn.LogSoftmax(dim = 1))
        
    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out