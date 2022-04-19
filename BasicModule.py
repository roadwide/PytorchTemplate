'''
Author: RoadWide
Date: 2022-04-19 21:23:29
LastEditTime: 2022-04-19 21:23:29
FilePath: /PytorchTemplate/BasicModule.py
Description: 
'''
import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self):
        pass