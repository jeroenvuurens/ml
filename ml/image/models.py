from torchvision.models import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def single_layer_perceptron(input, output):

    class SingleLayerPerceptron(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.w1 = nn.Linear(input, output)

        def convert_input(self, x):
            return x[:,0,:,:].view(x.shape[0], -1)

        def forward(self, x):
            x = self.convert_input(x)
            x = self.w1(x)
            x = torch.log_softmax(x, dim=1)
            return x

    return SingleLayerPerceptron
