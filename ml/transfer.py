import torch.nn as nn
import torchvision.models as models

def last_container(module):
    last = module
    children = list(last.children())
    l = []
    while len(children) > 0:
        l.append(last) 
        last = children[-1]
        children = list(last.children())
    return l[-1]

class transfer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def set_last_linear(self, out_features):
        container = last_container(self.model)
        name, last = container._modules.popitem()
        container.add_module(name, nn.Linear(last.in_features, out_features))

    def forward(self, x):
        return self.model( x )

    def freeze(self):
        for c in list(self.model.children())[:-1]:
            for p in c.parameters():
                p.requires_grad=False

    def unfreeze(self):
        for c in list(self.model.children())[:-1]:
            for p in c.parameters():
                p.requires_grad=True

    @classmethod
    def resnet34(cls, out, pretrained=False):
        m = cls(models.resnet34(pretrained=pretrained))
        m.set_last_linear( out )
        return m

