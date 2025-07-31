import torch.nn as nn
from torchvision import models

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        # 使用torchvision中已有的经典的网络结构
        self.model = models.resnet18(pretrained=True)
        # 获取fc层输入的维度
        self.num_features = self.model.fc.in_features
        # 定义为10分类
        self.model.fc = nn.Linear(self.num_features, 10)

    def forward(self, x):
        out = self.model(x)
        return out

def pytorch_resnet18():
    return resnet18()