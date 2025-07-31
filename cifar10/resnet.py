import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        # 定义主干分支
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=3, stride=stride, padding=1),
            # 需要传入，输出的channel的数量
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        # 定义跳连分支
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)
        return  out

class ResNet(nn.Module):
    # 由于Resnet采用了跳连结构，导致其不会出现vggnet的梯度消失问题
    # 所以可以有很深的层数，比如50层，而vggnet至多只有10层
    # 所以需要定义一个循环，批量定义每一层
    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(
                block(self.in_channel,
                      out_channel,
                      in_stride))
            self.in_channel = out_channel

        return nn.Sequential(*layers_list)

    def __init__(self, ResBlock):
        super(ResNet, self).__init__()
        self.in_channel = 32
        # 第一个卷积通常不会很复杂
        self.conv1 = nn.Sequential(
            # 输入channel=3，输出的channel=64，3*3的核，1的步长，1的扩展
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的channel的数量
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = self.make_layer(ResBlock, 64, 2, 2)
        self.layer2 = self.make_layer(ResBlock, 128, 2, 2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, 2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, 2)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def resnet():
    return ResNet(ResBlock)