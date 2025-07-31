import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    def conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel,
                      kernel_size=3, stride=stride, padding=1,
                      groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),

            # 定义点卷积，核只有1 x 1
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=1, stride=1, padding=0,
                      groups=in_channel, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def __init__(self):
        super(MobileNet, self).__init__()
        # 搭建神经网络时，第一层往往都是标准的卷积层
        # 第1个卷积
        self.conv1 = nn.Sequential(
            # 输入channel=3，输出的channel=32，3*3的核，1的步长，1的扩展
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的channel的数量
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # mobilenet的核心单元：深组可分离卷积
        self.conv_dw2 = self.conv_dw(32, 32, 1)
        self.conv_dw3 = self.conv_dw(32, 64, 2)

        self.conv_dw4 = self.conv_dw(64, 64, 1)
        self.conv_dw5 = self.conv_dw(64, 128, 2)

        self.conv_dw6 = self.conv_dw(128, 128, 1)
        self.conv_dw7 = self.conv_dw(128, 256, 2)

        self.conv_dw8 = self.conv_dw(256, 256, 1)
        self.conv_dw9 = self.conv_dw(256, 512, 2)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)

        out = F.avg_pool2d(out, 2)
        out = out.view(-1, 512)
        out = self.fc(out)

        return out

def mobilenetv1_small():
    return MobileNet()