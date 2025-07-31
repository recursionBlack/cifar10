import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        # 输入图像大小为3 * 28 * 28，因为被裁剪了32=》28
        # 第1个卷积
        self.conv1 = nn.Sequential(
            # 输入channel=3，输出的channel=64，3*3的核，1的步长，1的扩展
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的channel的数量
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 第1个maxpooling
        # 每次pooling后，channel数翻倍
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 输入图像大小为14 * 14
        # 第2个卷积
        self.conv2_1 = nn.Sequential(
            # 3*3的核，1的步长，1的扩展
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的数量
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            # 3*3的核，1的步长，1的扩展
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的数量
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 第2个maxpooling
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 图片大小：7 * 7
        # 第3个卷积
        self.conv3_1 = nn.Sequential(
            # 3*3的核，1的步长，1的扩展
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的数量
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            # 3*3的核，1的步长，1的扩展
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的数量
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # 第3个maxpooling
        # 由于像素数不能整除3，所以补充padding
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2,
                                         stride=2,
                                         padding=1)

        # 图片大小：4 * 4
        # 第4个卷积
        self.conv4_1 = nn.Sequential(
            # 3*3的核，1的步长，1的扩展
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的数量
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            # 3*3的核，1的步长，1的扩展
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # 需要传入，输出的数量
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # 第4个maxpooling
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层
        # batchsize * 512 * 2 * 2 reshape==> batchsize * (512 * 4)
        self.fc = nn.Linear(512 * 4, 10)

    # 完成对输入Tensor的网络处理
    def forward(self, x):
        # 将在__init__内定义的基本算子串联起来，定义出串联网络
        batchsize = x.size(0)
        out = self.conv1(x)
        out = self.max_pooling1(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.max_pooling2(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.max_pooling3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.max_pooling4(out)

        # 全链接前先展平，展平成batchsize * n的形式
        # batchsize * c * h * w ==> batchsize * n
        out = out.view(batchsize, -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)

        return out

def VGGNet():
    return VGGBase()