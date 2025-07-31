import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from resnet import resnet
from inceptionModule import InceptionNetSmall
from mobilenetv1 import mobilenetv1_small
from pre_resnet import pytorch_resnet18
from load_cifar10 import train_loader, test_loader
import os
import tensorboardX


# 有gpu就用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_num = 200
lr = 0.01
batch_size = 128

# 以下网络任选其一
# net = VGGNet().to(device)
net = resnet().to(device)
# net = mobilenetv1_small().to(device)
# net = InceptionNetSmall().to(device)
# net = pytorch_resnet18().to(device)

# loss
# 分类问题用交叉熵
loss_func = nn.CrossEntropyLoss()

# optimizer
# Adam
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# SGD
# optimizer = torch.optim.SGD(net.parameters(), lr=lr,
#                             momentum=0.9, weight_decay=5e-4)

# 学习率自动衰减，防止训练后期的学习率过大
# 每5个epoch后衰减一次，变成上一次的0.9倍
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

if not os.path.exists("log"):
    os.mkdir("log")
writer = tensorboardX.SummaryWriter("log")

step_n = 0
# 定义训练过程
for epoch in range(epoch_num):
    net.train()     # train BN dropout

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, dim=1)

        correct = pred.eq(labels.data).cpu().sum()
        # 测试时把这些log先给注释掉
        # print("epoach is ", epoch)
        # # 打印学习率
        # print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])
        # print("train step", i, "loss is:", loss.item(),
        #       "mini-batch correct is:", 100.0 * correct / batch_size)

        # 写的日志可以通过tensorboardX查看，但会降低速度
        # writer.add_scalar("train loss", loss.item(), global_step=step_n)
        # writer.add_scalar("train correct", 100.0 * correct.item() / batch_size, global_step=step_n)

        step_n += 1
    # 每次跑完一个epoch后，存放一下模型
    if not os.path.exists("models"):
        os.mkdir("models")
    # 训练集为50000，batch_size为128，所以每个epoch有390step
    torch.save(net.state_dict(), "models/{}.path".format(epoch + 1))
    # 学习率自动衰减
    scheduler.step()

    sum_loss = 0
    sum_correct = 0
    # 每训练一个epoch之后呢，就对网络尽心一次测试
    # 采用和train相同的方式，不过这里不再使用反向传播,只进行前向运算，
    # 并且拿到loss和准确率
    # 从test_loader里加载数据
    for i, data in enumerate(test_loader):
        net.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        # test里不需要反向传播
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).cpu().sum()

        sum_loss += loss.item()
        sum_correct += correct.item()

        writer.add_scalar("test loss", loss.item(), global_step=step_n)
        writer.add_scalar("train correct", 100.0 * correct.item() / batch_size, global_step=step_n)

    # 计算每个batch平均准确率
    test_loss = sum_loss * 1.0 / len(test_loader)
    test_correct = sum_correct * 100.0 / len(test_loader) / batch_size
    print("epoch is ", epoch + 1, "loss is:", test_loss,
          "test correct is:", test_correct)

writer.close()
