import torch
import glob
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from resnet import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = resnet()

# 加载已经训练好的模型文件
net.load_state_dict(torch.load("finishmodels/2.path"))

im_list = glob.glob("dataset/cifar-10-batches-py/TEST/*/*")

np.random.shuffle(im_list)

net.to(device)

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

# 数据增强
test_transform = transforms.Compose([
    transforms.CenterCrop((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# 遍历图片数据
for im_path in im_list:
    net.eval()
    im_data = Image.open(im_path)

    inputs = test_transform(im_data)
    # 放到GPU上准备计算
    inputs = inputs.to(device)
    # 维度的扩充，增加batch
    # 扩充后的维度：batch_size * channel * h * w
    inputs = torch.unsqueeze(inputs, dim=0)
    outputs = net.forward(inputs)

    _, pred = torch.max(outputs.data, dim=1)

    print(label_name[pred.cpu().numpy()[0]])

    img = np.asarray(im_data)
    # opencv是bgr格式，需要进行格式转换
    img = img[:, :, [1, 2, 0]]

    img = cv2.resize(img, (300, 300))
    cv2.imshow("im", img)
    cv2.waitKey(0)