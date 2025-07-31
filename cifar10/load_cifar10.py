from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob

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

label_dict = {}
# 通过字典，将字符串全部转化为数字
for idx, name in enumerate(label_name):
    label_dict[name] = idx

# 返回通过PIL读取的图片数据
def default_loader(path):
    return Image.open(path).convert("RGB")

# 数据增强只在训练时有
# train_transform2 = transforms.Compose([
#     # 随机裁剪一些图片
#     transforms.RandomResizedCrop((28, 28)),
#     # 随机翻转一些图片
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     # 随机旋转
#     transforms.RandomRotation(90),
#     # 随机转化为灰度图
#     transforms.RandomGrayscale(0.1),
#     # 颜色的增强
#     transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#     # toTensor将PIL文件转化为网络输入的数据
#     transforms.ToTensor()
# ])

# 实际使用中，数据强化要简单一些
train_transform = transforms.Compose([
    transforms.RandomCrop(28),
    # 随机翻转一些图片
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, im_list,
                 transform=None,
                 loader=default_loader):
        # im_list:数据列表，拿到当前文件夹下，所有文件的一个列表
        # transform: 进行数据增强的函数
        # default_loader:采用pro，完成对图像数据的读取
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:
            # im_item : 带有绝对路径的图片文件名
            # E:\PyProject\PytorchLearning\cifar10\dataset\cifar-10-batches-py\TRAIN\
            # airplane\aeroplane_s_000021.png
            # 类别号，标签名
            im_label_name = im_item.split("/")[-2]
            # 列表中，每一个元素包含一个路径和lable的id
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    # 读取图片数据中元素的一种方法
    def __getitem__(self, index):
        # 定义数据的读取和数据的增强，然后返回图片的数据和label
        im_path, im_label = self.imgs[index]

        im_data = self.loader(im_path)

        # 如果数据有增强的话
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    # 返回我们样本的数量
    def __len__(self):
        return len(self.imgs)


# 首先通过global拿到训练图片的列表
im_train_list = glob.glob(r"dataset/cifar-10-batches-py/TRAIN/*/*.png")
im_test_list = glob.glob(r"dataset/cifar-10-batches-py/TEST/*/*.png")

train_dataset = MyDataset(im_train_list, transform=train_transform)
test_dataset = MyDataset(im_test_list, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset,
                               batch_size=128,
                               shuffle=True,
                               num_workers=4)
test_loader = DataLoader(dataset=test_dataset,
                               batch_size=128,
                               shuffle=False,
                               num_workers=4)

print("num_of_train", len(train_dataset))
print("num_of_test", len(test_dataset))