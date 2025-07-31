
# 从官网上粘贴过来的读取脚本
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
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

import glob
import numpy as np
import cv2
import os

train_list = glob.glob("dataset/cifar-10-batches-py/test_batch")
print(train_list)
save_path = "dataset/cifar-10-batches-py/TEST"

for l in train_list:
    # print(l)
    l_dict = unpickle(l)
    # print(l_dict)
    # print(l_dict.keys())

    for im_idx, im_data in enumerate(l_dict[b'data']):
        # print(im_idx)
        # print(im_data)

        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]

        # print(im_label, im_name, im_data)

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        # 数据通道的转换，从 [3, 32, 32]变成[32, 32，3]
        im_data = np.transpose(im_data, (1, 2, 0))

        # 32*32太小了，将其变为200*200的尺寸
        # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
        # 防止图片被快速刷新掉，有了这个语句，按空格才会切换图片
        # cv2.waitKey(0)

        # 如果不存在这个路径，就先创建出路径来
        if not os.path.exists("{}/{}".format(save_path,
                                             im_label_name)):
            # 根据每个标签，创建出对应的文件夹，共10个
            os.mkdir("{}/{}".format(save_path, im_label_name))
        # 创建图片
        cv2.imwrite("{}/{}/{}".format(save_path,
                                      im_label_name,
                                      im_name.decode("utf-8")),
                    im_data)