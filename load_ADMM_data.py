import os
import random

import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable
def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1],128,128
    divid_point_h = 64
    divid_point_w = 64
    output_img = torch.zeros(c,h,w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
        if i > 50:
            print("i > 50")
            break
        scene_path = os.path.join(path, scene_list[i])
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand'] / 65536.
        elif "img" in img_dict:
            img = img_dict['img'] / 65536.
        elif "data" in img_dict:
            img = img_dict['data']
        img = img.astype(np.float32)
        if img.max() == 0:
            continue
        img = img[::2,::2,:]
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 128, 128, 121))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['data']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data
def shuffle_crop(train_data, batch_size, crop_size=128, argument=True):
    if argument:
        # ground truth
        gt_batch = []
        # 随机选择一半的样本索引，用于使用原始数据进行处理。
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, 121), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            h, w, _ = img.shape
            # 随机生成裁剪点
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # 随机选取数据
        processed_data = np.zeros((4, 64, 64, 121), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                h,w,_ = train_data[sample_list[j]].shape
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                # print(h,w,x_index, y_index, crop_size, crop_size // 2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 121), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch
if __name__ == '__main__':
    data_path = r"D:\learn\MST-main\datasets\wayho_data"
    train_set = LoadTraining(data_path)
    gt_batch = shuffle_crop(train_set, batch_size=10, crop_size=128)
    gt = Variable(gt_batch).float()
    print()
    print()

