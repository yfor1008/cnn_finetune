#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# generate_my_data.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/7/13 下午5:23:28
# @Explanation  : 按照cifar10数据格式，生成自己的数据
#
# 原始数据格式：一个文件为一个分类，文件夹名为数字，如：
# data：
#       class0
#           image0,image1, ...
#       class1
#           image0,image1, ...
#       class2
#           image0,image1, ...
#       ......
# 处理，将所有图像放在一个文件夹，同时生成train和valid的list
# list里面，标注了图像的分类，如
# image0 0
# image1 0
# ......
# imagen m
# n为图像个数，m为分类种类数
#
# 利用train和valid的list生成cifar10的文件格式
#
"""

import os
import cPickle
import shutil
import random
import cv2
import numpy as np

# import win32file

def get_train_valid_list(src_path, dst_path, ratio):
    """
    ### 说明：
        - 生成训练和测试list
    ### 参数：
        - src_path: str, 原始数据目录，目录下每个文件夹为一个分类，注意这里文件夹名为数字
        - dst_path: str, 保存结果目录，该目录下会生成train和valid的list，同时有data文件夹保存所有数据
        - ratio: float, 训练train的比例，剩下的为测试valid的，范围为(0.0, 1.0)
    ### 返回：
        - 无
    """

    # 生成结果目录
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    if not os.path.exists(os.path.join(dst_path, 'data')):
        os.mkdir(os.path.join(dst_path, 'data'))

    # 生成数据list
    train_list = open(os.path.join(dst_path, 'train_list.txt'), 'w')
    valid_list = open(os.path.join(dst_path, 'valid_list.txt'), 'w')
    folders = os.listdir(src_path)
    for folder in folders:
        print folder
        imgs = os.listdir(os.path.join(src_path, folder))
        train_num = int(len(imgs) * ratio)

        for img in imgs:
            # win32file.CopyFile(os.path.join(src_path, folder, img), os.path.join(dst_path, 'data', img), 0)
            shutil.copy(os.path.join(src_path, folder, img), os.path.join(dst_path, 'data', img))

        for idx, img in enumerate(imgs):
            if idx < train_num:
                train_list.write('%s %s\n' % (img, folder))
            else:
                valid_list.write('%s %s\n' % (img, folder))

    train_list.close()
    valid_list.close()

def pickled(savepath, datas, labels, fnames, batch_num=1, mode='train'):
    """
    ### 说明：
        - 将数据保存成cifar10格式
    ### 参数：
        - savepath: str, 保存目录
        - datas: numpy, 图像数据，[nums, pixels]，nums为图像数量，pixels为图像像素个数
        - labels: int, 图像标签，[nums, 1]
        - fnames: str, 图像名字，[nums, ]
        - batch_num: int, 将图像保存成batch的个数
        - mode: str, 模式, train or valid
    ### 返回：
        - 无
    """

    samples_per_batch = len(labels) / batch_num
    for i in xrange(batch_num):
        start = i * samples_per_batch
        end = min((i + 1) * samples_per_batch, len(labels))

        dicts = {'data': datas[start:end, :], 'labels': labels[start:end], 'filenames': fnames[start:end]}

        if mode == 'train':
            dicts['batch_label'] = "train batch {} of {}".format(i, batch_num)
        else:
            dicts['batch_label'] = "valid batch {} of {}".format(i, batch_num)
        with open(savepath + '_' + str(i), 'wb') as fid_w:
            cPickle.dump(dicts, fid_w)

def get_my_data(data_path, shape):
    """
    ### 说明：
        - 利用list生成cifar10格式数据，生成前需对图像进行缩放，list都为txt格式，
    ### 参数：
        - data_path: str, 数据目录，包含图像数据、train_list、valid_list
        - shape: int, 保存为cifar文件时，图像的大小
    ### 返回：
        - 无
    """

    # 保存时图像大小
    image_size = shape * shape

    # 处理每个list
    data_lists = [lst for lst in os.listdir(data_path) if lst.endswith('txt')]
    for lst in data_lists:
        print lst
        with open(os.path.join(data_path, lst), 'r') as fid_r:
            lines = fid_r.read().splitlines()

        # cifar10每幅图像为一行数据
        # 图像数据
        datas = np.zeros((len(lines), image_size * 3), dtype=np.uint8)
        # 图像名字
        fnames = [fname.split(' ')[0] for fname in lines]
        # 图像label
        labels = [int(label.split(' ')[1]) for label in lines]

        # 随机打乱顺序
        random_list = random.sample(range(len(labels)), len(labels))
        fnames = [fnames[i] for i in random_list]
        labels = [labels[i] for i in random_list]

        for idx, fname in enumerate(fnames):
            img = cv2.imread(os.path.join(data_path, 'data', fname), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 截取中间区域
            img = img[80:400, 80:400, :]

            img = cv2.resize(img, (shape, shape))

            datas[idx, :image_size] = np.reshape(img[:, :, 0], image_size)
            datas[idx, image_size : image_size*2] = np.reshape(img[:, :, 1], image_size)
            datas[idx, image_size*2 :] = np.reshape(img[:, :, 2], image_size)

        # 保存为cifar10格式
        print lst[:-4]
        if lst[:-4] == 'train_list':
            pickled(os.path.join(data_path, lst[:-4]), datas, labels, fnames, batch_num=30, mode='train')
        else:
            pickled(os.path.join(data_path, lst[:-4]), datas, labels, fnames, batch_num=30, mode='valid')

if __name__ == '__main__':

    # 原始数据目录
    # for windows
    # SRC_PATH = u'//172.16.100.5/windows D/yuanwenjin/tensorflow/samples/train_val_wrong_change'
    # for linux
    SRC_PATH = u'/home/yuanwenjin/tensorflow/samples/train_val_wrong_change'
    # 保存目录
    DST_PATH = u'./my_data_set'

    # 生成list
    # get_train_valid_list(SRC_PATH, DST_PATH, 0.7)
    # 生成cifar10格式数据
    get_my_data(DST_PATH, 224)
