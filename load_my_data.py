#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# load_my_data.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/7/14 上午9:23:13
# @Explanation  : 从cifar10格式数据中导入自己数据
"""

import os
import sys
import numpy as np
import cv2
from six.moves import cPickle
from keras import backend as K
from keras.utils import np_utils

def load_batch(fpath, label_key='labels'):
    """
    # 说明：
        - 从生成cifar10数据的batch中导入数据
    # 参数：
        - fpath: str, batch所在目录
        - label_key: str, 检索的key值
    # 返回：
        - datas: numpy, 大小为[nums, channels, width, height]
        - labels: list, 大小为[nums, 1]
    """

    fid_r = open(fpath, 'rb')
    if sys.version_info < (3,):
        dicts = cPickle.load(fid_r)
    else:
        dicts = cPickle.load(fid_r, encoding='bytes')
        dicts_decoded = {}
        for k, val in dicts.items():
            dicts_decoded[k.decode('utf8')] = val
        dicts = dicts_decoded
    fid_r.close()

    datas = dicts['data']
    labels = dicts[label_key]

    # 这里需设置为生成cifar10文件时，图像的大小
    datas = datas.reshape(datas.shape[0], 3, 224, 224)

    return datas, labels

def load_mydata_with_cifar10(fpath, mode, num_samples, num_batch, img_size, num_classes, ratio):
    """
    # 说明:
        - 从cifar10格式数据中，导入自己生成的数据
    # 参数:
        - fpath: str, cifar10格式文件目录
        - mode: str, train or valid
        - num_samples: int, 样本总数
        - num_batch: int, 样本batch数
        - img_size: 图像大小，对应模型的图像大小
        - num_classes: int, 样本分类数
        - ratio: float，用来训练或测试比例
    # 返回:
        - datas: numpy, 大小为[nums, img_size, img_size, channels]
        - labels: list, 大小为[nums, num_classes]
    """

    num = int(num_samples * ratio)

    # 这里图像大小为，生成cifar时的图像大小
    datas = np.zeros((num_samples, 3, 224, 224), dtype='uint8')
    labels = np.zeros((num_samples, ), dtype='uint8')

    samples_per_batch = num_samples / num_batch
    if mode == 'train':
        for i in range(num_batch):
            datas_batch, labels_batch = load_batch(os.path.join(fpath, 'train_list_' + str(i)))

            start = i * samples_per_batch
            end = min((i + 1) * samples_per_batch, num_samples)
            datas[start:end, :, :, :] = datas_batch
            labels[start:end] = labels_batch

            # 取到足够数据后，退出
            if end > num:
                break
    else:
        for i in range(num_batch):
            datas_batch, labels_batch = load_batch(os.path.join(fpath, 'valid_list_' + str(i)))

            start = i * samples_per_batch
            end = min((i + 1) * samples_per_batch, num_samples)
            datas[start:end, :, :, :] = datas_batch
            labels[start:end] = labels_batch

            # 取到足够数据后，退出
            if end > num:
                break

    # 转换为[nums, width, height, channels]
    if K.image_data_format() == 'channels_last':
        datas = datas.transpose(0, 2, 3, 1)

    # 图像大小转换，转换成模型需要大小
    if K.image_dim_ordering() == 'th':
        datas = np.array([cv2.resize(img.transpose(1, 2, 0), (img_size, img_size)).transpose(2, 0, 1) \
                                for img in datas[:num, :, :, :]])
    else:
        datas = np.array([cv2.resize(img, (img_size, img_size)) for img in datas[:num, :, :, :]])

    # 标签类型转换
    labels = np_utils.to_categorical(labels[:num], num_classes)

    return datas, labels

if __name__ == '__main__':

    NUM_TRAIN_SAMPLES = 23968
    NUM_VALID_SAMPLES = 10275
    NUM_BATCH = 30

    X_TEST, Y_TEST = load_mydata_with_cifar10('./my_data_set/', 'valid', NUM_VALID_SAMPLES, NUM_BATCH, 224, 3, 0.01)

    # cv2.imwrite('111111.png', X_TEST[0])

    # x_test, y_test = load_batch(u'./my_data_set/train_list_0')
    print X_TEST[0].shape, Y_TEST.shape
