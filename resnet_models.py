#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# resnet_models.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/9/4 下午4:31:48
# @Explanation  : resnet模型
"""

import os
import math
import argparse
import numpy as np
import scipy.io as scio
import cv2
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import ZeroPadding2D, Dropout, Flatten, add, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, save_model
from keras import backend as K

from custom_layers.scale_layer import Scale

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard

import sys
sys.setrecursionlimit(3000)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    ### 说明:
        - The identity_block is the block that has no conv layer at shortcut

    ### Arguments:
        - input_tensor: input tensor
        - kernel_size: defualt 3, the kernel size of middle conv layer at main path
        - filters: list of integers, the nb_filters of 3 conv layer at main path
        - stage: integer, current stage label, used for generating layer names
        - block: 'a','b'..., current block label, used for generating layer names
    """
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    tensor_x = Conv2D(nb_filter1, (1, 1), use_bias=False, name=conv_name_base + '2a')(input_tensor)
    tensor_x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(tensor_x)
    tensor_x = Scale(axis=bn_axis, name=scale_name_base + '2a')(tensor_x)
    tensor_x = Activation('relu', name=conv_name_base + '2a_relu')(tensor_x)

    tensor_x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(tensor_x)
    tensor_x = Conv2D(nb_filter2, (kernel_size, kernel_size), use_bias=False, name=conv_name_base + '2b')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(tensor_x)
    tensor_x = Scale(axis=bn_axis, name=scale_name_base + '2b')(tensor_x)
    tensor_x = Activation('relu', name=conv_name_base + '2b_relu')(tensor_x)

    tensor_x = Conv2D(nb_filter3, (1, 1), use_bias=False, name=conv_name_base + '2c')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(tensor_x)
    tensor_x = Scale(axis=bn_axis, name=scale_name_base + '2c')(tensor_x)

    tensor_x = add([tensor_x, input_tensor], name='res' + str(stage) + block)
    tensor_x = Activation('relu', name='res' + str(stage) + block + '_relu')(tensor_x)
    return tensor_x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''
    ### 说明
        - conv_block is the block that has a conv layer at shortcut

    ### Arguments
        - input_tensor: input tensor
        - kernel_size: defualt 3, the kernel size of middle conv layer at main path
        - filters: list of integers, the nb_filters of 3 conv layer at main path
        - stage: integer, current stage label, used for generating layer names
        - block: 'a','b'..., current block label, used for generating layer names

    ### 注意
        - Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        - And the shortcut should have strides=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    tensor_x = Conv2D(nb_filter1, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '2a')(input_tensor)
    tensor_x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(tensor_x)
    tensor_x = Scale(axis=bn_axis, name=scale_name_base + '2a')(tensor_x)
    tensor_x = Activation('relu', name=conv_name_base + '2a_relu')(tensor_x)

    tensor_x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(tensor_x)
    tensor_x = Conv2D(nb_filter2, (kernel_size, kernel_size), use_bias=False, name=conv_name_base + '2b')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(tensor_x)
    tensor_x = Scale(axis=bn_axis, name=scale_name_base + '2b')(tensor_x)
    tensor_x = Activation('relu', name=conv_name_base + '2b_relu')(tensor_x)

    tensor_x = Conv2D(nb_filter3, (1, 1), use_bias=False, name=conv_name_base + '2c')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(tensor_x)
    tensor_x = Scale(axis=bn_axis, name=scale_name_base + '2c')(tensor_x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    tensor_x = add([tensor_x, shortcut], name='res' + str(stage) + block)
    tensor_x = Activation('relu', name='res' + str(stage) + block + '_relu')(tensor_x)

    return tensor_x

def resnet_model(img_rows, img_cols, color_type=1, \
                 model_size=None, nb_layers=None):
    '''
    ### 说明:
        - DenseNet Model for Keras
        - Model Schema is based on https://github.com/flyyufelix/DenseNet-Keras
        - ImageNet Pretrained Weights

    ### Arguments:
        - img_rows: (int), image height
        - img_cols: (int), image width
        - color_type: (int), image channel
        - model_size: (string), layer number of model
        - nb_layers: (list), number of filters in each block
    ### Returns
        - A Keras model instance.
    '''

    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # conv1
    tensor_x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    tensor_x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(tensor_x)
    tensor_x = Scale(axis=bn_axis, name='scale_conv1')(tensor_x)
    tensor_x = Activation('relu', name='conv1_relu')(tensor_x)
    tensor_x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(tensor_x)

    # conv2_x
    tensor_x = conv_block(tensor_x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # for i in range(1, nb_layers[0]):
    #     tensor_x = identity_block(tensor_x, 3, [64, 64, 256], stage=2, block='b' + str(i))
    tensor_x = identity_block(tensor_x, 3, [64, 64, 256], stage=2, block='b')
    tensor_x = identity_block(tensor_x, 3, [64, 64, 256], stage=2, block='c')

    # conv3_x
    tensor_x = conv_block(tensor_x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, nb_layers[1]):
        tensor_x = identity_block(tensor_x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    # conv4_x
    tensor_x = conv_block(tensor_x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, nb_layers[2]):
        tensor_x = identity_block(tensor_x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    # conv5_x
    tensor_x = conv_block(tensor_x, 3, [512, 512, 2048], stage=5, block='a')
    # for i in range(1, nb_layers[3]):
    #     tensor_x = identity_block(tensor_x, 3, [512, 512, 2048], stage=5, block='b' + str(i))
    tensor_x = identity_block(tensor_x, 3, [512, 512, 2048], stage=5, block='b')
    tensor_x = identity_block(tensor_x, 3, [512, 512, 2048], stage=5, block='c')

    # fc
    # x_fc = AveragePooling2D((7, 7), name='avg_pool')(tensor_x)
    # x_fc = Flatten()(x_fc)
    # x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    x_fc = GlobalAveragePooling2D(name='avg_pool')(tensor_x)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'imagenet_models/resnet' + model_size + '_weights_th.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'imagenet_models/resnet' + model_size + '_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    return model

def add_new_layer(base_model, layer_name, nb_classes):
    '''
    ### 说明:
        - 增加模型的层
    ### 参数:
        - base_model: 原始模型
        - layer_name: (string), 模型某一层的名字
        - nb_classes: (int), 新模型的分类数目
    ### 返回:
        - model: 新模型
    '''

    # 获取模型输出
    tensor_x = base_model.get_layer(layer_name).output

    # 增加新的层
    x_newfc = GlobalAveragePooling2D(name='avg_pool')(tensor_x)
    x_newfc = Dense(nb_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(inputs=base_model.input, outputs=x_newfc)
    model.load_weights('resnet_101_finetune_weights_92%.h5')
    return model

def step_decay(epoch):
    '''
    ### 说明：
        - 调整学习率

    ### 参数：
        - epoch: (int), 迭代次数

    ### 返回：
        - lr_rate: (float), 学习率
    '''

    initial_lr_rate = 0.001
    drop = 0.6
    epochs_drop = 5.0
    lr_rate = initial_lr_rate * math.pow(drop, math.floor((1.0 + epoch) / epochs_drop))

    return lr_rate

def setup_to_transfer_or_finetune(model, base_model, mode):
    '''
    ### 说明:
        - 固定参数，设置迁移学习，不固定参数，进行fine-tune
    ### 参数:
        - model: 新模型
        - base_model: 原始模型
        - mode: 模式，transfer or finetune
    ### 返回:
        - 无
    '''

    if mode == 'transfer':
        for layer in base_model.layers:
            layer.trainable = False
    # rmsprop = RMSprop(lr=0.0001, decay=0.0)
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

def generate_resnet(img_rows, img_cols, color_type=1, \
                    num_classes=None, model_size=None, train_mode='finetune'):
    '''
    ### 说明:
        - DenseNet Model for Keras
        - Model Schema is based on https://github.com/flyyufelix/DenseNet-Keras
        - ImageNet Pretrained Weights

    ### Arguments:
        - img_rows: (int), image height
        - img_cols: (int), image width
        - color_type: (int), image channel
        - model_size: (string), layer number of model
        - train_mode: (string), finetune or transfer
    ### Returns
        - A Keras model instance.
    '''

    if model_size == '101':
        nb_layers = [3, 4, 23, 3]
    elif model_size == '152':
        nb_layers = [3, 8, 36, 3]
    # 生成模型
    base_model = resnet_model(img_rows=img_rows, img_cols=img_cols, color_type=color_type, \
                              model_size=model_size, nb_layers=nb_layers)

    # 模型全连接前一层名字
    # layer_name = 'res' + str(len(nb_layers) + 1) + '_blk'
    layer_name = 'res5c_relu'

    model = add_new_layer(base_model, layer_name, num_classes)
    setup_to_transfer_or_finetune(model, base_model, train_mode)

    return model

def parse_args():
    '''
    ### 说明：
        - 输入参数说明
    '''
    parser = argparse.ArgumentParser(description='resnet model')

    parser.add_argument('--img_rows', dest='img_rows', help='image height', default=480, type=int)
    parser.add_argument('--img_cols', dest='img_cols', help='image width', default=480, type=int)
    parser.add_argument('--color_type', dest='color_type', help='image channel', default=3, type=int)
    parser.add_argument('--model_size', dest='model_size', help='layers of model', \
                        default='101', type=str, choices=['101', '152'])
    parser.add_argument('--train_mode', dest='train_mode', help='training mode', \
                        default='finetune', type=str, choices=['finetune', 'transfer'])
    parser.add_argument('--num_classes', dest='num_classes', help='classification number', \
                        default=3, type=int)
    parser.add_argument('--use_mode', dest='use_mode', help='1-train or 0-test', \
                        default=0, type=int, choices=[0, 1])

    args = parser.parse_args()

    return args

###########################################################################################################
if __name__ == '__main__':

    ARGS = parse_args()

    # 生成模型
    MODEL = generate_resnet(img_rows=ARGS.img_rows, img_cols=ARGS.img_cols, color_type=ARGS.color_type, \
                            model_size=ARGS.model_size, num_classes=ARGS.num_classes, train_mode=ARGS.train_mode)
    # MODEL.summary()

    # 训练or测试
    if ARGS.use_mode:
        # 训练参数
        BATCH_SIZE = 8
        EPOCHS = 30
        # 生成数据
        TRAIN_DIR = '/home/get_samples/samples_train/train'
        # VALID_DIR = '/home/get_samples/samples_train_1/train'
        VALID_DIR = '/home/get_samples/samples_train/valid'
        # VALID_DIR = '/home/get_samples/samples_train_1/valid'
        DATA_GEN = ImageDataGenerator(rescale=1./255, \
                                        shear_range=10, \
                                        width_shift_range=0.1, \
                                        height_shift_range=0.1)
        TRAIN_GEN = DATA_GEN.flow_from_directory(directory=TRAIN_DIR, \
                                                target_size=(ARGS.img_rows, ARGS.img_cols), \
                                                batch_size=BATCH_SIZE, \
                                                class_mode='categorical')
        VALID_GEN = DATA_GEN.flow_from_directory(directory=VALID_DIR, \
                                                target_size=(ARGS.img_rows, ARGS.img_cols), \
                                                batch_size=BATCH_SIZE, \
                                                class_mode='categorical')

        # start
        TENSORBOARD = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=False, write_images=True)
        LR_RATE = LearningRateScheduler(step_decay)
        EATLYSTOP = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
        MODELCHECHPOINT = ModelCheckpoint('resnet_' + ARGS.model_size + '_' + ARGS.train_mode + '_' \
                                                    + str(ARGS.img_rows) + '_' + str(ARGS.img_cols) + '_weights.h5', \
                                            monitor='val_loss', \
                                            # save_best_only=True, \
                                            verbose=1, save_weights_only=True)
        HIST = MODEL.fit_generator(TRAIN_GEN, \
                                steps_per_epoch=27500, \
                                epochs=EPOCHS, \
                                verbose=1, \
                                callbacks=[MODELCHECHPOINT, TENSORBOARD], \
                                validation_data=VALID_GEN, \
                                validation_steps=7000)
        # print HIST.history

        MODEL.save_weights('resnet_' + ARGS.model_size + '_' + ARGS.train_mode + '_' \
                                     + str(ARGS.img_rows) + '_' + str(ARGS.img_cols) + '_weights.h5')
    else:
        # 测试参数
        BATCH_SIZE = 512

        # load模型
        MODEL.load_weights('resnet_' + ARGS.model_size + '_' + ARGS.train_mode + '_' \
                                     + str(ARGS.img_rows) + '_' + str(ARGS.img_cols) + '_weights.h5')

        # 数据路径
        # DIGESTIVE_DIR = u'/home/tensorflow/digestive_change'
        # DIGESTIVE_DIR = u'/home/get_samples/samples_train/train'
        DIGESTIVE_DIR = u'/home/get_samples/samples_train/valid'
        CLASSIFIED_DIR = u'/home/densenet/cnn_finetune-master/resnet_results' + \
                        '_' + ARGS.model_size + '_' + str(ARGS.img_rows) + '_' + ARGS.train_mode

        # 每个数据处理
        DATA_DIRS = []
        for data_dir in os.listdir(DIGESTIVE_DIR):
            DATA_DIRS.append(os.path.join(DIGESTIVE_DIR, data_dir))

        for data_dir in DATA_DIRS:
            # print os.path.basename(data_dir)
            imagelists = [img for img in os.listdir(data_dir) if img.endswith('jpg')]
            imagelists.sort()
            # imagelists = imagelists[0:2]

            batch_num = int(math.ceil(float(len(imagelists)) / float(BATCH_SIZE)))
            # print batch_num

            predictions = []
            for batch_idx in xrange(batch_num):
                start = batch_idx * BATCH_SIZE
                end = min((batch_idx + 1) * BATCH_SIZE, len(imagelists))

                imgs = []
                for idx in xrange(start, end):
                    img = cv2.imread(os.path.join(data_dir, imagelists[idx]).encode('utf8'))
                    img = cv2.resize(img, (ARGS.img_rows, ARGS.img_cols))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgs.append(img)
                imgs = np.array(imgs, dtype='float32') / 255.0
                prediction = np.argmax(MODEL.predict(np.array(imgs)), 1).tolist()
                predictions.extend(prediction)
                # print batch_idx, len(predictions)
            scio.savemat(os.path.join(CLASSIFIED_DIR, os.path.basename(data_dir) + '.mat'), {'data': predictions})
