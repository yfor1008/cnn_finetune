#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# densenet121_transfer_predict_digestive.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/7/25 上午8:34:38
# @Explanation  : 批量预测
"""

import os
import math
import numpy as np
import scipy.io as scio
import cv2
from keras.optimizers import SGD
from keras.layers import Input, concatenate, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

# from sklearn.metrics import log_loss

from custom_layers.scale_layer import Scale

# from load_cifar10 import load_cifar10_data

# from load_my_data import load_mydata_with_cifar10

def densenet121_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, \
                        nb_filter=64, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    ### 说明：
        - DenseNet 121 Model for Keras
        - Model Schema is based on https://github.com/flyyufelix/DenseNet-Keras
        - ImageNet Pretrained Weights
        - Theano:
            https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ
        - TensorFlow:
            https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc

    ### Arguments
        - nb_dense_block: number of dense blocks to add to end
        - growth_rate: number of filters to add per dense block
        - nb_filter: initial number of filters
        - reduction: reduction factor of transition blocks.
        - dropout_rate: dropout rate
        - weight_decay: weight decay factor
        - classes: optional number of classes to classify images
        - weights_path: path to pre-trained weights
    ### Returns
        - A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global CONCAT_AXIS
    if K.image_dim_ordering() == 'tf':
        CONCAT_AXIS = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        CONCAT_AXIS = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6, 12, 24, 16] # For DenseNet-121

    # Initial convolution
    tensor_x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    tensor_x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=CONCAT_AXIS, name='conv1_bn')(tensor_x)
    tensor_x = Scale(axis=CONCAT_AXIS, name='conv1_scale')(tensor_x)
    tensor_x = Activation('relu', name='relu1')(tensor_x)
    tensor_x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(tensor_x)
    tensor_x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(tensor_x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        tensor_x, nb_filter = dense_block(tensor_x, stage, nb_layers[block_idx], nb_filter, \
                                            growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        tensor_x = transition_block(tensor_x, stage, nb_filter, compression=compression, \
                                            dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    tensor_x, nb_filter = dense_block(tensor_x, final_stage, nb_layers[-1], nb_filter, \
                                            growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    tensor_x = BatchNormalization(epsilon=eps, axis=CONCAT_AXIS, name='conv' + str(final_stage) + '_blk_bn')(tensor_x)
    tensor_x = Scale(axis=CONCAT_AXIS, name='conv' + str(final_stage) + '_blk_scale')(tensor_x)
    tensor_x = Activation('relu', name='relu' + str(final_stage) + '_blk')(tensor_x)

    x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(tensor_x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'imagenet_models/densenet121_weights_th.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'imagenet_models/densenet121_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # # 固定参数
    # for layer in model.layers:
    #     layer.trainable = False

    # # Truncate and replace softmax layer for transfer learning
    # # Cannot use model.layers.pop() since model is not of Sequential() type
    # # The method below works since pre-trained weights are stored in layers but not in the model
    # x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(tensor_x)
    # x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    # x_newfc = Activation('softmax', name='prob')(x_newfc)

    # new_model = Model(img_input, x_newfc)

    # # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def conv_block(tensor_x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''
    ### 说明:
        Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
    ### Arguments
        - tensor_x: input tensor
        - stage: index for dense block
        - branch: layer index within each dense block
        - nb_filter: number of filters
        - dropout_rate: dropout rate
        - weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    tensor_x = BatchNormalization(epsilon=eps, axis=CONCAT_AXIS, name=conv_name_base + '_x1_bn')(tensor_x)
    tensor_x = Scale(axis=CONCAT_AXIS, name=conv_name_base + '_x1_scale')(tensor_x)
    tensor_x = Activation('relu', name=relu_name_base + '_x1')(tensor_x)
    tensor_x = Conv2D(inter_channel, (1, 1), name=conv_name_base + '_x1', use_bias=False)(tensor_x)

    if dropout_rate:
        tensor_x = Dropout(dropout_rate)(tensor_x)

    # 3x3 Convolution
    tensor_x = BatchNormalization(epsilon=eps, axis=CONCAT_AXIS, name=conv_name_base+'_x2_bn')(tensor_x)
    tensor_x = Scale(axis=CONCAT_AXIS, name=conv_name_base + '_x2_scale')(tensor_x)
    tensor_x = Activation('relu', name=relu_name_base + '_x2')(tensor_x)
    tensor_x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(tensor_x)
    tensor_x = Conv2D(nb_filter, (3, 3), name=conv_name_base + '_x2', use_bias=False)(tensor_x)

    if dropout_rate:
        tensor_x = Dropout(dropout_rate)(tensor_x)

    return tensor_x

def transition_block(tensor_x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    '''
    ### 说明:
        Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
    ### Arguments
        - tensor_x: input tensor
        - stage: index for dense block
        - nb_filter: number of filters
        - compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
        - dropout_rate: dropout rate
        - weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    tensor_x = BatchNormalization(epsilon=eps, axis=CONCAT_AXIS, name=conv_name_base+'_bn')(tensor_x)
    tensor_x = Scale(axis=CONCAT_AXIS, name=conv_name_base+'_scale')(tensor_x)
    tensor_x = Activation('relu', name=relu_name_base)(tensor_x)
    tensor_x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(tensor_x)

    if dropout_rate:
        tensor_x = Dropout(dropout_rate)(tensor_x)

    tensor_x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(tensor_x)

    return tensor_x


def dense_block(tensor_x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, \
                weight_decay=1e-4, grow_nb_filters=True):
    '''
    # 说明:
        Build a dense_block where the output of each conv_block is fed to subsequent ones
    # Arguments
        - tensor_x: input tensor
        - stage: index for dense block
        - nb_layers: the number of layers of conv_block to append to the model.
        - nb_filter: number of filters
        - growth_rate: growth rate
        - dropout_rate: dropout rate
        - weight_decay: weight decay factor
        - grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    # eps = 1.1e-5
    concat_feat = tensor_x

    for i in range(nb_layers):
        branch = i + 1
        tensor_x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, tensor_x], axis=CONCAT_AXIS, \
                                    name='concat_' + str(stage) + '_' + str(branch))
        # concat_feat = add([concat_feat, tensor_x])

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

def add_new_layer(base_model, nb_classes):
    '''
    ### 说明:
        - 增加模型的层
    ### 参数:
        - base_model: 原始模型
        - nb_classes: 新模型的分类
    ### 返回:
        - model: 新模型
    '''

    # 并获取模型输出
    # temp_model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('relu5_blk').output)
    # tenser_x = temp_model.outputs
    tenser_x = base_model.get_layer('relu5_blk').output

    # 增加新的层
    tenser_x = GlobalAveragePooling2D()(tenser_x)
    # tenser_x = Dense(1024, activation='relu')(tenser_x) # new FC
    predicts = Dense(nb_classes, activation='softmax')(tenser_x) # new softmax

    model = Model(inputs=base_model.input, outputs=predicts)
    return model

def setup_to_transfer_learn(model, base_model):
    '''
    ### 说明:
        - 固定参数，设置迁移学习
    ### 参数:
        - model: 新模型
        - base_model: 原始模型
    ### 返回:
        - 无
    '''

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':

    IMG_ROWS, IMG_COLS = 224, 224 # Resolution of inputs
    CHANNEL = 3
    NUM_CLASSES = 3

    BATCH_SIZE = 512

    # Load our model
    BASE_MODEL = densenet121_model(img_rows=IMG_ROWS, img_cols=IMG_COLS, color_type=CHANNEL, num_classes=NUM_CLASSES)
    MODEL = add_new_layer(BASE_MODEL, NUM_CLASSES)
    MODEL.load_weights('cifar10_transfer_weights.h5', by_name=True)
    # setup_to_transfer_learn(MODEL, BASE_MODEL)

    # 数据路径
    DIGESTIVE_DIR = u'/home/yuanwenjin/tensorflow/digestive_change'
    CLASSIFIED_DIR = u'/home/yuanwenjin/densenet/cnn_finetune-master/results_transfer'

    # 每个数据处理
    DATA_DIRS = []
    for data_dir in os.listdir(DIGESTIVE_DIR):
        DATA_DIRS.append(os.path.join(DIGESTIVE_DIR, data_dir))

    for data_dir in DATA_DIRS:
        print os.path.basename(data_dir)
        imagelists = [img for img in os.listdir(data_dir) if img.endswith('jpg')]
        imagelists.sort()
        # imagelists = imagelists[0:2]

        batch_num = int(math.ceil(float(len(imagelists)) / float(BATCH_SIZE)))
        print batch_num

        predictions = []
        for batch_idx in xrange(batch_num):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, len(imagelists))

            imgs = []
            for idx in xrange(start, end):
                img = cv2.imread(os.path.join(data_dir, imagelists[idx]).encode('utf8'))
                img = cv2.resize(img[80:400, 80:400, :], (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
            imgs = np.array(imgs, dtype='float32') / 255.0
            prediction = np.argmax(MODEL.predict(np.array(imgs)), 1).tolist()
            predictions.extend(prediction)
            print batch_idx, len(predictions)
        scio.savemat(os.path.join(CLASSIFIED_DIR, os.path.basename(data_dir) + '.mat'), {'data': predictions})
