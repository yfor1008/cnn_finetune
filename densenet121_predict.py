#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# densenet121_predict.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/7/14 下午3:00:41
# @Explanation  : 使用训练好的模型，进行预测
"""

import numpy as np
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

    # x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(tensor_x)
    # x_fc = Dense(1000, name='fc6')(x_fc)
    # x_fc = Activation('softmax', name='prob')(x_fc)

    # model = Model(img_input, x_fc, name='densenet')

    # if K.image_dim_ordering() == 'th':
    #     # Use pre-trained weights for Theano backend
    #     weights_path = 'imagenet_models/densenet121_weights_th.h5'
    # else:
    #     # Use pre-trained weights for Tensorflow backend
    #     weights_path = 'imagenet_models/densenet121_weights_tf.h5'

    # model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(tensor_x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)
    model.load_weights('cifar10_weights.h5', by_name=True)

    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def conv_block(tensor_x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
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
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
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
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        ### Arguments
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

if __name__ == '__main__':

    # Example to fine-tune on n samples from digestive data

    IMG_ROWS, IMG_COLS = 480, 480 # Resolution of inputs
    CHANNEL = 3
    NUM_CLASSES = 3

    # Load our model
    MODEL = densenet121_model(img_rows=IMG_ROWS, img_cols=IMG_COLS, color_type=CHANNEL, num_classes=NUM_CLASSES)

    IMG = cv2.imread('2_03068.jpg')
    # IMG = cv2.resize(IMG[80:400, 80:400, :], (224, 224))
    # if K.image_dim_ordering() == 'th':
    #     IMG = IMG.transpose((2, 0, 1))

    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('22222.png', IMG)
    IMG = np.expand_dims(IMG, axis=0)

    IMG = IMG.astype('float32') / 255.0

    OUT = MODEL.predict(IMG)
    print OUT, np.argmax(OUT)

    # predictions = model.predict(X_train, batch_size=batch_size, verbose=1)
    # print predictions
    # print np.argmax(predictions, 1)
    # print np.argmax(Y_train, 1)
