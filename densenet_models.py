#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# densenet_models.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/9/4 下午1:56:00
# @Explanation  : densenet模型
"""

import os
import math
import argparse
import numpy as np
import scipy.io as scio
import cv2
from keras.layers import Input, concatenate, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

from custom_layers.scale_layer import Scale

def conv_block(tensor_x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''
    ### 说明:
        - Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout

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
    # 说明:
        Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
    # Arguments
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

def densenet_model(img_rows, img_cols, color_type=1, \
                   nb_dense_block=4, growth_rate=None, \
                   nb_filter=None, nb_layers=None, \
                   reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, \
                   model_size=None):
    '''
    ### 说明:
        - DenseNet Model for Keras
        - Model Schema is based on https://github.com/flyyufelix/DenseNet-Keras
        - ImageNet Pretrained Weights

    ### Arguments:
        - img_rows: (int), image height
        - img_cols: (int), image width
        - color_type: (int), image channel
        - nb_dense_block: (int), number of dense blocks to add to end
        - growth_rate: (int), number of filters to add per dense block
        - nb_filter: (int), initial number of filters
        - nb_layers: (list), number of filters in each block
        - reduction: (float), reduction factor of transition blocks.
        - dropout_rate: (float), dropout rate
        - weight_decay: (float), weight decay factor
        - model_size: (string), layer number of model
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
        stage = block_idx + 2
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
        weights_path = 'imagenet_models/densenet' + model_size + '_weights_th.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'imagenet_models/densenet' + model_size + '_weights_tf.h5'

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
    eps = 1.1e-5
    tensor_x = Conv2D(128, (3, 3), name='conv_new', use_bias=False)(tensor_x)
    tensor_x = AveragePooling2D((2, 2), strides=(2, 2), name='pool_new')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=CONCAT_AXIS, name='bn_new')(tensor_x)
    tensor_x = Activation('relu', name='relu_new')(tensor_x)

    # tensor_x = AveragePooling2D((2, 2), strides=(2, 2))(tensor_x)
    # tensor_x = GlobalMaxPooling2D()(tensor_x)
    # # tensor_x = Dense(1024, activation='relu')(tensor_x) # new FC
    # predicts = Dense(nb_classes, activation='softmax')(tensor_x) # new softmax
    tensor_x = GlobalAveragePooling2D(name='pool'+str(5))(tensor_x)
    tensor_x = Dense(nb_classes, name='fc6')(tensor_x)
    predicts = Activation('softmax', name='prob')(tensor_x)

    model = Model(inputs=base_model.input, outputs=predicts)
    # model.load_weights('densenet_161_finetune_weights.h5')

    return model

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
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def generate_densenet(img_rows, img_cols, color_type, \
                      reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, \
                      model_size=None, num_classes=None, train_mode='finetune'):
    '''
    ### 说明：
        - 生成需要的模型

    ### 参数：
        - img_rows: (int), image height
        - img_cols: (int), image width
        - color_type: (int), image channel
        - reduction: (float), reduction factor of transition blocks.
        - dropout_rate: (float), dropout rate
        - weight_decay: (float), weight decay factor
        - model_size: (string), layer number of model
        - num_classes: (int), classes
        - train_mode: (string), finetune or transfer

    ### 返回：
        - MODEL: 生成好的模型
    '''

    # dense模型参数: 121/169/161
    if model_size == '121':
        nb_dense_block = 4
        growth_rate = 32
        nb_filter = 64
        nb_layers = [6, 12, 24, 16]
    elif model_size == '169':
        nb_dense_block = 4
        growth_rate = 32
        nb_filter = 64
        nb_layers = [6, 12, 32, 32]
    elif model_size == '161':
        nb_dense_block = 4
        growth_rate = 48
        nb_filter = 96
        nb_layers = [6, 12, 36, 24]

    # 模型全连接前一层名字
    layer_name = 'relu' + str(nb_dense_block + 1) + '_blk'

    base_model = densenet_model(img_rows=img_rows, img_cols=img_cols, color_type=color_type, \
                                nb_dense_block=nb_dense_block, growth_rate=growth_rate, \
                                nb_filter=nb_filter, nb_layers=nb_layers, \
                                reduction=reduction, dropout_rate=dropout_rate, weight_decay=weight_decay, \
                                model_size=model_size)

    model = add_new_layer(base_model, layer_name, num_classes)
    setup_to_transfer_or_finetune(model, base_model, train_mode)

    return model

def parse_args():
    '''
    ### 说明：
        - 输入参数说明
    '''
    parser = argparse.ArgumentParser(description='Densenet model')

    parser.add_argument('--img_rows', dest='img_rows', help='image height', default=480, type=int)
    parser.add_argument('--img_cols', dest='img_cols', help='image width', default=480, type=int)
    parser.add_argument('--color_type', dest='color_type', help='image channel', default=3, type=int)
    parser.add_argument('--model_size', dest='model_size', help='layers of model', \
                        default='121', type=str, choices=['121', '161', '169'])
    parser.add_argument('--train_mode', dest='train_mode', help='training mode', \
                        default='finetune', type=str, choices=['finetune', 'transfer'])
    parser.add_argument('--num_classes', dest='num_classes', help='classification number', \
                        default=3, type=int)
    parser.add_argument('--use_mode', dest='use_mode', help='1-train or 0-test', \
                        default=0, type=int, choices=[0, 1])

    args = parser.parse_args()

    return args

######################################################################################
if __name__ == '__main__':

    ARGS = parse_args()

    # 生成模型
    MODEL = generate_densenet(img_rows=ARGS.img_rows, img_cols=ARGS.img_cols, color_type=ARGS.color_type, \
                              model_size=ARGS.model_size, num_classes=ARGS.num_classes, train_mode=ARGS.train_mode)
    # MODEL.summary()

    # 训练or测试
    if ARGS.use_mode:
        # 训练参数
        BATCH_SIZE = 4
        EPOCHS = 30
        # 生成数据
        TRAIN_DIR = '/home/get_samples/samples_train/train'
        VALID_DIR = '/home/get_samples/samples_train/valid'
        DATA_GEN = ImageDataGenerator(rescale=1./255, \
                                    rotation_range=15, \
                                    shear_range=10, \
                                    horizontal_flip=True, \
                                    vertical_flip=True)
        TRAIN_GEN = DATA_GEN.flow_from_directory(directory=TRAIN_DIR, \
                                                target_size=(ARGS.img_rows, ARGS.img_cols), \
                                                batch_size=BATCH_SIZE, \
                                                class_mode='categorical')
        VALID_GEN = DATA_GEN.flow_from_directory(directory=VALID_DIR, \
                                                target_size=(ARGS.img_rows, ARGS.img_cols), \
                                                batch_size=BATCH_SIZE, \
                                                class_mode='categorical')

        # start
        EATLYSTOP = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
        MODELCHECHPOINT = ModelCheckpoint('densenet_' + ARGS.model_size + '_' + ARGS.train_mode + '_weights.h5', \
                                            monitor='val_loss', \
                                            verbose=1, save_weights_only=True)
        HIST = MODEL.fit_generator(TRAIN_GEN, \
                                steps_per_epoch=6000, \
                                epochs=EPOCHS, \
                                verbose=1, \
                                callbacks=[MODELCHECHPOINT], \
                                validation_data=VALID_GEN, \
                                validation_steps=1500)
        print HIST.history

        MODEL.save_weights('densenet_' + ARGS.model_size + '_' + ARGS.train_mode + '_weights.h5')
    else:
        # 测试参数
        BATCH_SIZE = 512

        # load模型
        MODEL.load_weights('densenet_' + ARGS.model_size + '_' + ARGS.train_mode + '_weights.h5')

        # 数据路径
        DIGESTIVE_DIR = u'/home/tensorflow/digestive_change'
        CLASSIFIED_DIR = u'/home/densenet/cnn_finetune-master/densenet_results' + \
                        '_' + ARGS.model_size + '_' + str(ARGS.img_rows) + '_' + ARGS.train_mode

        # 每个数据处理
        DATA_DIRS = []
        for data_dir in os.listdir(DIGESTIVE_DIR):
            DATA_DIRS.append(os.path.join(DIGESTIVE_DIR, data_dir))

        for data_dir in DATA_DIRS[64:96]:
            print os.path.basename(data_dir)
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
                    # img = cv2.resize(img[80:400, 80:400, :], (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgs.append(img)
                imgs = np.array(imgs, dtype='float32') / 255.0
                prediction = np.argmax(MODEL.predict(np.array(imgs)), 1).tolist()
                predictions.extend(prediction)
                # print batch_idx, len(predictions)
            scio.savemat(os.path.join(CLASSIFIED_DIR, os.path.basename(data_dir) + '.mat'), {'data': predictions})
