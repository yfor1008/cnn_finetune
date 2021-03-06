# -*- coding: utf-8 -*-
"""
# resnet_152.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/7/14 上午9:23:13
# @Explanation  : finetune自己数据
"""

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, Dropout, Flatten, add, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, save_model
from keras import backend as K

from sklearn.metrics import log_loss

from custom_layers.scale_layer import Scale

# from load_cifar10 import load_cifar10_data

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

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

    x = Conv2D(nb_filter1, (1, 1), use_bias=False, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), use_bias=False, name=conv_name_base + '2b')(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), use_bias=False, name=conv_name_base + '2c')(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

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

    x = Conv2D(nb_filter1, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), use_bias=False, name=conv_name_base + '2b')(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), use_bias=False, name=conv_name_base + '2c')(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)

    return x

def resnet152_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    ### Resnet 152 Model for Keras

    	- Model Schema and layer naming follow that of the original Caffe implementation
    	- https://github.com/KaimingHe/deep-residual-networks

    ### ImageNet Pretrained Weights 
    	- Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfZHhUT3lWVWxRN28/view?usp=sharing
    	- TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing

    ### Parameters:
        - img_rows, img_cols - resolution of inputs
        - channel - 1 for grayscale, 3 for color 
        - num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    # x_fc = Flatten()(x_fc)
    # x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    x_fc = GlobalAveragePooling2D(name='avg_pool')(x)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'imagenet_models/resnet152_weights_th.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'imagenet_models/resnet152_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    # x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    # x_newfc = Flatten()(x_newfc)
    # x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)
    # x_newfc = GlobalAveragePooling2D(name='avg_pool')(x)
    # x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    # model = Model(img_input, x_newfc)

    # # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

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
    tensor_x = base_model.get_layer('res5c_relu').output

    # 增加新的层
    tensor_x = GlobalAveragePooling2D(name='avg_pool')(tensor_x)
    # tenser_x = Dense(1024, activation='relu')(tenser_x) # new FC
    predicts = Dense(nb_classes, activation='softmax')(tensor_x) # new softmax

    model = Model(inputs=base_model.input, outputs=predicts)
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

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    IMG_ROWS, IMG_COLS = 480, 480 # Resolution of inputs
    CHANNEL = 3
    NUM_CLASSES = 3
    BATCH_SIZE = 8
    EPOCHS = 30

    # 设置训练类型，transfer learning or finetune
    MODE = 'transfer'

    # 数据生成
    TRAIN_DIR = '/home/get_samples/samples_train/train'
    VALID_DIR = '/home/get_samples/samples_train/valid'
    DATA_GEN = ImageDataGenerator(rescale=1./255, \
                                  rotation_range=15, \
                                  shear_range=10, \
                                  horizontal_flip=True, \
                                  vertical_flip=True)
    TRAIN_GEN = DATA_GEN.flow_from_directory(directory=TRAIN_DIR, \
                                             target_size=(IMG_ROWS, IMG_COLS), \
                                             batch_size=BATCH_SIZE, \
                                             class_mode='categorical')
    VALID_GEN = DATA_GEN.flow_from_directory(directory=VALID_DIR, \
                                             target_size=(IMG_ROWS, IMG_COLS), \
                                             batch_size=BATCH_SIZE, \
                                             class_mode='categorical')

    # Load our model
    BASE_MODEL = resnet152_model(img_rows=IMG_ROWS, img_cols=IMG_COLS, color_type=CHANNEL, \
                                   num_classes=NUM_CLASSES)
    MODEL = add_new_layer(BASE_MODEL, NUM_CLASSES)
    setup_to_transfer_or_finetune(MODEL, BASE_MODEL, MODE)
    # MODEL.summary()

    # Start Fine-tuning
    # EATLYSTOP = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
    MODELCHECHPOINT = ModelCheckpoint('resnet_gen_' + MODE + '_weights.h5', monitor='val_loss', \
                                        verbose=1, save_weights_only=True)
    HIST = MODEL.fit_generator(TRAIN_GEN, \
                               steps_per_epoch=6000, \
                               epochs=EPOCHS, \
                               verbose=1, \
                               callbacks=[MODELCHECHPOINT], \
                               validation_data=VALID_GEN, \
                               validation_steps=1500)
    print HIST.history

    # save model
    # save_model(MODEL, 'resnet_gen_' + MODE + '.h5')
    MODEL.save_weights('resnet_gen_' + MODE + '_weights.h5')
