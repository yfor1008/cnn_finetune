#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# test_new_samples.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/9/5 下午5:12:29
# @Explanation  : 测试新的病例数据
"""

import os
import math
from resnet_models import parse_args, generate_resnet
from adv2jpg import getPicByNum
import numpy as np
import scipy.io as scio
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# def iterfindfiles(path, fnexp):
#     for root, dirs, files in os.walk(path):
#         for filename in fnmatch.filter(files, fnexp):
#             yield os.path.join(root)

if __name__ == '__main__':

    ARGS = parse_args()

    # mask
    MASK = Image.open(os.path.join(os.path.dirname(__file__), 'mask.png'))
    MASK = np.array(MASK, 'uint8') > 0

    # 测试参数
    BATCH_SIZE = 512
    # 生成模型
    MODEL = generate_resnet(img_rows=ARGS.img_rows, img_cols=ARGS.img_cols, color_type=ARGS.color_type, \
                            model_size=ARGS.model_size, num_classes=ARGS.num_classes, train_mode=ARGS.train_mode)
    # load模型
    MODEL.load_weights('resnet_' + ARGS.model_size + '_' + ARGS.train_mode + '_weights.h5')
    print 'load: ' + 'resnet_' + ARGS.model_size + '_' + ARGS.train_mode + '_weights.h5'

    # 数据路径
    SRC_PATH = u'./test/jiyinsheng'

    FILE_LIST = os.listdir(SRC_PATH)
    FILE_LIST.sort()
    # print FILE_LIST

    batch_num = int(math.ceil(float(len(FILE_LIST))/float(BATCH_SIZE)))

    predictions = []
    for batch_idx in xrange(batch_num):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, len(FILE_LIST))

        ims = []
        for idx_img in xrange(start, end):
            # print os.path.join(path, filename, 'data/')
            imgs = Image.open(os.path.join(SRC_PATH, FILE_LIST[idx_img]))
            # print idx_img, len(imgs)
            im = np.array(imgs)
            im[:, :, 0] = im[:, :, 0] * MASK
            im[:, :, 1] = im[:, :, 1] * MASK
            im[:, :, 2] = im[:, :, 2] * MASK
            ims.append(im)
        ims = np.array(ims, dtype='float32') / 255.0
        temp_pre = [np.where(datalist > 0.8)[0].tolist() for datalist in MODEL.predict(np.array(ims))]
        prediction = []
        for loc in temp_pre:
            if loc:
                prediction.append(loc[0])
            else:
                prediction.append(-1)
        # prediction = np.argmax(MODEL.predict(np.array(ims)), 1).tolist()
        predictions.extend(prediction)
    scio.savemat(os.path.join(os.path.basename(SRC_PATH) + '.mat'), {'data': predictions})

    # for idx, filename in enumerate(FILE_LIST):
    #     ims = []
    #     im = Image.open(os.path.join(SRC_PATH, filename))
    #     im = np.array(im)
    #     im[:, :, 0] = im[:, :, 0] * MASK
    #     im[:, :, 1] = im[:, :, 1] * MASK
    #     im[:, :, 2] = im[:, :, 2] * MASK
    #     ims.append(im)
    #     # ims.append(im)
    #     ims = np.array(ims, dtype='float32') / 255.0
    #     # prediction = np.argmax(MODEL.predict(np.array(ims)), 1).tolist()
    #     # print filename, MODEL.predict(np.array(ims)), np.argmax(MODEL.predict(np.array(ims)), 1)
    #     # print filename, ['%.4f' % pre for pre in MODEL.predict(np.array(ims))[0]], np.argmax(MODEL.predict(np.array(ims)), 1), \
    #     #         [np.where(datalist > 0.8)[0].tolist() for datalist in MODEL.predict(np.array(ims))]

