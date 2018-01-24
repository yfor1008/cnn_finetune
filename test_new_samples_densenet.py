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
from densenet_models import parse_args, generate_densenet
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
    MODEL = generate_densenet(img_rows=ARGS.img_rows, img_cols=ARGS.img_cols, color_type=ARGS.color_type, \
                                model_size=ARGS.model_size, num_classes=ARGS.num_classes, train_mode=ARGS.train_mode)
    # load模型
    MODEL.load_weights('densenet_' + ARGS.model_size + '_' + ARGS.train_mode + '_weights.h5')
    print 'load: ' + 'densenet_' + ARGS.model_size + '_' + ARGS.train_mode + '_weights.h5'

    # 数据路径
    SRC_PATH = [u'/home/RawData/20170519/noulcer/']

    # 数据保存
    CLASSIFIED_DIR = u'/home/RawData_results_20170519_noulcer/'

    # labels
    LABELS_DIR = u'/home/get_samples/new_patient_labels/'
    LABELS_LIST = os.listdir(LABELS_DIR)
    # print LABELS_LIST

    WRONG_FILE_LIST = [u'黄剑峰5bf21e987cb57a27f45d812d17962ff9']

    for spath in SRC_PATH:

        # file_list
        # file_list = iterfindfiles(spath, '*.adi')
        FILE_LIST = os.listdir(spath)
        # print FILE_LIST

        for idx, filename in enumerate(FILE_LIST):

            if filename in WRONG_FILE_LIST:
                continue

            if os.path.exists(os.path.join(CLASSIFIED_DIR, filename + '.mat')):
                continue

            if filename + '_label.txt' in LABELS_LIST:
                with open(os.path.join(LABELS_DIR, filename + '_label.txt'), 'rb') as fid_r:
                    segs = fid_r.readline().split()
                batch_num = int(math.ceil(float(int(segs[3])) / float(BATCH_SIZE)))
                print spath, idx, filename, batch_num, segs

                predictions = []
                for batch_idx in xrange(batch_num):
                    start = batch_idx * BATCH_SIZE
                    end = min((batch_idx + 1) * BATCH_SIZE, int(segs[3]))

                    ims = []
                    for idx_img in xrange(start, end):
                        # print os.path.join(path, filename, 'data/')
                        imgs = [np.array(img.resize((480, 480)), 'uint8') \
                                for img in getPicByNum(os.path.join(spath, filename, 'data/'), idx_img, 1)]
                        im = imgs[0]
                        im[:, :, 0] = im[:, :, 0] * MASK
                        im[:, :, 1] = im[:, :, 1] * MASK
                        im[:, :, 2] = im[:, :, 2] * MASK
                        ims.append(im)
                    ims = np.array(ims, dtype='float32') / 255.0
                    prediction = np.argmax(MODEL.predict(np.array(ims)), 1).tolist()
                    predictions.extend(prediction)
                scio.savemat(os.path.join(CLASSIFIED_DIR, filename + '.mat'), {'data': predictions})
