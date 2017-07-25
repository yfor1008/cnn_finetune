#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# cifar10_read.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2017/7/14 下午3:00:41
# @Explanation  : 读取cifar10数据格式文件
"""
import cPickle

def unpickle(filename):
    '''
    ### 说明:
        - load cifar10格式文件
    ### 参数:
        - filename: str, 文件名
    ### 返回:
        - data_dict: dict, cifar10格式数据，字典形式
    '''
    with open(filename, 'rb') as fid_r:
        data_dict = cPickle.load(fid_r)
    return data_dict

if __name__ == '__main__':
    print unpickle('./my_data_set/train_list_0')
