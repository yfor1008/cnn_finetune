#!/usr/bin/env python
# -*- coding: utf-8 -*-

import struct
import os
import numpy as np
import sys
import types
import logging
from PIL import Image
from PIL import ImageFile
from StringIO import StringIO
ImageFile.LOAD_TRUNCATED_IMAGES = True


def data2img(file_path,save_path):
    """
    从adv文件中读取图片数据，并将图片数据存到硬盘上
    
    input:  file_path -- adv文件的路径（包括文件名）
            save_path -- 图片保存的位置
    """
    try:
        fdata=open(file_path,'rb')
        fdata.seek(6)               #偏移6个字节 
        while True:
            tem_buf=fdata.read(8)
            if len(tem_buf)<8:
                break
            img_size,img_num=struct.unpack('2i',tem_buf)  #读取8个字节，图像大小和图像编号
            fdata.seek(58,1)       #从当前位置偏移58个字节
            buf=fdata.read(img_size-62)  #读取图像数据
            file_name = file_path[-5:-4] + '_' + '%06d'%img_num #命名加上.adv文件的编号
            img=file(save_path+file_name+'.jpg','wb')
            img.write(buf)
            img.flush()
            img.close()
    finally:
        fdata.close()


#==============================================================================
# def getOnePic(adv_path, offset = 0):
#     """
#     从adv文件中读取图片数据，返回numpy array格式的图片数据
#     
#     input：   adv_path  -- adv文件的路径
#               offset    -- 图片数据的偏移量
#     output：  imArray   -- numpy array格式的图片数据
#               imgNum    -- 图片编号
#               newOffset -- 下一幅图片的偏移量
#     """
#     try:
#         fdata = open(adv_path, 'rb')
#         if offset == 0:
#             fdata.seek(6)
#         else:
#             fdata.seek(offset)
#         temBuf = fdata.read(8)
#         if len(temBuf) == 8:
#             imgSize,imgNum=struct.unpack('2i',temBuf)  #读取8个字节，图像大小和图像编号
#             fdata.seek(58,1)       #从当前位置偏移58个字节
#             buf=StringIO(fdata.read(imgSize-62))  #读取图像数据
#             im = Image.open(buf)
#             imArray = np.array(im)  # 将PIL image 转换为numpy array
#             newOffset = fdata.tell()
#         else:     #读取到最后一张时返回newOffset为负
#             imArray = []
#             imgNum = -1
#             newOffset = -1            
#         return imArray, imgNum, newOffset
#     finally:
#         fdata.close()
# 
# def getAdvPath(data_path):
#     """
#     输入data目录的路径，输出adv文件的路径
#     
#     input：     data_path -- data目录的路径
#     output      adv_path  -- adv文件的路径
#     """
#     list_file = os.listdir(data_path)
#     adv_path = []
#     for line in list_file:
#         if line[-4:] == '.adv':    #查找以.adv结尾的文件
#             adv_path.append(data_path + line)
#     return adv_path
# 
# 
#==============================================================================

def getNumSum(dataPath):
    """
    根据adi文件的大小，获得图像总数
    input:    dataPath      -- data目录的路径
    output:   picNumSum     -- 图像总数
    """
    listFile = os.listdir(dataPath)
    picNumSum = 0
    offset = 5  # ankon V4版本需修改此处的5为6
    for line in listFile:
        if line.endswith('.adi'):    #查找以.adi结尾的文件
            adiPath = dataPath + line
            adiSize = os.path.getsize(adiPath)
            with open(adiPath, 'rb') as adi_fdata:
                adi_fdata.seek(5)
                temBuf = adi_fdata.read(1)
                if len(temBuf) == 1:
                    version, =struct.unpack('c',temBuf)
                    if version == 'k':        #ankon(V1)/ankom(V2)/ankol(V3)/ankok(V4)
                        offset = 6
            picNumSum = (adiSize - 6) // offset   
    return picNumSum
    
def getPicByNum(dataPath, startIndex = 0, outputPicNum = 1):
    """
    根据adi文件的索引，从adv文件中读出图片数据
    input:    dataPath      -- data目录的路径
              startIndex    -- 起始索引(从0开始)（需为非负）
              outputPicNum  -- 输出的图片数(需为正整数，最大为2000张)
    output:   outArr        -- 图片数据的dict，图片数据是numpy的array格式
    """

    if (startIndex < 0) or (type(startIndex) != types.IntType):
        raise ValueError("Input 'startIndex' must be an positive integer")
    if ((outputPicNum <= 0) or (outputPicNum > 2000) 
                            or (type(startIndex) != types.IntType)):
        raise ValueError("Input 'outputPicNum' must be an positive integer \
                          and in range[1,2000]")
    
    listFile = os.listdir(dataPath)
    outArr = []
    advPath = []
    adi_offset = 5   # ankon V4版本需修改此处的5为6
    adv_offset = 58  # ankon V1版本需修改此处的58为8
    for line in listFile:
        if line.endswith('.adi'):    #查找以.adi结尾的文件
            adiPath = dataPath + line
        if line.endswith('.adv'):    #查找以.adv结尾的文件
            advPath.append(dataPath + line)
        
    try:
        advData = []
        advPath.sort(key=lambda x:x[-5]) #根据数据文件的序号进行排序
        for pa in advPath:
            advData.append(open(pa, 'rb'))
        adi_fdata = open(adiPath, 'rb')
        adi_fdata.seek(5)
        temBuf = adi_fdata.read(1)
        if len(temBuf) == 1:
            version, =struct.unpack('c',temBuf)
            if version == 'k':        #ankon(V1)/ankom(V2)/ankol(V3)/ankok(V4)
                adi_offset = 6
            if version == 'n':
                adv_offset = 8
        adi_fdata.seek(6 + startIndex * adi_offset)  # adi文件偏移
        for ind in range(outputPicNum):
            #print startIndex + ind            
            temBuf = adi_fdata.read(1)
            if len(temBuf) == 1:
                advIndex, =struct.unpack('B',temBuf)
            else:
                break
            temBuf = adi_fdata.read(4)
            if len(temBuf) == 4:
                offset, = struct.unpack('i',temBuf)
            else:
                break
            if adi_offset == 6:
                temBuf = adi_fdata.read(1)
            # 根据图像序列的序号和偏移字节读数据            
            fdata = advData[advIndex]
            fdata.seek(offset)
            temBuf = fdata.read(8)
            if len(temBuf) == 8:
                imgSize,imgNum=struct.unpack('2i',temBuf)  #读取8个字节，图像大小和图像编号
                fdata.seek(adv_offset,1)       #从当前位置偏移adv_offset个字节
                buf=StringIO(fdata.read(imgSize-(adv_offset+4)))  #读取图像数据                
                try:
                    im =  Image.open(buf) 
                    outArr.append(im.resize((480,480)))
                except Exception,e:
                    logging.error(e)
                    logging.error(startIndex + ind)
                    outArr.append(Image.new("RGB",(480,480),(0,0,0)))
            else:
                break
        return outArr
    except Exception,e:
        logging.error(e)
    finally:
        for item in advData:
            item.close()
        adi_fdata.close()

#==============================================================================
# path = r'D:/zhang(2014090502)_13091000692_20140905/data/'
# arr1 = getPicByNum(path,1,2000)
# i = 1
# for item in arr1:
#     img11=Image.fromarray(item)  
#     img11.save(path + str(i) +  ".jpg")
#     i = i+1
# print 'done' 
# print getNumSum(path)
#==============================================================================

if __name__ == '__main__': 
    if len(sys.argv) < 2:
        os._exit() 
    else:
        arr1 = getPicByNum(sys.argv[1],1,2000)
        print ("done!")

