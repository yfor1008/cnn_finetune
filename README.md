###  Fine-tune自己数据

#####  生成自己数据
```
python generate_my_data.py
```
数据需要格式，详见`generate_my_data.py`

#####  修改数据接口读取自己数据
```
python load_my_data.py
```

##### 训练
```
python resnet_models.py --img_rows=480 \
                        --img_cols=480 \
                        --color_type=3 \
                        --model_size=121 \
                        --train_model=finetune \
                        --num_classes=3 \
                        --use_mode=1
```
这里：
- `img_rows`和`img_cols`为训练图像大小；
- `color_type`为图像类型，3-RGB，1-灰度图像；
- `model_size`为模型大小，`resnet`模型有[101,152]可选，`densenet`有[121,161,169]可选；
- `train_model`为训练方式，finetune-对所有参数进行微调；transfer-固定全连接以前的所有参数，仅训练全连接层；
- `num_classes`为分类个数；
- `use_mode`为使用方式，1-train，训练；0-test，测试；


补充说明：

- 对于`densenet` 模型，使用方式与上面相同；
- 对于`tensorRT` 加速，可以参见 [https://github.com/yfor1008/tensorRT_for_keras](https://github.com/yfor1008/tensorRT_for_keras)

