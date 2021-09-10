#!/usr/bin/python
# -*- coding: UTF-8 -*-

from cnn_layers_xrh import *
from initializer_xrh import *

from utils_xrh import *

import pickle


class MultiClassifyCNN:
    """
    基础的卷积神经网络,

    适用于构建:
    (1) Mnist 图片分类器

    模型结构为:

    0. 图片输入层
        input: shape (N,1,28,28), 28*28=784

    1. 卷积层 'conv1'
        config_conv1 = { 'f':3, 's':1, 'p':1, 'n_c':6 }
        'f' - 卷积核大小, 's' -窗口滑动步长, 'p' - padding填充的个数

                         N  C   H   W
        input :  shape ( N, 1, 28, 28)
        output : shape ( N, 6, 28, 28)

        N - 样本个数, C - 通道个数, H - 特征图的高度, W - 特征图的宽度

    2. relu 激活层 'relu1'

    3. 最大池化层 'max_pool1'
        config_pool1 = {'f':2, 's':2}
        'f' - 池化核大小, 's' -窗口滑动步长

        input :  shape (N,6,28,28)
        output : shape (N,6,14,14)

    4. 卷积层 'conv2'
    config_conv1 = { 'f':5, 's':1, 'p':0, 'n_c':16 }
    'f' - 卷积核大小, 's' -窗口滑动步长, 'p' - padding填充的个数

                     N  C   H   W
    input :  shape ( N, 6, 14, 14)
    output : shape ( N, 16, 10, 10)

    N - 样本个数, C - 通道个数, H - 特征图的高度, W - 特征图的宽度

    5. relu 激活层 'relu2'

    6. 最大池化层 'max_pool2'
        config_pool1 = {'f':2, 's':2}
        'f' - 池化核大小, 's' -窗口滑动步长

        input :  shape (N,16,10,10)
        output : shape (N,16,5,5)

    7. 全连接层 'affine1'

        input :  shape (N,16,5,5) , 15*5*5 = 400
        output : shape (N,10)

    Author: xrh
    Date: 2021-09-09

    """

    def __init__(self, picture_dim=(1, 28, 28), class_num=10,
                 dtype=np.float32,
                 model_path='model/cnn_multi_classify.model',
                 use_pre_train=True
                 ):
        """
        CNN 参数的初始化

        :param picture_dim: 输入图片的维度 (pict_c, pict_h, pict_w)
                            pict_c - 通道个数
                            pict_h - 图片高度
                            pict_w - 图片宽度

        :param class_num: 分类的个数

        :param dtype: 模型参数的数据类型, 可以配置模型的精度
                      np.float32
                      np.float64

        :param model_path: 预训练模型的路径
        :param use_pre_train: 是否使用预训练的模型
                     True: 读取预训练模型的参数后直接可以进行推理, 训练时在预训练的基础上进行训练
                     False: 从头开始训练模型
        """

        if not use_pre_train:  # 从头开始训练模型

            self.params = {}  # 模型参数(需要更新)
            self.dtype = dtype  # 模型参数的数据类型

            self.pict_c, self.pict_h, self.pict_w = picture_dim
            # pict_c - 通道个数
            # pict_h - 图片高度
            # pict_w - 图片宽度

            self.class_num = class_num  # 分类个数

            # 需要 随机初始化的参数:
            #  W_conv1 shape (n_c=6, n_c_prev=1, f=3, f=3), W_conv2 shape (n_c=16, n_c_prev=6, f=5, f=5)
            #  W_affine1 shape (class_num=10, n_k=400)

            scope = 0.01
            W_conv1 = np.random.randn(6, 1, 3, 3) * scope
            W_conv2 = np.random.randn(16, 6, 5, 5) * scope
            W_affine1 = np.random.randn(10, 400) * scope

            # 需要 0初始化的参数:
            # b_conv1 shape (n_c=6),  b_conv2 shape (n_c=16), b_affine1 shape (class_num=10,1)

            b_conv1 = np.zeros(6)
            b_conv2 = np.zeros(16)
            b_affine1 = np.zeros((10, 1))

            self.params = {"W_conv1": W_conv1, "W_conv2": W_conv2, "W_affine1": W_affine1,
                           "b_conv1": b_conv1, "b_conv2": b_conv2, "b_affine1": b_affine1
                           }

            self.cnn_layer = CNNLayer()

            # 将所有的参数转换为 dtype 类型
            for k, v in self.params.items():
                self.params[k] = v.astype(self.dtype)


        else:  # 使用预训练模型

            self.load(model_path)

    def save(self, model_dir):
        """
        保存训练好的模型

        :param train_data_dir:
        :return:
        """
        save_dict = {}

        save_dict['params'] = self.params
        save_dict['dtype'] = self.dtype

        save_dict['pict_c'] = self.pict_c
        save_dict['pict_h'] = self.pict_h
        save_dict['pict_w'] = self.pict_w

        save_dict['class_num'] = self.class_num

        save_dict['cnn_layer'] = self.cnn_layer

        with open(model_dir, 'wb') as f:
            pickle.dump(save_dict, f)

        print("Save model successful!")

    def load(self, file_path):
        """
        读取预训练的模型

        :param file_path:
        :return:
        """

        with open(file_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.params = save_dict['params']
        self.dtype = save_dict['dtype']

        self.pict_c = save_dict['pict_c']
        self.pict_h = save_dict['pict_h']
        self.pict_w = save_dict['pict_w']

        self.class_num = save_dict['class_num']

        self.cnn_layer = save_dict['cnn_layer']

        print("Load model successful!")

    def fit_batch(self, batch_picture, batch_label):
        """
        用一个批次的训练数据拟合 CNN,
        依次运行各个层的前向传播算法 和 后向传播算法, 计算损失函数, 计算模型参数的梯度

        :param batch_picture: 一批图片 shape (N, pict_c, pict_h, pict_w)
                            pict_c - 通道个数
                            pict_h - 图片高度
                            pict_w - 图片宽度

        :param batch_label: 图片的分类标签 shape (N, )

        :return: loss - 模型的损失 ;
                 grads - 模型的梯度
        """

        N, pict_c, pict_h, pict_w = np.shape(batch_picture)

        assert (pict_c, pict_h, pict_w) == (self.pict_c, self.pict_h, self.pict_w)  # 输入的训练数据必须满足初始化时的设定

        # 1. 卷积层 'conv1'

        config_conv1 = {'f': 3, 's': 1, 'p': 1, 'n_c': 6}
        out_conv1, cache_conv1 = self.cnn_layer.convolution_forward(parameters=self.params, layer_name='conv1',
                                                                    config_conv=config_conv1, a_prev=batch_picture)
        # out_conv1 shape (N, n_c, n_h, n_w)

        # 2. relu 激活层 'relu1'

        out_relu1, cache_relu1 = self.cnn_layer.relu_forward(a_prev=out_conv1)

        # 3. 最大池化层 'max_pool1'

        config_pool1 = {'f': 2, 's': 2}
        out_max_pool1, cache_max_pool1 = self.cnn_layer.max_pool_forward(layer_name='max_pool1',
                                                                         config_pool=config_pool1, a_prev=out_relu1)
        # out_max_pool1 shape (N, n_c, n_h, n_w)

        # 4. 卷积层 'conv2'

        config_conv2 = {'f': 5, 's': 1, 'p': 0, 'n_c': 16}
        out_conv2, cache_conv2 = self.cnn_layer.convolution_forward(parameters=self.params, layer_name='conv2',
                                                                    config_conv=config_conv2, a_prev=out_max_pool1)
        # out_conv2 shape (N, n_c, n_h, n_w)

        # 5. relu 激活层 'relu2'

        out_relu2, cache_relu2 = self.cnn_layer.relu_forward(a_prev=out_conv2)

        # 6. 最大池化层 'max_pool2'

        config_pool2 = {'f': 2, 's': 2}
        out_max_pool2, cache_max_pool2 = self.cnn_layer.max_pool_forward(layer_name='max_pool2',
                                                                         config_pool=config_pool2, a_prev=out_relu2)
        # out_max_pool2 shape (N, n_c, n_h, n_w)

        # 7. 全连接层 'affine1'

        z_y, y_ba, cache_affine1 = self.cnn_layer.affine_forward(parameters=self.params, layer_name='affine1',
                                                                 a_prev=out_max_pool2)
        # z_y  shape (class_num,N)
        # y_ba  shape (class_num,N)

        # 样本标签的 one-hot 化 shape (N,) -> (N,class_num)
        y_onehot = Utils.convert_to_one_hot(x=batch_label, class_num=self.class_num)  # shape (N,T-1,class_num)

        # 计算损失函数
        loss, grad_z_y = self.cnn_layer.multi_classify_loss_func(z_y=z_y, y_ba=y_ba, y_onehot=y_onehot.T)

        # 计算梯度

        # 7. 全连接层 'affine1'

        grad_a_prev_affine1, grad_dic_affine1 = self.cnn_layer.affine_backward(grad_z_y=grad_z_y, cache=cache_affine1)

        # 6. 最大池化层 'max_pool2'

        grad_a_prev_max_pool2 = self.cnn_layer.max_pool_bakward(grad_out=grad_a_prev_affine1, cache=cache_max_pool2)

        # 5. relu 激活层 'relu2'

        grad_a_prev_relu2 = self.cnn_layer.relu_backward(grad_out=grad_a_prev_max_pool2, cache=cache_relu2)

        # 4. 卷积层 'conv2'
        grad_a_prev_conv2, grad_dic_conv2 = self.cnn_layer.convolution_bakward(grad_out=grad_a_prev_relu2,
                                                                               cache=cache_conv2)

        # 3. 最大池化层 'max_pool1'

        grad_a_prev_max_pool1 = self.cnn_layer.max_pool_bakward(grad_out=grad_a_prev_conv2, cache=cache_max_pool1)

        # 2. relu 激活层 'relu1'

        grad_a_prev_relu1 = self.cnn_layer.relu_backward(grad_out=grad_a_prev_max_pool1, cache=cache_relu1)

        # 1. 卷积层 'conv1'
        _, grad_dic_conv1 = self.cnn_layer.convolution_bakward(grad_out=grad_a_prev_relu1, cache=cache_conv1)

        # 各个层梯度合并
        grads = {**grad_dic_conv1, **grad_dic_conv2, **grad_dic_affine1}

        return loss, grads

    def inference_batch(self, batch_picture):
        """
        利用训练好的模型进行推理, 输出对输入图片的分类的分值(属于某个类的概率)

        :param batch_picture: 一批图片 shape (N, pict_c, pict_h, pict_w)
                    pict_c - 通道个数
                    pict_h - 图片高度
                    pict_w - 图片宽度

        :return: y_ba - shape (class_num,N)

        """

        N, pict_c, pict_h, pict_w = np.shape(batch_picture)

        assert (pict_c, pict_h, pict_w) == (self.pict_c, self.pict_h, self.pict_w)  # 输入的训练数据必须满足初始化时的设定

        # 1. 卷积层 'conv1'

        config_conv1 = {'f': 3, 's': 1, 'p': 1, 'n_c': 6}
        out_conv1, cache_conv1 = self.cnn_layer.convolution_forward(parameters=self.params, layer_name='conv1',
                                                                    config_conv=config_conv1, a_prev=batch_picture)
        # out_conv1 shape (N, n_c, n_h, n_w)

        # 2. relu 激活层 'relu1'

        out_relu1, cache_relu1 = self.cnn_layer.relu_forward(a_prev=out_conv1)

        # 3. 最大池化层 'max_pool1'

        config_pool1 = {'f': 2, 's': 2}
        out_max_pool1, cache_max_pool1 = self.cnn_layer.max_pool_forward(layer_name='max_pool1',
                                                                         config_pool=config_pool1, a_prev=out_relu1)
        # out_max_pool1 shape (N, n_c, n_h, n_w)

        # 4. 卷积层 'conv2'

        config_conv2 = {'f': 5, 's': 1, 'p': 0, 'n_c': 16}
        out_conv2, cache_conv2 = self.cnn_layer.convolution_forward(parameters=self.params, layer_name='conv2',
                                                                    config_conv=config_conv2, a_prev=out_max_pool1)
        # out_conv2 shape (N, n_c, n_h, n_w)

        # 5. relu 激活层 'relu2'

        out_relu2, cache_relu2 = self.cnn_layer.relu_forward(a_prev=out_conv2)

        # 6. 最大池化层 'max_pool2'

        config_pool2 = {'f': 2, 's': 2}
        out_max_pool2, cache_max_pool2 = self.cnn_layer.max_pool_forward(layer_name='max_pool2',
                                                                         config_pool=config_pool2, a_prev=out_relu2)
        # out_max_pool2 shape (N, n_c, n_h, n_w)

        # 7. 全连接层 'affine1'

        z_y, y_ba, cache_affine1 = self.cnn_layer.affine_forward(parameters=self.params, layer_name='affine1',
                                                                 a_prev=out_max_pool2)
        # z_y  shape (class_num,N)
        # y_ba  shape (class_num,N)

        return y_ba

    def predict(self, batch_picture):
        """
        推理测试数据集，返回样本标签

        :param batch_picture: 一批图片 shape (N, pict_c, pict_h, pict_w)
            pict_c - 通道个数
            pict_h - 图片高度
            pict_w - 图片宽度

        :return:
        """

        P = self.inference_batch(batch_picture)  # shape (class_num,N)

        res = np.argmax(P, axis=0)  # axis=0 干掉第0个维度, shape: (N,)

        return res

class UnitTest:
    """
    单元测试

    """

    def test_lossfunc(self):
        pass


if __name__ == '__main__':

    test = UnitTest()

    test.test_lossfunc()
