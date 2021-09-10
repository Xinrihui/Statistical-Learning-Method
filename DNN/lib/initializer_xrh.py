#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np


class Initializer:
    """
    模型参数的初始化器

    class ZeroInitializer   初始化为 0
    class OneInitializer   初始化为 1
    class RandomInitializer 随机初始化
    class XavierInitializer 配合 Relu 使用的一种随机初始化

    Author: xrh
    Date: 2021-07-12

    """

    def init_W(self, layer_dim,prev_layer_dim):
        """
        初始化 W

        :param layer_dim:
        :param prev_layer_dim:
        :return:
        """
        pass

    def initialize_parameters(self, W_dims_list,*args):
        """
        初始化神经网络的参数

        :param W_dims_list: 需要初始化的所有参数矩阵的维度列表

        eg.
        W_dims_list= [ (n_h,m),(n_h, n_h)]

        W1 shape: (n_h,m)
        W2 shape: (n_h,n_h)

        :return:
            W_list

        """

        W_list = []

        for layer_dim,prev_layer_dim in W_dims_list:

            W = self.init_W(layer_dim, prev_layer_dim)

            W_list.append(W)


        return W_list


class ZeroInitializer(Initializer):

    def init_W(self, layer_dim,prev_layer_dim):
        """
        初始化 W

        :param layer_dim:
        :param prev_layer_dim:
        :return:
        """
        W = np.zeros((layer_dim, prev_layer_dim))  # 初始化为 0 会导致无法打破对称性

        return W

class OneInitializer(Initializer):

    def init_W(self, layer_dim,prev_layer_dim):
        """
        初始化 W

        :param layer_dim:
        :param prev_layer_dim:
        :return:
        """
        W = np.ones((layer_dim, prev_layer_dim))  # 初始化为 0 会导致无法打破对称性

        return W



class RandomInitializer(Initializer):

    def init_W(self, layer_dim,prev_layer_dim,scope=0.01):
        """
        初始化 W

        :param layer_dim:
        :param prev_layer_dim:
        :param scope: 参数的大小范围 ,
                     scope=0.01 参数随机选取的范围在 [-0.01,0.01]

        :return:
        """
        W = np.random.randn(layer_dim, prev_layer_dim) * scope  # 随机初始化, 生成的数值满足标准正态分布

        # 若使用 sigmoid 激活函数, 初始化的参数值不宜过大,
        # 因为会导致 sigmoid 运行在梯度很小的地方, 模型收敛速度变慢

        return W

    def initialize_parameters(self, W_dims_list,scope=0.01):
        """
        初始化神经网络的参数

        :param W_dims_list: 需要初始化的所有参数矩阵的维度列表
        :param scope: 参数的大小范围 ,
                     scope=0.01 参数随机选取的范围在 [-0.01,0.01]

        eg.
        W_dims_list= [ (n_h,m),(n_h, n_h)]

        W1 shape: (n_h,m)
        W2 shape: (n_h,n_h)

        :return:
            W_list
        """

        W_list = []

        for layer_dim,prev_layer_dim in W_dims_list:

            W = self.init_W(layer_dim, prev_layer_dim,scope)

            W_list.append(W)

        return W_list

class XavierInitializer(Initializer):

    def init_W(self, layer_dim,prev_layer_dim):
        """
        初始化 W

        :param layer_dim:
        :param prev_layer_dim:
        :return:
        """
        W = np.random.randn(layer_dim, prev_layer_dim) * np.sqrt(2. / prev_layer_dim)

        return W





