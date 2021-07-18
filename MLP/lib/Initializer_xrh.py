#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np


class Initializer:
    """
    模型参数的初始化器

    class ZeroInitializer   初始化为 0
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

    def initialize_parameters(self, layers_dims):
        """
        初始化 MLP 的参数

        :param layers_dims: 从前往后 MLP各个层的向量维度

        eg.
        layers_dims= [m,h,K]
        m: 输入层向量的维度
        h: 隐藏层向量的维度
        K: 输出层向量的维度


        :return:
        parameters:
            W0 -- weight matrix of shape (h, m)
            b0 -- bias vector of shape (h, 1)
            W1 -- weight matrix of shape (K, h)
            b1 -- bias vector of shape (K, 1)
        """

        W_list = []
        b_list = []

        gama_list = []
        beta_list = []

        for i in range(0, len(layers_dims) - 1):  # layers_dims: [m,h,K]

            W = self.init_W(layers_dims[i + 1], layers_dims[i])

            # i=0 W1 (n_h, n_x)
            # i=1 W2 (n_y, n_h)

            b = np.zeros((layers_dims[i + 1], 1))
            # i=0 b1 (n_h, 1)
            # i=1 b2 (n_y, 1)

            # batchnorm 的参数初始化
            gama = np.ones((layers_dims[i + 1], 1)) # 方差初始化为1
            beta = np.zeros((layers_dims[i + 1], 1)) # 均值初始化为0

            W_list.append(W)
            b_list.append(b)

            gama_list.append(gama)
            beta_list.append(beta)


        parameters = {}

        parameters['W'] = W_list
        parameters['b'] = b_list

        parameters['gama'] = gama_list
        parameters['beta'] = beta_list

        return parameters


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

class RandomInitializer(Initializer):

    def init_W(self, layer_dim,prev_layer_dim):
        """
        初始化 W

        :param layer_dim:
        :param prev_layer_dim:
        :return:
        """
        W = np.random.randn(layer_dim, prev_layer_dim) * 0.01  # 随机初始化, 生成的数值满足标准正态分布
        # 若使用 sigmoid 激活函数, 初始化的参数值不宜过大,
        # 因为会导致 sigmoid 运行在梯度很小的地方, 模型收敛速度变慢

        return W



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





