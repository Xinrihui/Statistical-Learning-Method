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

    pass

class ZeroInitializer(Initializer):

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

        for i in range(0, len(layers_dims) - 1):  # layers_dims: [m,h,K]

            W = np.zeros((layers_dims[i + 1], layers_dims[i]))  # 初始化为 0 会导致无法打破对称性

            # i=0 W1 (n_h, n_x)
            # i=1 W2 (n_y, n_h)

            b = np.zeros((layers_dims[i + 1], 1))
            # i=0 b1 (n_h, 1)
            # i=1 b2 (n_y, 1)

            W_list.append(W)
            b_list.append(b)

        parameters = {}

        parameters['W'] = W_list
        parameters['b'] = b_list

        return parameters


class RandomInitializer(Initializer):

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

        for i in range(0, len(layers_dims) - 1):  # layers_dims: [m,h,K]

            W = np.random.randn(layers_dims[i + 1], layers_dims[i]) * 0.01  # 随机初始化, 生成的数值满足标准正态分布
            # 若使用 sigmoid 激活函数, 初始化的参数值不宜过大,
            # 因为会导致 sigmoid 运行在梯度很小的地方, 模型收敛速度变慢

            # i=0 W1 (n_h, n_x)
            # i=1 W2 (n_y, n_h)

            b = np.zeros((layers_dims[i + 1], 1))
            # i=0 b1 (n_h, 1)
            # i=1 b2 (n_y, 1)

            W_list.append(W)
            b_list.append(b)

        parameters = {}

        parameters['W'] = W_list
        parameters['b'] = b_list

        return parameters


class XavierInitializer(Initializer):

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

        for i in range(0, len(layers_dims) - 1):  # layers_dims: [m,h,K]

            W = np.random.randn(layers_dims[i + 1], layers_dims[i]) * np.sqrt(2. / layers_dims[i])

            # i=0 W1 (n_h, n_x)
            # i=1 W2 (n_y, n_h)

            b = np.zeros((layers_dims[i + 1], 1))
            # i=0 b1 (n_h, 1)
            # i=1 b2 (n_y, 1)

            W_list.append(W)
            b_list.append(b)

        parameters = {}

        parameters['W'] = W_list
        parameters['b'] = b_list

        return parameters



