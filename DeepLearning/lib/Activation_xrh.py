#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

class Activation:
    """
    MLP 相关的激活函数

    Author: xrh
    Date: 2021-07-11
    """

    @staticmethod
    def tanh(X):
        """
        tanh 激活函数

        :param X:
        :return:
        """

        return np.tanh(X)

    @staticmethod
    def grad_tanh( z):
        """
        tanh 函数的一阶导数

        :param z:
        :return:
        """
        return  1- (np.tanh(z))**2

    @staticmethod
    def sigmoid( X):
        """
        sigmoid 激活函数

        :param X:
        :return:
        """
        a = 0

        try:
            a = 1 / (1 + np.exp(-X))

        except Warning as e:
            print(e)  # debug 时 , 在此处打断点

        return a

    @staticmethod
    def grad_sigmoid( z):
        """
        sigmoid 函数的一阶导数

        :param z:
        :return:
        """
        p = Activation.sigmoid(z)

        return p * (1 - p)

    @staticmethod
    def relu( z):
        """
        relu 激活函数

        :param z:
        :return:
        """
        a = np.maximum(0, z)

        return a

    @staticmethod
    def grad_relu( z):
        """
        relu 函数的一阶导数

        :param z:
        :return:
        """
        a = np.ones(np.shape(z))

        a[z <= 0] = 0

        return a

    @staticmethod
    def softmax( X):
        """
        softmax处理，将结果转化为概率

        解决了 softmax的 上溢出 和 下溢出的问题

        ref: https://www.cnblogs.com/guoyaohua/p/8900683.html

        :param X: shape (K,N) K-分类的类别 N-样本个数
        :return: shape (K,N)
        """

        X_max = np.max(X, axis=0)
        X = X - X_max

        return np.exp(X) / np.sum(np.exp(X), axis=0)  # softmax处理，将结果转化为概率

    @staticmethod
    def log_softmax(X):
        """
        对 softmax 取对数

        解决了  softmax 的上下溢出问题

        :param X: shape (K,N)
        :return:
        """

        Max = np.max(X, axis=0)  # shape (N,)

        s = np.sum(np.exp(X - Max), axis=0)  # shape (N,)

        log_s = np.log(s)  # shape (N,)

        return X - Max - log_s  # shape (K,N)