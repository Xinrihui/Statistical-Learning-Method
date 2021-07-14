#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np

class Normalize:
    """
    MLP 相关的模型参数初始化器

    Author: xrh
    Date: 2021-07-10

    """

    @staticmethod
    def tow_norm_normalize(X):
        """
        对所有样本进行二范数归一化

        ref:
        https://blog.csdn.net/hqh131360239/article/details/79061535

        :param X: shape(N,m)
                  N - 样本的个数
                  m - 样本的维度
        :return:
        """

        X_norm_2 = np.linalg.norm(X, ord=2, axis=1, keepdims=True)  # 每一个样本的模(二范数) shape(N,1)

        # X_norm_2 = X_norm_2.reshape(-1,1)

        X = X / X_norm_2

        return X


    @staticmethod
    def Z_Score_normalize(X):
        """
        0均值归一化( Z-Score Normalization)

        对所有特征进行 0均值归一化

        :param X: shape(N,m)
                  N - 样本的个数
                  m - 样本的维度
        :return:
        """

        # N, m = np.shape(X)  # N 个样本, m 个特征

        mu = np.mean(X, axis=0)  # 每一个特征的均值 shape:(m,)

        s = np.std(X, axis=0)  # 每一个特征的标准差 shape:(m,)

        X = (X - mu) / s

        return X
