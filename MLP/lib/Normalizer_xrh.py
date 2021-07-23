#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

class Normalizer:
    """
    MLP 相关的数据标准化器(归一化)

    标准化处理对于计算距离的机器学习方法是非常重要的，因为特征的尺度不同会导致计算出来的距离倾向于尺度大的特征，
    为保证距离对每一列特征都是公平的，必须将所有特征缩放到同一尺度范围内

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

        对所有特征进行 0均值归一化, 将特征归一化为均值为0 方差为1

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

    @staticmethod
    def min_max_normalize(Xarray):
        """
        对特征进行 min-max 标准化，将数据缩放到0-1之间

        :param Xarray:
        :return:
        """

        for f in range(Xarray.shape[1]):
            maxf = np.max(Xarray[:, f])
            minf = np.min(Xarray[:, f])
            for n in range(Xarray.shape[0]):
                Xarray[n][f] = (Xarray[n][f] - minf) / (maxf - minf)
        return Xarray