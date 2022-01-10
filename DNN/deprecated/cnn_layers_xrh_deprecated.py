#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from deprecated import deprecated


@deprecated(version='1.0', reason="You should use another function")
class CNNLayer:

    def zero_padding(self, X, padding):
        """
        对图片进行 padding 填充, 结果填充后可以实现 same 卷积 (卷积前后图片的大小不变)

        :param X: 待 padding 的图片集合 shape (m, n_H, n_W, n_C)
                  m - 样本个数, n_H - 图片高度, n_W - 图片宽度, n_C - 通道个数

        :param padding: 填充个数

        :return: 经过 padding 后的图片
             shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """

        X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=0)

        return X_pad

    def convolution_one_step(self, parameters, layer, chanel, sub_X):
        """
        单步的卷积乘法(一个卷积核)

        :param parameters: 模型参数
        :param layer: 当前所处的层
        :param chanel: 当期的卷积核
        :param sub_X: 图像中正在被卷积计算的区域 shape (m, f, f, n_c_prev)
        :return:
        """

        W = parameters['W' + '_' + str(layer)]  # shape ( f, f, n_c_prev, n_c)
        # f - 卷积核大小,
        # n_c_prev - 卷积核的层数(图片的通道数目)
        # n_c - 卷积核的个数

        b = parameters['b' + '_' + str(layer)]  # shape (1, 1, 1, n_c)

        z = W[:, :, :, chanel] * sub_X + b[:, :, :, chanel]  # shape (m, f, f, n_c_prev)
        # W[:, :, :, chanel] shape (f, f, n_c_prev) , sub_X shape (m, f, f, n_c_prev)

        res = np.sum(z, axis=(1, 2, 3))  # 去掉 第1维 第2维 第3维: shape (m,)

        return res

    def max_pool_one_step(self, sub_X):
        """
        单步的最大池化 (一个卷积核)

        :param sub_X: 图像中正在被计算的区域 shape (m, f, f, n_c_X)
        :return:
        """

        res = np.max(sub_X, axis=(1, 2, 3))  # 去掉 第1维 第2维 第3维: shape (m,)

        return res

    def average_pool_one_step(self, sub_X):
        """
        单步的平均池化 (一个卷积核)

        :param sub_X: 图像中正在被计算的区域 shape (m, f, f, n_c_X)
        :return:
        """
        res = np.average(sub_X, axis=(1, 2, 3))  # 去掉 第1维 第2维 第3维: shape (m,)

        return res

    def convolution_forward(self, parameters, layer, config, a_prev):
        """
        卷积层的前向传播算法

        :param parameters: 模型参数
        :param layer: 当前所处的层
        :param config: 超参数
        :param a_prev: 上一层的输出 shape (m, n, n, n_c_prev), 作为本层的输入
        :return:
        """

        f = config['f' + '_' + str(layer)]  # 卷积核的大小 3
        s = config['s' + '_' + str(layer)]  # 窗口滑动步长
        p = config['p' + '_' + str(layer)]  # padding 填充的个数
        n_c = config['n_c' + '_' + str(layer)]  # 卷积核的个数(输出的通道个数)

        a_prev_pad = self.zero_padding(a_prev, p)

        m, n_h, n_w, n_c_prev = np.shape(a_prev_pad)  # m-样本个数
        # n_h-填充后图片的高度 7
        # n_w-图片的宽度
        # n_c_prev-卷积核的层数(输入的通道个数)

        n_h_out = np.floor((n_h + 2 * p - f) / s + 1)
        n_w_out = np.floor((n_w + 2 * p - f) / s + 1)

        out = np.zeros((m, n_h_out, n_w_out, n_c))  # 输出的图片

        for row in range(0, n_h + 1 - f, s):  # row=0,1,...,4
            for col in range(0, n_h + 1 - f, s):  # col=0,1,...,4
                for c in range(n_c):  # 遍历所有的卷积核

                    sub_a_prev = a_prev_pad[:, row:row + f, col:col + f, :]
                    out[:, row, col, c] = self.convolution_one_step(parameters=parameters, layer=layer, chanel=c,
                                                                    sub_X=sub_a_prev)

        cache = (parameters, a_prev_pad, layer, config)  # 缓存中间结果, 用于反向传播

        return out, cache

    def convolution_bakward(self, grad_z, cache):
        """
        卷积层的反向传播算法

        :param grad_z: 上一层传递过来的梯度 , shape (m, n_h_out, n_w_out, n_c)

        :return:
        """

        (parameters, a_prev_pad, layer, config) = cache

        f = config['f' + '_' + str(layer)]  # 卷积核的大小 3
        s = config['s' + '_' + str(layer)]  # 窗口滑动步长
        p = config['p' + '_' + str(layer)]  # padding 填充的个数

        grad_z_pad = self.zero_padding(grad_z, p)

        m, n_h, n_w, n_c_prev = np.shape(a_prev_pad)  # m-样本个数
        # n_h-填充后图片的高度 7
        # n_w-图片的宽度
        # n_c_prev-卷积核的层数(输入的通道个数)

        W = parameters['W' + '_' + str(layer)]  # shape ( f, f, n_c_prev, n_c)
        # n_c - 卷积核的个数
        # f - 卷积核大小,
        # n_c_X - 卷积核的层数(图片的通道数目)



    def max_pool_forward(self, layer, config, a_prev):
        """
        最大池化的前向传播算法

        :param layer: 当前所处的层
        :param config: 超参数
        :param a_prev: 上一层的输出 shape (m,n,n,n_c_X), 作为本层的输入
        :return:
        """

        f = config['f' + '_' + str(layer)]  # 卷积核的大小 3
        s = config['s' + '_' + str(layer)]  # 窗口滑动步长
        n_c = config['n_c' + '_' + str(layer)]  # 卷积核的个数(输出的通道个数)

        m, n_h, n_w, n_c_X = np.shape(a_prev)  # m-样本个数
        # n_h-填充后图片的高度 7
        # n_w-图片的宽度
        # n_c_X-卷积核的层数(图片的通道数目)

        n_h_out = np.floor((n_h - f) / s + 1)
        n_w_out = np.floor((n_w - f) / s + 1)

        out = np.zeros((m, n_h_out, n_w_out, n_c))  # 输出的图片

        for row in range(0, n_h + 1 - f, s):  # row=0,1,...,4
            for col in range(0, n_h + 1 - f, s):  # col=0,1,...,4
                for c in range(n_c):  # 遍历所有的卷积核

                    sub_a_prev = a_prev[:, row:row + f, col:col + f, :]
                    out[:, row, col, c] = self.max_pool_one_step(sub_X=sub_a_prev)

        cache = (a_prev)  # 缓存中间结果, 为了反向传播

        return out, cache

    def average_pool_forward(self, layer, config, a_prev):
        """
        最大池化的前向传播算法

        :param layer: 当前所处的层
        :param config: 超参数
        :param a_prev: 上一层的输出 shape (m,n,n,n_c_X), 作为本层的输入
        :return:
        """

        f = config['f' + '_' + str(layer)]  # 卷积核的大小 3
        s = config['s' + '_' + str(layer)]  # 窗口滑动步长
        n_c = config['n_c' + '_' + str(layer)]  # 卷积核的个数(输出的通道个数)

        m, n_h, n_w, n_c_X = np.shape(a_prev)  # m-样本个数
        # n_h-填充后图片的高度 7
        # n_w-图片的宽度
        # n_c_X-卷积核的层数(图片的通道数目)

        n_h_out = np.floor((n_h - f) / s + 1)
        n_w_out = np.floor((n_w - f) / s + 1)

        out = np.zeros((m, n_h_out, n_w_out, n_c))  # 输出的图片

        for row in range(0, n_h + 1 - f, s):  # row=0,1,...,4
            for col in range(0, n_h + 1 - f, s):  # col=0,1,...,4
                for c in range(n_c):  # 遍历所有的卷积核

                    sub_a_prev = a_prev[:, row:row + f, col:col + f, :]
                    out[:, row, col, c] = self.average_pool_one_step(sub_X=sub_a_prev)

        cache = (a_prev)  # 缓存中间结果, 为了反向传播

        return out, cache
