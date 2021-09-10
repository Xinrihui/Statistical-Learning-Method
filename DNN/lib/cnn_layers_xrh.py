#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from deprecated import deprecated

from activation_xrh import *

from gradient_check_xrh import *


class CNNLayer:
    """
    实现 CNN 的相关层

    Author: xrh
    Date: 2021-09-01

    ref:
    https://zhuanlan.zhihu.com/p/81675803

    """

    def zero_padding(self, X, padding):
        """
        对图片进行 padding 填充, 结果填充后可以实现 same 卷积 (卷积前后图片的大小不变)

        :param X: 待 padding 的图片集合 shape (N, n_c, h, w)
                  N - 样本个数, n_c - 通道个数, h - 图片高度, w - 图片宽度

        :param padding: 填充个数

        :return: 经过 padding 后的图片
             shape (N, n_c, h + 2*pad, w + 2*pad)
        """

        X_pad = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)

        return X_pad

    def convolution_forward(self, parameters, layer_name, config_conv, a_prev):
        """
        卷积层的前向传播算法

        :param parameters: 模型参数
        :param layer_name: 当前层的名字
        :param config_conv: 卷积层相关的超参数
        :param a_prev: 上一层的输出作为本层的输入 shape (N, n_c_prev, h, w)
                                  N - 样本个数
                                  c - 卷积核的层数(输入的通道个数)
                                  h - 输入图片的高度
                                  w - 输入图片的宽度

        :return: out - shape (N, n_c, n_h, n_w)
                 cache - (parameters, layer_name, config_conv, W_ba, a_prev, a_prev_pad, a_prev_pad_ba)

        """

        f = config_conv['f']  # 卷积核的大小
        s = config_conv['s']  # 窗口滑动步长
        p = config_conv['p']  # padding 填充的个数
        n_c = config_conv['n_c']  # 卷积核的个数(输出的通道个数)

        N, c, h, w = np.shape(a_prev)  # N-样本个数
        # c - 卷积核的层数(输入的通道个数)
        # h - 输入图片的高度
        # w - 输入图片的宽度

        # 输出特征图的高度和宽度
        n_h = int((h + 2 * p - f) / s + 1)
        n_w = int((w + 2 * p - f) / s + 1)

        a_prev_pad = self.zero_padding(a_prev, padding=p)
        # a_prev_pad shape (N, n_c_prev, n_h_prev, n_w_prev)

        N, n_c_prev, n_h_prev, n_w_prev = np.shape(a_prev_pad)  # N-样本个数
        # n_c_prev - 卷积核的层数(输入的通道个数)
        # n_h_prev - 填充后的图片的高度
        # n_w_prev - 填充后的图片的宽度

        W = parameters['W' + '_' + str(layer_name)]  # shape (n_c, n_c_prev, f, f)
        # n_c - 卷积核的个数(这一层输出的通道个数)
        # n_c_prev - 单个卷积核的层数(上一层输出的通道数目)
        # f - 卷积核大小

        b = parameters['b' + '_' + str(layer_name)]  # shape (n_c,)

        W_ba = W.reshape((n_c, n_c_prev * f * f)).T  # shape (n_c_prev*f*f, n_c)
        a_prev_pad_ba = np.zeros((N, n_h * n_w, n_c_prev * f * f))  # shape (N, n_h*n_w, n_c_prev*f*f)

        idx = 0
        for row in range(0, n_h_prev + 1 - f, s):  # row=0,1,...,4
            for col in range(0, n_w_prev + 1 - f, s):  # col=0,1,...,4

                sub_a_prev_pad = a_prev_pad[:, :, row:row + f, col:col + f]  # shape (N, n_c_prev, f, f)
                sub_a_prev_pad = np.reshape(sub_a_prev_pad, (N, n_c_prev * f * f))  # shape (N, n_c_prev*f*f)

                a_prev_pad_ba[:, idx, :] = sub_a_prev_pad

                idx += 1

        y_ba = np.dot(a_prev_pad_ba, W_ba) + b  # shape (N, n_h*n_w, n_c)
        # a_prev_pad_ba shape (N, n_h*n_w, n_c_prev*f*f) , W_ba shape (n_c_prev*f*f, n_c)

        y_ba_trans = np.transpose(y_ba, (0, 2, 1))  # shape (N, n_c, n_h*n_w)

        out = np.reshape(y_ba_trans, (N, n_c, n_h, n_w))

        cache = (parameters, layer_name, config_conv, W_ba, a_prev, a_prev_pad, a_prev_pad_ba)  # 缓存中间结果, 用于反向传播

        return out, cache

    def convolution_bakward(self, grad_out, cache):
        """
        卷积层的反向传播算法

        :param grad_out: 上一层传递过来的梯度 , shape (N, n_c, n_h, n_w)
        :param cache: 前向传播的缓存

        :return: grad_a_prev - shape (N, n_c_prev, h, w)
                 grad_dic - 需要更新的模型的参数的梯度
        """

        (parameters, layer_name, config_conv, W_ba, a_prev, a_prev_pad, a_prev_pad_ba) = cache

        f = config_conv['f']  # 卷积核的大小
        s = config_conv['s']  # 窗口滑动步长
        p = config_conv['p']  # padding 填充的个数
        n_c = config_conv['n_c']  # 卷积核的个数(输出的通道个数)

        N, n_c_prev, n_h_prev, n_w_prev = np.shape(a_prev_pad)  # N-样本个数
        # n_c_prev-卷积核的层数(输入的通道个数)
        # n_h_prev-填充后图片的高度
        # n_w_prev-填充后图片的宽度

        N, c, h, w = np.shape(a_prev)  # N-样本个数
        # c - 卷积核的层数(输入的通道个数)
        # h - 输入图片的高度
        # w - 输入图片的宽度

        # 输出特征图的高度和宽度
        n_h = int((h + 2 * p - f) / s + 1)
        n_w = int((w + 2 * p - f) / s + 1)

        # grad_out shape (N, n_c, n_h, n_w)
        grad_y_ba_tran = grad_out.reshape(N, n_c, n_h * n_w).transpose(0, 2, 1). \
            reshape(N * n_h * n_w, n_c)  # shape (N*n_h*n_w, n_c)

        # a_prev_pad_ba shape (N, n_h*n_w, n_c_prev*f*f)
        a_prev_pad_ba_tran = a_prev_pad_ba.reshape(N * n_h * n_w, n_c_prev * f * f).T  # shape (n_c_prev*f*f, N*n_h*n_w)

        # W_ba shape (n_c_prev*f*f, n_c)
        grad_W_ba = np.dot(a_prev_pad_ba_tran, grad_y_ba_tran)  # shape (n_c_prev*f*f, n_c)
        #  shape (n_c_prev*f*f, N*n_h*n_w), shape (N*n_h*n_w, n_c) -> shape (n_c_prev*f*f, n_c)

        # W shape (n_c, n_c_prev, f, f)
        grad_W = grad_W_ba.T.reshape(n_c, n_c_prev, f, f)

        # b shape (n_c,)
        grad_b = np.sum(grad_y_ba_tran, axis=0)  # shape (n_c,)

        # a_prev_pad_ba shape  (N, n_h*n_w, n_c_prev*f*f)
        grad_a_prev_pad_ba = np.dot(grad_y_ba_tran, W_ba.T).reshape((N, n_h * n_w, n_c_prev * f * f))
        # shape (N*n_h*n_w, n_c) , shape (n_c, n_c_prev*f*f) -> shape (N*n_h*n_w, n_c_prev*f*f)

        # 将 grad_a_prev_pad_ba 还原为 grad_a_prev, 以便传递给上一层

        # a_prev_pad shape (N, n_c_prev, n_h_prev, n_w_prev)
        grad_a_prev_pad = np.zeros((N, n_c_prev, n_h_prev, n_w_prev))  # shape (N, n_c_prev, n_h_prev, n_w_prev)

        idx = 0
        for row in range(0, n_h_prev + 1 - f, s):
            for col in range(0, n_w_prev + 1 - f, s):

                arr_row = grad_a_prev_pad_ba[:, idx, :]  # shape  (N, n_c_prev*f*f)
                grad_a_prev_pad[:, :, row:row + f, col:col + f] += arr_row.reshape(N, n_c_prev, f, f)  # TODO: 为啥是累加 ==

                idx += 1

        # 对 grad_a_prev_pad 做剪裁，即去掉 padding 填充的0
        #  a_prev shape (N, n_c_prev, h, w)

        if p != 0:
            grad_a_prev = grad_a_prev_pad[:, :, p:-p, p:-p]

        else:
            grad_a_prev = grad_a_prev_pad

        # 将需要更新的模型参数的梯度包装为 dict
        grad_dic = {"grad_W_" + str(layer_name): grad_W, "grad_b_" + str(layer_name): grad_b
                    }

        return grad_a_prev, grad_dic

    def max_pool_forward(self, layer_name, config_pool, a_prev):
        """
        最大池化的前向传播算法

        :param layer_name: 当前层的名字
        :param config_pool: 池化层相关的超参数
        :param a_prev: 上一层的输出作为本层的输入 shape (N, n_c, h, w)
                                  N-样本个数
                                  n_c-输入通道的个数
                                  h-图片的高度
                                  w-图片的宽度

        :return: out - shape (N, n_c, n_h, n_w)
                 cache - (layer_name, config_pool, a_prev, a_prev_ba)
        """

        f = config_pool['f']  # 池化核的大小
        s = config_pool['s']  # 窗口滑动步长

        N, n_c, h, w = np.shape(a_prev)  # N-样本个数
        # n_c-在池化层中我们不改变通道的数目, 即输入的通道数目和输出的相同
        # h-图片的高度
        # w-图片的宽度

        # 在池化层中我们不会加 padding 填充
        # 输出特征图的高度和宽度
        n_h = int((h - f) / s + 1)
        n_w = int((w - f) / s + 1)

        a_prev_ba = np.zeros((N, n_c, n_h * n_w, f * f))  # shape (N, n_c, n_h*n_w, f * f)

        idx = 0
        for row in range(0, h + 1 - f, s):
            for col in range(0, w + 1 - f, s):

                sub_a_prev = a_prev[:, :, row:row + f, col:col + f]  # shape (N, n_c, f, f)
                sub_a_prev_tran = sub_a_prev.reshape((N, n_c, f * f))  # shape (N, n_c, f*f)
                a_prev_ba[:, :, idx, :] = sub_a_prev_tran

                idx += 1

        # a_prev_ba shape (N, n_c, n_h*n_w, f * f)
        y_ba = np.max(a_prev_ba, axis=3)  # shape (N, n_c, n_h*n_w)

        out = y_ba.reshape((N, n_c, n_h, n_w))  # 输出的特征图

        cache = (layer_name, config_pool, a_prev, a_prev_ba)  # 缓存中间结果, 为了反向传播

        return out, cache

    def max_pool_bakward(self, grad_out, cache):
        """
        最大池化层的反向传播算法

        :param grad_out: 上一层传递过来的梯度 , shape (N, n_c, n_h, n_w)
        :param cache: 前向传播的缓存

        :return: grad_a_prev - shape (N, n_c, h, w)

        """

        (layer_name, config_pool, a_prev, a_prev_ba) = cache

        f = config_pool['f']  # 池化核的大小
        s = config_pool['s']  # 窗口滑动步长

        N, n_c, h, w = np.shape(a_prev)  # N-样本个数
        # n_c-在池化层中我们不改变通道的数目, 即输入的通道数目和输出的相同
        # h-图片的高度
        # w-图片的宽度

        # 在池化层中我们不会加 padding 填充
        # 输出特征图的高度和宽度
        n_h = int((h - f) / s + 1)
        n_w = int((w - f) / s + 1)

        # y_ba shape (N, n_c, n_h*n_w)
        grad_y_ba = grad_out.reshape((N, n_c, n_h * n_w))  # shape (N, n_c, n_h*n_w)


        # a_prev_ba shape (N, n_c, n_h*n_w, f * f)
        grad_a_prev_ba = np.zeros((N, n_c, n_h * n_w, f * f))  # shape (N, n_c, n_h*n_w, f * f)

        max_idx = np.argmax(a_prev_ba, axis=3)  # a_prev_ba 的每一行中的最大元素的标号 shape (N, n_c, n_h*n_w)

        # 把数组拍平
        n_flat = N * n_c * n_h * n_w
        max_idx_flat = max_idx.reshape(n_flat)
        grad_a_prev_ba_flat = grad_a_prev_ba.reshape(n_flat, f * f)
        grad_y_ba_flat = grad_y_ba.reshape(n_flat)

        # 利用最大值对应的标号来赋值
        grad_a_prev_ba_flat[np.arange(N * n_c * n_h * n_w), max_idx_flat] = grad_y_ba_flat
        # np.arange(N*n_c*n_h*n_w) 为行索引, max_idx_flat 为列索引

        # 把拍平后的数组还原
        grad_a_prev_ba = grad_a_prev_ba_flat.reshape((N, n_c, n_h * n_w, f * f))

        # 将 grad_a_prev_ba 还原为 grad_a_prev, 以便传递给上一层

        # a_prev shape (N, n_c, h, w)
        grad_a_prev = np.zeros((N, n_c, h, w))  # shape (N, n_c, h, w)

        idx = 0
        for row in range(0, h + 1 - f, s):
            for col in range(0, w + 1 - f, s):
                grad_a_prev[:, :, row:row + f, col:col + f] += grad_a_prev_ba[:, :, idx, :].reshape((N, n_c, f, f))
                idx += 1

        return grad_a_prev

    def average_pool_forward(self, layer_name, config_pool, a_prev):
        """
        平均池化层的前向传播算法

        :param layer_name: 当前层的名字
        :param config_pool: 池化层相关的超参数
        :param a_prev: 上一层的输出作为本层的输入 shape (N, n_c, h, w)
                                  N-样本个数
                                  n_c-输入通道的个数
                                  h-图片的高度
                                  w-图片的宽度

        :return: out - shape (N, n_c, n_h, n_w)
                 cache - (layer_name, config_pool, a_prev, a_prev_ba, W_ba)

        """

        f = config_pool['f']  # 池化核的大小
        s = config_pool['s']  # 窗口滑动步长

        N, n_c, h, w = np.shape(a_prev)  # N-样本个数
        # n_c-在池化层中我们不改变通道的数目, 即输入的通道数目和输出的相同
        # h-图片的高度
        # w-图片的宽度

        # 在池化层中我们不会加 padding 填充
        # 输出特征图的高度和宽度
        n_h = int((h - f) / s + 1)
        n_w = int((w - f) / s + 1)

        a_prev_ba = np.zeros((N, n_c, n_h * n_w, f * f))  # shape (N, n_c, n_h*n_w, f * f)

        idx = 0
        for row in range(0, h + 1 - f, s):
            for col in range(0, w + 1 - f, s):

                sub_a_prev = a_prev[:, :, row:row + f, col:col + f]  # shape (N, n_c, f, f)
                sub_a_prev_tran = sub_a_prev.reshape((N, n_c, f * f))  # shape (N, n_c, f*f)
                a_prev_ba[:, :, idx, :] = sub_a_prev_tran

                idx += 1

        # a_prev_ba shape (N, n_c, n_h*n_w, f * f)

        # y_ba = np.average(a_prev_ba, axis=3)  # shape (N, n_c, n_h*n_w)

        W_ba = 1 / (f * f) * np.ones((f * f, 1))  # shape (f*f,1)
        y_ba = np.dot(a_prev_ba, W_ba).squeeze()  # shape (N, n_c, n_h*n_w)
        # shape (N, n_c, n_h*n_w, f * f), shape (f*f,1) -> (N, n_c, n_h*n_w, 1)

        out = y_ba.reshape((N, n_c, n_h, n_w))  # 输出的特征图

        cache = (layer_name, config_pool, a_prev, a_prev_ba, W_ba)  # 缓存中间结果, 为了反向传播

        return out, cache

    def average_pool_bakward(self, grad_out, cache):
        """
        平均池化层的反向传播算法

        :param grad_out: 上一层传递过来的梯度 , shape (N, n_c, n_h, n_w)
        :param cache: 前向传播缓存的中间结果

        :return: grad_a_prev - shape (N, n_c, h, w)

        """

        (layer_name, config_pool, a_prev, a_prev_ba, W_ba) = cache

        f = config_pool['f']  # 池化核的大小
        s = config_pool['s']  # 窗口滑动步长

        N, n_c, h, w = np.shape(a_prev)  # N-样本个数
        # n_c-在池化层中我们不改变通道的数目, 即输入的通道数目和输出的相同
        # h-图片的高度
        # w-图片的宽度

        # 在池化层中我们不会加 padding 填充
        # 输出特征图的高度和宽度
        n_h = int((h - f) / s + 1)
        n_w = int((w - f) / s + 1)

        # y_ba shape (N, n_c, n_h*n_w)
        grad_y_ba = grad_out.reshape((N, n_c, n_h * n_w))  # shape (N, n_c, n_h*n_w)

        # a_prev_ba shape (N, n_c, n_h*n_w, f * f)
        grad_a_prev_ba = np.dot(np.expand_dims(grad_y_ba, axis=3), W_ba.T)  # shape (N, n_c, n_h*n_w, f * f)
        # shape (N, n_c, n_h*n_w, 1) , shape (1, f*f) -> shape (N, n_c, n_h*n_w, f * f)

        # 将 grad_a_prev_ba 还原为 grad_a_prev, 以便传递给上一层

        # a_prev shape (N, n_c, h, w)
        grad_a_prev = np.zeros((N, n_c, h, w))  # shape (N, n_c, h, w)

        idx = 0
        for row in range(0, h + 1 - f, s):
            for col in range(0, w + 1 - f, s):
                grad_a_prev[:, :, row:row + f, col:col + f] += grad_a_prev_ba[:, :, idx, :].reshape((N, n_c, f, f))
                idx += 1

        return grad_a_prev

    def relu_forward(self, a_prev):
        """
        relu激活函数层的前向传播算法

        :param a_prev: 上一层的输出作为本层的输入 shape (N, n_c_prev, h, w)
                          N-样本个数
                          n_c_prev-卷积核的层数(输入的通道个数)
                          h-图片的高度
                          w-图片的宽度

        :return: out - shape (N, n_c_prev, h, w)
                 cache - 缓存中间结果 a_prev
        """

        out = Activation.relu(a_prev)
        cache = a_prev  # 缓存中间结果, 用于反向传播

        return out, cache

    def relu_backward(self, grad_out, cache):
        """
        relu激活函数层的反向传播算法

        :param grad_out: 上一层传递过来的梯度 , shape (N, n_c, n_h, n_w)
        :param cache: 前向传播缓存的中间结果

        :return: grad_a_prev - shape (N, n_c, n_h, n_w)

        """

        a_prev = cache

        grad_a_prev = grad_out * Activation.grad_relu(a_prev)

        return grad_a_prev

    def affine_forward(self, parameters, layer_name, a_prev):
        """
        全连接层的前向传播算法

        :param parameters: 模型参数
        :param layer_name: 当前层的名字
        :param a_prev: 上一层的输出作为本层的输入 shape (N, n_c_prev, h, w)
                                  N-样本个数
                                  n_c_prev-卷积核的层数(输入的通道个数)
                                  h-图片的高度
                                  w-图片的宽度

        :return: z_y - shape (class_num,N)
                 y_ba - shape (class_num,N)
                 cache - (parameters, layer_name, z_y, a_prev_flat, a_prev)
        """

        N, n_c_prev, h, w = np.shape(a_prev)  # N-样本个数
        # n_c_prev-输入的通道数目
        # h-图片的高度
        # w-图片的宽度

        W = parameters['W' + '_' + str(layer_name)]  # shape (class_num,n_k)
        b = parameters['b' + '_' + str(layer_name)]  # shape (class_num,1)

        # 把 a_prev 拍平
        n_k = n_c_prev * h * w
        a_prev_flat = a_prev.reshape(N, n_k)  # shape (N, n_k)

        z_y = np.dot(W, a_prev_flat.T) + b  # shape (class_num,N)
        # shape (class_num,n_k), shape (n_k, N) -> shape (class_num,N)

        y_ba = Activation.softmax(z_y)  # shape (class_num, N)

        cache = (parameters, layer_name, z_y, a_prev_flat, a_prev)  # 缓存中间结果, 用于反向传播

        return z_y, y_ba, cache

    def affine_backward(self, grad_z_y, cache):
        """
        全连接层的反向传播算法

        :param grad_z_y: 对 z_y 的梯度 shape (class_num,N)
        :param cache: 前向传播缓存的中间结果

        :return: grad_a_prev - 传递给前一层的梯度, shape (N, n_c_prev, h, w) ;
                 grad_dic - 需要更新的模型的参数的梯度

        """
        (parameters, layer_name, z_y, a_prev_flat, a_prev) = cache

        N, n_c_prev, h, w = np.shape(a_prev)  # N-样本个数
        # n_c_prev-输入的通道数目
        # h-图片的高度
        # w-图片的宽度

        W = parameters['W' + '_' + str(layer_name)]  # shape (class_num,n_k)
        b = parameters['b' + '_' + str(layer_name)]  # shape (class_num,1)

        # a_prev_flat shape (N, n_k=n_c_prev * h * w)
        # a_prev shape (N, n_c_prev, h, w)

        grad_W = np.dot(grad_z_y, a_prev_flat)  # shape (class_num, n_k)
        # shape (class_num, N), shape (N, n_k) ->  shape (class_num, n_k)

        grad_b = np.sum(grad_z_y, axis=1, keepdims=True)  # shape (class_num,1)

        grad_a_prev_flat = np.dot(W.T, grad_z_y).T  # shape (N, n_k)
        # shape (n_k, class_num)  , shape (class_num,N) -> shape (n_k, N)

        # 将 grad_a_prev_flat 还原为 grad_a_prev, 以便传递给上一层

        grad_a_prev = grad_a_prev_flat.reshape((N, n_c_prev, h, w))  # shape (N, n_c_prev, h, w)

        # 将需要更新的模型参数的梯度包装为 dict
        grad_dic = {"grad_W_" + str(layer_name): grad_W, "grad_b_" + str(layer_name): grad_b
                    }

        return grad_a_prev, grad_dic

    def multi_classify_loss_func(self, z_y, y_ba, y_onehot):
        """
        计算多分类下的损失函数

        :param z_y: shape (class_num,N)

        :param y_ba: shape (class_num,N)
        :param y_onehot: 样本标签 shape (class_num,N)
                         N- 样本个数
                         class_num-分类的个数

        :return: loss - 损失函数的值 ;
                 grad_z_y -  shape (class_num,N)

        """
        class_num, N = np.shape(z_y)

        grad_z_y = y_ba - y_onehot  # shape (class_num,N)
        grad_z_y /= N

        loss = np.sum((-y_onehot) * Activation.log_softmax(z_y))  # shape:(1,)
        #  y_onehot shape (class_num,N) , z_y shape: (class_num,N)

        loss = loss / N

        return loss, grad_z_y


class UnitTest:
    """
    单元测试

    """

    def test_convolution_forward(self):

        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)

        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        W_conv1 = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b_conv1 = np.linspace(-0.1, 0.2, num=3)

        param = {"W_conv1": W_conv1, "b_conv1": b_conv1}

        conv_param = {'f': 4, 's': 2, 'p': 1, 'n_c': 3}

        cnn_layer = CNNLayer()

        out, _ = cnn_layer.convolution_forward(parameters=param, layer_name='conv1', config_conv=conv_param, a_prev=x)

        correct_out = np.array([[[[-0.08759809, -0.10987781],
                                  [-0.18387192, -0.2109216]],
                                 [[0.21027089, 0.21661097],
                                  [0.22847626, 0.23004637]],
                                 [[0.50813986, 0.54309974],
                                  [0.64082444, 0.67101435]]],
                                [[[-0.98053589, -1.03143541],
                                  [-1.19128892, -1.24695841]],
                                 [[0.69108355, 0.66880383],
                                  [0.59480972, 0.56776003]],
                                 [[2.36270298, 2.36904306],
                                  [2.38090835, 2.38247847]]]])

        # Compare your output to ours; difference should be around e-8
        print('Testing conv_forward')
        print('difference: ', rel_error(out, correct_out))

    def test_convolution_bakward(self):

        np.random.seed(231)  # 每次生成固定的随机数

        x = np.random.randn(4, 3, 5, 5)
        W_conv1 = np.random.randn(2, 3, 3, 3)
        b_conv1 = np.random.randn(2, )

        grad_out = np.random.randn(4, 2, 5, 5)

        conv_param = {'f': 3, 's': 1, 'p': 1, 'n_c': 2}

        params = {"W_conv1": W_conv1, "b_conv1": b_conv1}

        cnn_layer = CNNLayer()
        out, cache = cnn_layer.convolution_forward(parameters=params, layer_name='conv1', config_conv=conv_param,
                                                   a_prev=x)

        grad_a_prev, grad_dic = cnn_layer.convolution_bakward(grad_out=grad_out, cache=cache)
        dx = grad_a_prev

        fx = lambda x: \
            cnn_layer.convolution_forward(parameters=params, layer_name='conv1', config_conv=conv_param, a_prev=x)[0]

        def fW(W):
            tmp = params['W_conv1']
            params['W_conv1'] = W
            res = cnn_layer.convolution_forward(parameters=params, layer_name='conv1', config_conv=conv_param,
                                                   a_prev=x)[0]
            params['W_conv1'] = tmp
            return res

        def fb(b):
            tmp = params['b_conv1']
            params['b_conv1'] = b
            res = cnn_layer.convolution_forward(parameters=params, layer_name='conv1', config_conv=conv_param,
                                                   a_prev=x)[0]
            params['b_conv1'] = tmp
            return res

        dx_num = eval_numerical_gradient_array(fx, x, grad_out)

        dW_num = eval_numerical_gradient_array(fW, W_conv1, grad_out)

        db_num = eval_numerical_gradient_array(fb, b_conv1, grad_out)

        print('dx error: ', rel_error(dx_num, dx))
        print('dW error: ', rel_error(dW_num, grad_dic['grad_W_conv1']))
        print('db error: ', rel_error(db_num, grad_dic['grad_b_conv1']))

    def test_max_pool_forward(self):

        x_shape = (2, 3, 4, 4)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)

        config_pool1 = {'f': 2, 's': 2}

        cnn_layer = CNNLayer()
        out, _ = cnn_layer.max_pool_forward(layer_name='max_pool1', config_pool=config_pool1, a_prev=x)

        correct_out = np.array([[[[-0.26315789, -0.24842105],
                                  [-0.20421053, -0.18947368]],
                                 [[-0.14526316, -0.13052632],
                                  [-0.08631579, -0.07157895]],
                                 [[-0.02736842, -0.01263158],
                                  [0.03157895, 0.04631579]]],
                                [[[0.09052632, 0.10526316],
                                  [0.14947368, 0.16421053]],
                                 [[0.20842105, 0.22315789],
                                  [0.26736842, 0.28210526]],
                                 [[0.32631579, 0.34105263],
                                  [0.38526316, 0.4]]]])

        # Compare your output with ours. Difference should be on the order of e-8.
        print('Testing max_pool_forward function:')
        print('difference: ', rel_error(out, correct_out))

    def test_max_pool_bakward(self):

        np.random.seed(231)

        x = np.random.randn(3, 2, 8, 8)

        grad_out = np.random.randn(3, 2, 4, 4)

        config_pool1 = {'f': 2, 's': 2}

        cnn_layer = CNNLayer()
        out, cache = cnn_layer.max_pool_forward(layer_name='max_pool1', config_pool=config_pool1, a_prev=x)

        grad_a_prev = cnn_layer.max_pool_bakward(grad_out=grad_out, cache=cache)
        dx = grad_a_prev

        fx = lambda x: \
            cnn_layer.max_pool_forward(layer_name='max_pool1', config_pool=config_pool1, a_prev=x)[0]

        dx_num = eval_numerical_gradient_array(fx, x, grad_out)

        # Your error should be on the order of e-12
        print('Testing max_pool_backward function:')
        print('dx error: ', rel_error(dx, dx_num))

    def test_average_pool_bakward(self):

        np.random.seed(231)

        x = np.random.randn(3, 2, 8, 8)

        grad_out = np.random.randn(3, 2, 4, 4)

        config_pool1 = {'f': 2, 's': 2}

        cnn_layer = CNNLayer()
        out, cache = cnn_layer.average_pool_forward(layer_name='average_pool1', config_pool=config_pool1, a_prev=x)

        grad_a_prev = cnn_layer.average_pool_bakward(grad_out=grad_out, cache=cache)
        dx = grad_a_prev

        fx = lambda x: \
            cnn_layer.average_pool_forward(layer_name='average_pool1', config_pool=config_pool1, a_prev=x)[0]

        dx_num = eval_numerical_gradient_array(fx, x, grad_out)

        # Your error should be on the order of e-12
        print('Testing average_pool_backward function:')
        print('dx error: ', rel_error(dx, dx_num))


    def test_affine_backward(self):

        np.random.seed(231)  # 每次生成固定的随机数

        N = 4
        x = np.random.randn(N, 16, 5, 5)

        W_affine1 = np.random.randn(10, 400)
        b_affine1 = np.zeros((10, 1))

        # grad_z_y: 对 z_y 的梯度 shape (class_num,N)
        grad_out = np.random.randn(10, N)

        params = {"W_affine1": W_affine1, "b_affine1": b_affine1}

        cnn_layer = CNNLayer()

        _, _, cache = cnn_layer.affine_forward(parameters=params, layer_name='affine1', a_prev=x)

        grad_a_prev, grad_dic = cnn_layer.affine_backward(grad_z_y=grad_out, cache=cache)
        dx = grad_a_prev

        fx = lambda x: \
            cnn_layer.affine_forward(parameters=params, layer_name='affine1', a_prev=x)[0]

        def fW(W):
            tmp = params['W_affine1']
            params['W_affine1'] = W
            res = cnn_layer.affine_forward(parameters=params, layer_name='affine1', a_prev=x)[0]
            params['W_affine1'] = tmp
            return res

        def fb(b):
            tmp = params['b_affine1']
            params['b_affine1'] = b
            res = cnn_layer.affine_forward(parameters=params, layer_name='affine1', a_prev=x)[0]
            params['b_affine1'] = tmp
            return res

        dx_num = eval_numerical_gradient_array(fx, x, grad_out)

        dW_num = eval_numerical_gradient_array(fW, W_affine1, grad_out)

        db_num = eval_numerical_gradient_array(fb, b_affine1, grad_out)

        print('dx error: ', rel_error(dx_num, dx))
        print('dW error: ', rel_error(dW_num, grad_dic['grad_W_affine1']))
        print('db error: ', rel_error(db_num, grad_dic['grad_b_affine1']))



if __name__ == '__main__':
    test = UnitTest()

    # test.test_convolution_forward()
    #
    # test.test_convolution_bakward()
    #
    # test.test_max_pool_forward()

    # test.test_max_pool_bakward()

    # test.test_average_pool_bakward()

    test.test_affine_backward()