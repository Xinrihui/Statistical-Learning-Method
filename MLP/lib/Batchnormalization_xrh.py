#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

class BatchNormalization:
    """
    BatchNormalization 批归一化

    Author: xrh
    Date: 2021-07-17

    """

    @staticmethod
    def batchnorm_forward(l,x, gama, beta, bn_param):
        """
        batchnorm 的正向算法

        :param l : 记录从前往后第 l 层神经元
        :param x: 输入数据 shape: (n_current,N)
        :param gama: 方差参数 shape: (n_current,1)
        :param beta: 均值参数 shape: (n_current,1)
        :param bn_param:  batch normalizaion 的超参数
        :return:
        """

        mode = bn_param['mode']
        epsilon = bn_param.get('epsilon', 1e-5)
        beta1 = bn_param.get('beta1', 0.9)

        n_current, N = np.shape(x)

        # 均值和方差的指数加权平均数, 若 bn_param 中没有, 则初始化为 0
        average_mean = bn_param.get('average_mean'+str(l), np.zeros((n_current,1), dtype=x.dtype))
        average_var = bn_param.get('average_var'+str(l), np.zeros((n_current,1), dtype=x.dtype))

        if mode == 'training': #训练模式

            # X shape: (h,N)
            mean = np.mean(x , axis=1, keepdims=True) # shape: (n_current,1)
            var = np.var(x, axis=1, keepdims=True ) + epsilon # shape: (n_current,1)
            std = np.sqrt(var + epsilon) # shape: (n_current,1)

            x_ba = (x - mean) / std

            y = gama * x_ba + beta

            # 对 均值 和 方差 计算指数加权平均数, 推理时使用
            average_mean = beta1 * average_mean + (1-beta1)*mean
            average_var = beta1 * average_var + (1 - beta1) * var

            bn_param['average_mean'+str(l)] = average_mean
            bn_param['average_var'+str(l)] = average_var


        else: #  mode == 'inference': #推理模式

            #  推理时使用 指数加权平均数
            mean = bn_param['average_mean'+str(l)]
            var = bn_param['average_var'+str(l)]
            std = np.sqrt(var + epsilon)

            x_ba = (x - mean) / std
            y = gama * x_ba + beta

        out = y
        cache = {'x':x,'mean':mean,'std':std,'x_ba':x_ba,'y':y,'z_ba': y}

        return cache,out

    @staticmethod
    def batchnorm_bakward(N,gama,beta,grad_z_ba,cache_bn):
        """
        batchnorm 的反向算法

        :param N: min-batch 中一批样本的个数
        :param gama: shape: (n_current,1)
        :param beta: shape: (n_current,1)
        :param grad_z_ba: shape: (n_current,N)

        :param cache_bn:
        cache_bn = cache = {'x':x,'mean':mean,'std ':std,'x_ba':x_ba,'y':y,'z_ba': y}

        :return:
        """
        x_ba = cache_bn['x_ba'] # shape: (n_current,N)
        std = cache_bn['std']  # shape: (n_current,1)

        grad_gama = np.sum(grad_z_ba*x_ba, axis=1, keepdims=True)
        # grad_z_ba shape: (n_current,N) , x_ba shape: (n_current,N)

        grad_beta = np.sum(grad_z_ba, axis=1, keepdims=True)
        # grad_z_ba shape: (n_current,N)

        grad_x_ba =  grad_z_ba * gama # shape: (n_current,N)
        # grad_x_ba shape: (n_current,N) , gama shape: (n_current,1)

        # 计算 grad_x
        first = N*grad_x_ba
        # grad_x_ba shape: (n_current,N)

        second = np.sum(grad_x_ba, axis=1, keepdims=True) # shape: (n_current,1)
        # grad_x_ba shape: (n_current,N)

        third = x_ba * np.sum(grad_x_ba*x_ba, axis=1, keepdims=True) # shape: (n_current,N)
        # x_ba shape: (n_current,N) , grad_x_ba shape: (n_current,N)

        grad_x = (first - second - third) / (N*std) # shape: (n_current,N)

        return grad_x,grad_gama,grad_beta
