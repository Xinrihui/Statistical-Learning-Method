#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

import math

import time

class Optimizer:
    """
    优化算法

    class BGDOptimizer 批量梯度下降(BGD)
    class MinBatchOptimizer  Mini-batch 梯度下降
    class MomentumOptimizer 带动量(Momentum)的 Mini-batch 梯度下降
    class AdamOptimizer  Adam Mini-batch 梯度下降

    Author: xrh
    Date: 2021-07-14

    """


    def __init__(self, *args,**kwargs):
        """

        :param args: 不确定个数的(可能有多个, 也可能没有)位置参数
        :param kwargs: 不确定个数的键值参数

        eg.
        func()
        func(1,2,3)
        func( 1 ,arg2="two", arg3=3)

        """

        pass


    def fit(self, func_forwoard,func_backwoard): # TODO

        pass

    def get_batches(self,X, y_onehot,**kwargs):
        """
        获取 所有批次的训练数据

        :param X:
        :param y_onehot:
        :return:
        """
        pass

    def update_parameters(self, learning_rate, parameters, grad_W_list, grad_b_list,grad_gama_list,grad_beta_list,**kwargs):
        """
        根据反向传播计算得到梯度信息 更新 模型参数
        :param learning_rate:
        :param parameters:
        :param grad_W_list:
        :param grad_b_list:
        :return:
        """
        pass

class BGDOptimizer(Optimizer):

    def get_batches(self,X, y_onehot,**kwargs):
        """
        获取 所有批次的训练数据
        BGD 中就只有一个批次, 里面有整个训练集的数据
        :param X:
        :param y_onehot:
        :return:
        """
        batches = [(X,y_onehot)]

        return batches


    def update_parameters(self, learning_rate, parameters, grad_W_list, grad_b_list,grad_gama_list,grad_beta_list,**kwargs):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param learning_rate:
        :param parameters:
        :param grad_W_list:
        :param grad_b_list:
        :return:
        """
        W_list = parameters['W']
        b_list = parameters['b']

        gama_list = parameters['gama']
        beta_list = parameters['beta']

        L = len(W_list)  # MLP 的层数

        for l in range(L):  # 遍历输入层, 隐藏层和输入层 l = 0 ,...,L-1

            W_list[l] -= learning_rate * grad_W_list[l]
            b_list[l] -= learning_rate * grad_b_list[l]

            gama_list[l] -= learning_rate * grad_gama_list[l]
            beta_list[l] -= learning_rate * grad_beta_list[l]

        return parameters



class MinBatchOptimizer(Optimizer):

    def random_mini_batches(self, X, y_onehot, mini_batch_size=640):
        """
        从样本中生成所有的 mini_batch, 以列表的形式返回

        :param X: shape (N,m)
        :param y_onehot: shape (K,N)
        :param mini_batch_size: 最小批次的大小
        :return:

        mini_batches -- list of (mini_batch_X, mini_batch_Y)

        """
        # seed = int(time.time()) # 随机种子用系统时间取整数得到, seed 的分辨率为 1s, 即每间隔1s 随机数种子发生变化
        # np.random.seed(seed)  # 确保每一个 epcho 都以不一样的顺序打乱训练样本

        N = X.shape[0]  # 训练样本的个数
        mini_batches = []  #

        # 打乱训练样本的次序 (X, y)
        permutation = list(np.random.permutation(N))
        shuffled_X = X[permutation, :]
        shuffled_y = y_onehot[:, permutation]

        # 对样本集合进行分区 (shuffled_X, shuffled_y)
        num_complete_minibatches = math.floor(N / mini_batch_size)  # 向下取整

        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1) * mini_batch_size, :]
            mini_batch_y = shuffled_y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_y)
            mini_batches.append(mini_batch)

        # 处理最后一个分区 (last mini-batch < mini_batch_size)
        if N % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: N, :]
            mini_batch_y = shuffled_y[:, num_complete_minibatches * mini_batch_size: N]
            mini_batch = (mini_batch_X, mini_batch_y)
            mini_batches.append(mini_batch)

        return mini_batches


    def get_batches(self,X, y_onehot,mini_batch_size=640):
        """
        获取 所有批次的训练数据

        :param X:
        :param y_onehot:
        :return:
        """
        return self.random_mini_batches(X=X, y_onehot=y_onehot, mini_batch_size=mini_batch_size)


    def update_parameters(self, learning_rate, parameters, grad_W_list, grad_b_list,grad_gama_list,grad_beta_list,**kwargs):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param learning_rate:
        :param parameters:
        :param grad_W_list:
        :param grad_b_list:
        :return:
        """
        W_list = parameters['W']
        b_list = parameters['b']

        gama_list = parameters['gama']
        beta_list = parameters['beta']

        L = len(W_list)  # MLP 的层数

        for l in range(L):  # 遍历输入层, 隐藏层和输入层 l = 0 ,...,L-1

            W_list[l] -= learning_rate * grad_W_list[l]
            b_list[l] -= learning_rate * grad_b_list[l]

            gama_list[l] -= learning_rate * grad_gama_list[l]
            beta_list[l] -= learning_rate * grad_beta_list[l]


        return parameters


class MomentumOptimizer(MinBatchOptimizer):

    def __init__(self, parameters, beta1=0.9):
        """

        :param parameters:
        :param beta1: 相当于定义了计算指数加权平均数时的窗口大小

        eg.
        beta=0.5  窗口大小为: 1/(1-beta) = 2
        beta=0.9  窗口大小为: 1/(1-beta) = 10
        beta=0.99 窗口大小为: 1/(1-beta) = 100

        """

        self.beta1 = beta1

        W_list = parameters['W']
        b_list = parameters['b']

        gama_list = parameters['gama']
        beta_list = parameters['beta']

        L = len(W_list)  # MLP 的层数

        v_W_list = []  #
        v_b_list = []  #

        v_gama_list = []  #
        v_beta_list = []  #


        for l in range(L):  # 遍历输入层, 隐藏层和输入层 l = 0 ,...,L-1

            v_W_list.append(np.zeros(np.shape(W_list[l])))
            v_b_list.append(np.zeros(np.shape(b_list[l])))

            v_gama_list.append(np.zeros(np.shape(gama_list[l])))
            v_beta_list.append(np.zeros(np.shape(beta_list[l])))


        self.v_W_list = v_W_list
        self.v_b_list = v_b_list

        self.v_gama_list = v_gama_list
        self.v_beta_list = v_beta_list


    def update_parameters(self, learning_rate, parameters, grad_W_list, grad_b_list,grad_gama_list,grad_beta_list,**kwargs):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param learning_rate:
        :param parameters:
        :param grad_W_list:
        :param grad_b_list:
        :return:
        """
        W_list = parameters['W']
        b_list = parameters['b']

        gama_list = parameters['gamma']
        beta_list = parameters['beta']

        L = len(W_list)  # MLP 的层数

        for l in range(L):  # 遍历输入层, 隐藏层和输入层 l = 0 ,...,L-1

            self.v_W_list[l] = self.beta1 * self.v_W_list[l] + (1 - self.beta1) * grad_W_list[l]
            self.v_b_list[l] = self.beta1 * self.v_b_list[l] + (1 - self.beta1) * grad_b_list[l]

            self.v_gama_list[l] = self.beta1 * self.v_gama_list[l] + (1 - self.beta1) * grad_gama_list[l]
            self.v_beta_list[l] = self.beta1 * self.v_beta_list[l] + (1 - self.beta1) * grad_beta_list[l]


            W_list[l] -= learning_rate * self.v_W_list[l]
            b_list[l] -= learning_rate * self.v_b_list[l]

            gama_list[l] -= learning_rate * self.v_gama_list[l]
            beta_list[l] -= learning_rate * self.v_beta_list[l]

        return parameters

class AdamOptimizer(MinBatchOptimizer):

    def __init__(self, parameters,
                       beta1 = 0.9,
                       beta2 = 0.99,
                       epsilon = 1e-8):
        """

        :param parameters:
        :param beta1: 惯性保持, 历史梯度和当前梯度的平均 (默认 0.9)
        :param beta2: 环境感知, 为不同的模型参数产生自适应的学习率 (默认 0.99)
                      beta1, beta1 一般无需调节
        :param epsilon: 一个很小的数

        eg.
        beta1=0.5  历史梯度的窗口大小为: 1/(1-beta1) = 2
        beta1=0.9  历史梯度的窗口大小为: 1/(1-beta1) = 10
        beta1=0.99 历史梯度的窗口大小为: 1/(1-beta1) = 100

        """

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        W_list = parameters['W']
        b_list = parameters['b']

        gama_list = parameters['gama']
        beta_list = parameters['beta']

        L = len(W_list)  # MLP 的层数

        # 参数的一阶矩, 体现惯性保持
        m_W_list=[]
        m_b_list = []
        m_gama_list=[]
        m_beta_list = []

        # 参数的二阶矩, 体现环境感知
        v_W_list = []
        v_b_list = []
        v_gama_list = []
        v_beta_list = []

        for l in range(L):  # 遍历输入层, 隐藏层和输入层 l = 0 ,...,L-1

            m_W_list.append(np.zeros(np.shape(W_list[l])))
            m_b_list.append(np.zeros(np.shape(b_list[l])))

            m_gama_list.append(np.zeros(np.shape(gama_list[l])))
            m_beta_list.append(np.zeros(np.shape(beta_list[l])))

            v_W_list.append(np.zeros(np.shape(W_list[l])))
            v_b_list.append(np.zeros(np.shape(b_list[l])))

            v_gama_list.append(np.zeros(np.shape(gama_list[l])))
            v_beta_list.append(np.zeros(np.shape(beta_list[l])))


        self.m_W_list = m_W_list
        self.m_b_list = m_b_list
        self.m_gama_list = m_gama_list
        self.m_beta_list = m_beta_list

        self.v_W_list = v_W_list
        self.v_b_list = v_b_list
        self.v_gama_list = v_gama_list
        self.v_beta_list = v_beta_list


    def update_parameters(self, learning_rate, parameters, grad_W_list, grad_b_list,grad_gama_list,grad_beta_list,t=0,use_bias_correct=False):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param learning_rate:
        :param parameters:
        :param grad_W_list:
        :param grad_b_list:

        :param t: 当前时刻 t
        :param use_bias_correct: 开启偏差修正, 在 t 较小时, 使得均值的计算更准确
                                 ( 默认: 关闭 False)
        :return:
        """
        W_list = parameters['W']
        b_list = parameters['b']

        gama_list = parameters['gama']
        beta_list = parameters['beta']

        L = len(W_list)  # MLP 的层数

        for l in range(L):  # 遍历输入层, 隐藏层和输入层 l = 0 ,...,L-1

            # 一阶矩
            self.m_W_list[l] = self.beta1 * self.m_W_list[l] + (1-self.beta1)*grad_W_list[l]
            self.m_b_list[l] = self.beta1 * self.m_b_list[l] + (1 -self.beta1) * grad_b_list[l]

            self.m_gama_list[l] = self.beta1 * self.m_gama_list[l] + (1 - self.beta1) * grad_gama_list[l]
            self.m_beta_list[l] = self.beta1 * self.m_beta_list[l] + (1 - self.beta1) * grad_beta_list[l]

            # 二阶矩
            self.v_W_list[l] = self.beta2 * self.v_W_list[l] + (1-self.beta2)* np.square(grad_W_list[l])
            self.v_b_list[l] = self.beta2 * self.v_b_list[l] + (1 -self.beta2) * np.square(grad_b_list[l])

            self.v_gama_list[l] = self.beta2 * self.v_gama_list[l] + (1-self.beta2)* np.square(grad_gama_list[l])
            self.v_beta_list[l] = self.beta2 * self.v_beta_list[l] + (1 -self.beta2) * np.square(grad_beta_list[l])

            # 偏差修正 TODO: 会造成计算溢出,模型不收敛,原因未知
            if use_bias_correct:
                z1 = 1-(self.beta1**t)
                z2 = 1 - (self.beta2**t)
                self.m_W_list[l] = self.m_W_list[l] / z1
                self.m_b_list[l] = self.m_b_list[l] / z1
                self.m_gama_list[l] = self.m_gama_list[l] / z1
                self.m_beta_list[l] = self.m_beta_list[l] / z1

                self.v_W_list[l] = self.v_W_list[l] / z2
                self.v_b_list[l] = self.v_b_list[l] / z2
                self.v_gama_list[l] = self.v_gama_list[l] / z2
                self.v_beta_list[l] = self.v_beta_list[l] / z2


            # 一阶矩 二阶矩融合
            W_list[l] -= ((learning_rate * self.m_W_list[l]) / np.sqrt(self.v_W_list[l]+self.epsilon))

            b_list[l] -= ((learning_rate * self.m_b_list[l]) / np.sqrt(self.v_b_list[l]+self.epsilon))

            gama_list[l] -= ((learning_rate * self.m_gama_list[l]) / np.sqrt(self.v_gama_list[l]+self.epsilon))

            beta_list[l] -= ((learning_rate * self.m_beta_list[l]) / np.sqrt(self.v_beta_list[l]+self.epsilon))

        return parameters



