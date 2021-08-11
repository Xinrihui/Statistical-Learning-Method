#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

import time
from deprecated import deprecated

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt
import matplotlib as mpl

from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets

from lib.Normalizer_xrh import *

from lib.Activation_xrh import *
from lib.Batchnormalization_xrh import *

from lib import Optimizer_xrh as optim

from lib import Initializer_xrh as initial


import importlib

import pickle

import math

import warnings

# warnings.filterwarnings('error')


class MLP_2Classifier:
    """
    多层感知机 MLP (二分类)

    1.通过 数值优化方法 梯度下降 找到最优解


    Author: xrh
    Date: 2021-07-07

    ref:
    deeplearning.ai 吴恩达

    test1: 二分类任务
    数据集：Mnist
    参数: layers_dims=[784,50,10,1], use_reg=2, reg_lambda=0.7, learning_rate=1.0,max_iter=500
    训练集数量：60000
    测试集数量：10000
    正确率：0.99
    训练时长： 445s

    """

    def __init__(self, reg_alpha=0.1, reg_lambda=0.1, use_reg=2):
        """

        :param reg_alpha: L1 正则化参数
        :param reg_lambda: L2 正则化参数
        :param use_reg: 正则化类型选择, 2: L2 正则化 ; 1: L1 正则化 ; 0: 不使用正则化
        """

        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.use_reg = use_reg

        # 模型的参数
        self.parameters = {}

    def sigmoid(self, X):
        """
        sigmoid 激活函数

        :param X:
        :return:
        """

        return 1 / (1 + np.exp(-X))

    def initialize_parameters(self, layers_dims):
        """
        初始化 MLP 的参数

        :param layers_dims 从前往后各个层的维度

        eg.
        layers_dims= [n_x,n_h,n_y]
        n_x: 输入层向量的维度
        n_h: 隐藏层向量的维度
        n_y: 输出层向量的维度

        :return:
            W1 -- weight matrix of shape (n_h, n_x)
            b1 -- bias vector of shape (n_h, 1)
            W2 -- weight matrix of shape (n_y, n_h)
            b2 -- bias vector of shape (n_y, 1)
        """

        W_list = []
        b_list = []

        for i in range(0, len(layers_dims) - 1):  # layers_dims: [n_x,n_h,n_y]

            W = np.random.randn(layers_dims[i + 1], layers_dims[i]) * 0.01  # 随机初始化, 生成的数值满足标准正态分布
            # W = np.zeros((layers_dims[i+1], layers_dims[i]))

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

    def f_activation(self, z):
        """
        输入层 l=0
        隐藏层 l=1,...,L-2

        使用的激活函数
        :param z:
        :return:
        """
        a = self.sigmoid(z)

        return a

    def grad_activation(self, z):
        """
        激活函数的导数

        :param z:
        :return:
        """
        p = self.sigmoid(z)

        return p * (1 - p)

    def forwoard_propagation(self, parameters, X, y=None, mode='training'):
        """
        神经网络的前向传播算法

        parameters:
        W1 -- weight matrix of shape (n_h, m)
        b1 -- bias vector of shape (n_h,1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y,1)

        :param X: 样本特征 shape (N,m) N- 样本个数 , m-特征维度
        :param y: 样本标签
        :param parameters: 模型参数
        :param mode: 'training' 训练模式
                    'inference' 推理模式
                    前向传播算法在模型训练和推理时都要用到
        :return:
        """
        W_list = parameters['W']
        b_list = parameters['b']

        z_list = []
        a_list = []

        L = len(W_list)  # MLP 的层数 L=2

        # l=0 输入层
        l = 0

        z = np.dot(W_list[0], X.T) + b_list[0]  # shape (n_h,N)
        # W1 (n_h, m) , X.T (m,N) ->  (n_h,N)
        # b1 (n_h,1)  boardcast->  (n_h,N)
        a = self.f_activation(z)

        z_list.append(z)
        a_list.append(a)

        # 隐藏层 l=1,...,L-2
        for l in range(1, L - 1):
            z = np.dot(W_list[l], a) + b_list[l]  # shape (n_y,N)
            # W2 (n_y, n_h) , a1 (n_h,N) -> (n_y,N)
            # b2 (n_y,1)  boardcast->  (n_y,N)
            a = self.f_activation(z)  # shape (n_y,N)

            z_list.append(z)
            a_list.append(a)

        # 输出层 l=L-1
        z, a, loss = self.last_layer_loss(a_list[-1], parameters, y, mode=mode)

        z_list.append(z)
        a_list.append(a)

        return z_list, a_list, loss

    def last_layer_loss(self, a, parameters, y=None, mode='training'):
        """
        计算前向传播的最后一层, 并计算损失函数

        :param a: 倒数第二层的激活值, 作为最后一层的输入
        :param parameters:
        :param y:
        :param mode: 'training' 训练模式
                    'inference' 推理模式 , 不用计算损失, 也无需传入参数 y

        :return:
        """
        W_list = parameters['W']
        b_list = parameters['b']

        W = W_list[-1]  # 最后一层的 W
        b = b_list[-1]  # 最后一层的 b

        z = np.dot(W, a) + b  # shape (n_y,N)
        # W (n_y, n_h) , a (n_h,N) -> (n_y,N)
        # b2 (n_y,1)  boardcast->  (n_y,N)
        a = self.sigmoid(z)  # shape (n_y,N)

        loss = None

        if mode == 'training':

            N = np.shape(y)[0]  # N 个样本, m 个特征

            loss = np.sum(-(y.T * np.log(a) + (1 - y.T) * np.log(1 - a)))  # shape:(1,)
            # y.T shape (1,N) , a_final shape (n_y=1,N)

            loss = loss / N

            # 加上 L2 正则化
            if self.use_reg == 2:

                s = 0
                for W in W_list:  # 遍历所的 W, 并求平方和
                    s += np.sum(np.square(W))

                loss += (self.reg_lambda / (2 * N)) * s

            elif self.use_reg == 1:  # 加上 L1 正则化

                s = 0
                for W in W_list:
                    s += np.sum(np.abs(W))

                loss += (self.reg_alpha / (2 * N)) * s

        return z, a, loss

    def backwoard_propagation(self, parameters, z_list, a_list, X, y):
        """
        计算反向传播

        :param parameters:
        :param z_list:
        :param a_list:
        :param X:
        :param y:

        :return:
        """
        N = np.shape(y)[0]

        W_list = parameters['W']
        b_list = parameters['b']

        L = len(W_list)  # MLP 的层数 L=2

        grad_W_list = []
        grad_b_list = []
        grad_a_list = []
        grad_z_list = []  # delta

        grad_a = 0

        # 从后往前遍历 输出层 隐藏层 和 输入层  l = L-1 ,...,0
        for l in range(L - 1, -1, -1):

            if l == L - 1:  # 输出层
                grad_z = a_list[l] - y  # shape:(n_y,N)

            else:  # 隐藏层 和 输入层
                grad_z = grad_a * self.grad_activation(z_list[l])

            if l == 0:  # 输入层
                grad_W = np.dot(grad_z, X)  # TODO: 重要, 容易写错
                # grad_z shape:(n_h,N) ,  X shape:(N,m)

            else:  # 隐藏层 和 输出层
                grad_W = np.dot(grad_z, a_list[l - 1].T)
                # grad_z shape:(n_y,N) , a_list[l-1].T shape:(N,n_h)

            grad_W = grad_W / N

            # 加上 L2 正则化
            if self.use_reg == 2:
                grad_W += (self.reg_lambda / N) * W_list[l]

            # 加上 L1 正则化
            elif self.use_reg == 1:
                I = np.ones(np.shape(W_list[l]))
                I[W_list[l] < 0] = -1
                grad_W += (self.reg_alpha / N) * I

            grad_b = np.sum(grad_z, axis=1, keepdims=True)  # shape:(n_y,1)
            grad_b = grad_b / N

            grad_a = np.dot(W_list[l].T, grad_z)
            #  W_list[l].T shape:(n_h, n_y) ,  grad_z shape:(n_y,1)

            grad_W_list.append(grad_W)
            grad_b_list.append(grad_b)
            grad_a_list.append(grad_a)
            grad_z_list.append(grad_z)

        # 梯度列表全部反向, 接下来都要从前向后遍历
        grad_W_list.reverse()
        grad_b_list.reverse()
        grad_a_list.reverse()
        grad_z_list.reverse()

        return grad_W_list, grad_b_list

    def update_parameters(self, learning_rate, parameters, grad_W_list, grad_b_list):
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

        L = len(W_list)  # MLP 的层数

        for l in range(L):  # 遍历输入层, 隐藏层和输入层 l = 0 ,...,L-1

            W_list[l] -= learning_rate * grad_W_list[l]
            b_list[l] -= learning_rate * grad_b_list[l]

        return parameters

    def fit(self, X, y, layers_dims, learning_rate=1.0, max_iter=500, print_log=True, print_log_step=100):
        """
        训练模型

        :param X:
        :param y:
        :param layers_dims: MLP 每一层的维度 , [m,50,1]
        :param learning_rate: 学习率
        :param max_iter: 迭代次数
        :param print_log: 是否打印训练误差
        :param print_log_step: 打印训练误差的步长
        :return:
        """
        y = y.flatten()  # y 的维度必须为 (N,)

        N, m = np.shape(X)  # N 个样本, m 个特征

        print('train data num:{} , feature dimension:{}'.format(N, m))

        # 模型参数初始化
        parameters = self.initialize_parameters(layers_dims)

        for epoch in range(max_iter):

            z_list, a_list, loss = self.forwoard_propagation(parameters, X, y, mode='training')

            grad_W_list, grad_b_list = self.backwoard_propagation(parameters, z_list, a_list, X, y)

            parameters = self.update_parameters(learning_rate, parameters, grad_W_list, grad_b_list)

            # Print the cost every 100 iterations
            if print_log and epoch % print_log_step == 0:
                print('epcho: {} , loss:{}'.format(epoch, loss))

        # 保存训练参数, 用于推理
        self.parameters = parameters

    def predict_prob(self, X, parameters):
        """
        推理测试数据集, 返回样本的分值 (概率值),
        我们可以利用此分值得出 P-R 曲线

        :param X:
        :return:
        """

        z_list, a_list, _ = self.forwoard_propagation(parameters, X, mode='inference')  # 前向传播

        p = a_list[-1]

        return p

    def predict(self, X, threshold=0.5):
        """
        推理测试数据集，返回样本标签

        :param X:
        :param threshold:判断样本标签正负的阈值
        :return:
        """
        N = np.shape(X)[0]  # N 个样本

        res = np.zeros(N)

        parameters = self.parameters

        p = self.predict_prob(X, parameters)  # p:shape(N,1)
        # X: shape(N,m)

        p = p.flatten()  # p: shape(N,1) -> shape(N,)

        res[p > threshold] = 1

        return res


class MLP_MultiClassifier:
    """
    多层感知机 MLP (多分类)

    1.实现了向量化的前向传播和后向传播算法

    2.实现激活函数 sigmoid, relu

    3.实现如下优化算法:

     (1) 带正则化(L1, L2)的批量梯度下降(BGD)
     (2) Mini-batch 梯度下降
     (3) 带动量(Momentum)的 Mini-batch 梯度下降
     (4) Adam Mini-batch 梯度下降


    4.实现了 dropout 正则化

    5.实现了 Xavier 模型参数随机初始化


    Author: xrh
    Date: 2021-07-08

    ref:
    deeplearning.ai 吴恩达

    test1: 多分类任务
    数据集：Mnist
    超参数:
    layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    采用 Xavier参数随机初始化,
    开启 dropout 正则化, keep_prob=0.8
    使用 Adam 梯度下降,  beta1 = 0.9, beta2 = 0.99
    mini_batch_size = 640
    max_iter=100,
    learning_rate=0.01

    训练集数量：60000
    测试集数量：10000
    正确率：0.975
    训练时长： 323s

    """

    def __init__(self, K=None,
                 activation='sigmoid',

                 reg_alpha=0.5,
                 reg_lambda=0.5,
                 use_reg=0,

                 keep_prob=1.0,
                 use_dropout=False,

                 use_batchnorm=False,

                 model_path='model/xrh.model',
                 use_pre_train=True):
        """

        :param K : 多分类的类别
        :param activation: 输入层和隐藏层的激活函数
                           'sigmoid' (默认)
                           'relu'
        :param reg_alpha: L1 正则化参数 (默认 0.5)
        :param reg_lambda: L2 正则化参数 (默认 0.5 )
        :param use_reg: 正则化类型选择,
                        2: L2 正则化
                        1: L1 正则化
                        0: 不使用正则化(默认)

        :param keep_prob:  dropout 正则化中, 每一层的神经元有效的比例  (默认 1.0)
        :param use_dropout: 开启 dropout 正则化 (默认 False 不开启)

        :param use_batchnorm : 开启 batchnorm(默认 False 不开启)

        :param model_path: 预训练模型的路径
        :param use_pre_train: 是否使用预训练的模型
                             True: 读取预训练模型的参数后直接可以进行推理, 训练时在预训练的基础上进行训练
                             False: 从头开始训练模型

        """
        if not use_pre_train:  # 从头开始训练模型

            self.K = K

            self.activation = activation

            self.reg_alpha = reg_alpha
            self.reg_lambda = reg_lambda

            self.use_reg = use_reg

            self.keep_prob = keep_prob
            self.use_dropout = use_dropout

            self.bn_param = {} # BN 的超参数
            self.use_batchnorm = use_batchnorm

            # 模型的参数
            self.parameters = {}

            self.model_path = model_path

        else:  # 使用预训练模型

            self.load(model_path)



    def __forwoard_layer(self,l,a_prev, W, b, activation='sigmoid',
                            keep_prob=1.0, use_dropout=False,gamma=None,beta=None,bn_param=None,use_batchnorm=False):
        """
        单层的正向传播算法

        :param l : 记录从前往后第 l 层神经元
        :param a_prev: (n_prev,N)
        :param W: shape (n_current, n_prev)
                        n_current -当前层向量的维度
                        n_prev    -上一层向量的维度
        :param b: shape (n_current,1)
        :param activation: 选择的激活函数

        :param keep_prob:  dropout 正则化中, 每一层的神经元有效的比例  (默认 0.8)
        :param use_dropout: 开启 dropout 正则化 (默认 False 不开启)

        :param gamma: batch normalizaion 的方差参数
        :param beta:  batch normalizaion 的均值参数
        :param bn_param:  batch normalizaion 的超参数
        :param use_batchnorm: 开启 batch normalizaion
                              (默认 False 不开启)

        :return:
        """

        z = np.dot(W, a_prev) + b  # shape (n_current,N)
        # W shape (n_current, n_prev) , a_prev shape  (n_prev,N)
        # b (n_current,1)  boardcast->  (n_current,N)

        cache_bn = None
        if use_batchnorm: # 开启 batch normalizaion
            cache_bn,z_ba = BatchNormalization.batchnorm_forward(l,z, gamma, beta, bn_param)
        else:
            z_ba = z
            x_ba = None

        a = None

        if activation == 'sigmoid':
            a = Activation.sigmoid(z_ba)  # shape (n_current,N)

        elif activation == 'softmax':

            a = Activation.softmax(z_ba)  # shape (n_current,N)

        elif activation == 'relu':

            a = Activation.relu(z_ba)  # shape (n_current,N)

        d = None

        if use_dropout:  # 开启 dropout 正则化

            n_current, N = np.shape(a)
            d = np.random.rand(n_current, N)  # 生成[0.0, 1.0)之间的随机浮点数

            a[d > keep_prob] = 0  # a 以一定的概率失活

        return z, a, d,cache_bn

    def forwoard_propagation(self,
                             parameters,
                             X, y_onehot=None,
                             activation='sigmoid',
                             reg_alpha=0.5,
                             reg_lambda=0.5,
                             use_reg=0,
                             keep_prob=1.0,
                             use_dropout=False,
                             bn_param = None,
                             use_batchnorm = False,
                             mode='training'):
        """
        神经网络的前向传播算法

        :param parameters: 模型参数
        :param X: 样本特征 shape (N,m) N- 样本个数 , m-特征维度
        :param y_onehot: 样本标签
        :param activation: 输入层和隐藏层的激活函数
                           'sigmoid'
                           'relu'

        :param reg_alpha: L1 正则化参数 (默认 0.5)
        :param reg_lambda: L2 正则化参数 (默认 0.5 )
        :param use_reg: 正则化类型选择,
                        2: L2 正则化
                        1: L1 正则化
                        0: 不使用正则化(默认)

        :param keep_prob:  dropout 正则化中, 每一层的神经元有效的比例  (默认 1.0)
        :param use_dropout: 开启 dropout 正则化 (默认 False 不开启)

        :param bn_param:  batch normalizaion 的超参数
        :param use_batchnorm: 开启 batch normalizaion
                              (默认 False 不开启)

        :param mode: 'training' 训练模式
                    'inference' 推理模式
                    前向传播算法在模型训练和推理时都要用到
        :return:
        """

        if mode == 'inference':
            use_dropout = False  # 推理时肯定不开启 dropout 正则化
            bn_param['mode'] = 'inference' # 设置 batchnorm 模式为推理模式
        else:
            bn_param['mode'] = 'training'

        N = np.shape(X)[0]

        W_list = parameters['W']
        b_list = parameters['b']

        gama_list = parameters['gama']
        beta_list = parameters['beta']

        z_list = []
        a_list = []

        # 存储 dropout 中的失活向量
        d_list = []

        # 存储 batchnorm 过程中的 中间向量, 为了之后的反向传播
        #  cache = {'x':x,'mean':mean,'var':var,'x_ba':x_ba,'y':y,'z_ba': y}
        #  x_ba 归一化向量 , z_ba=y 等价变换向量

        cache_bn_list=[]

        L = len(W_list)  # MLP 的层数 L=2

        # l=0 输入层
        l = 0
        z, a, d, cache_bn = self.__forwoard_layer(l,a_prev=X.T, W=W_list[l], b=b_list[l], activation=activation,
                                        keep_prob=keep_prob, use_dropout=use_dropout,
                                        gamma=gama_list[l],beta=beta_list[l],bn_param=bn_param,use_batchnorm=use_batchnorm)
        # X shape -> X.T shape (m,N)

        z_list.append(z)
        a_list.append(a)
        d_list.append(d)
        cache_bn_list.append(cache_bn)

        # 隐藏层 l=1,...,L-2
        for l in range(1, L - 1):

            z, a, d,cache_bn = self.__forwoard_layer(l,a_prev=a, W=W_list[l], b=b_list[l], activation=activation,
                                            keep_prob=keep_prob, use_dropout=use_dropout,
                                            gamma=gama_list[l], beta=beta_list[l], bn_param=bn_param, use_batchnorm=use_batchnorm)

            z_list.append(z)
            a_list.append(a)
            d_list.append(d)
            cache_bn_list.append(cache_bn)

        # 输出层 l=L-1
        l = L - 1
        z, a, _, _= self.__forwoard_layer(l,a_prev=a, W=W_list[l], b=b_list[l], activation='softmax')
        # 输出层的激活函数 必须为 sofmax
        # 输出层的激活值不用加上 dropout 失活
        # 输出层不用考虑加上 batchnorm

        z_list.append(z)
        a_list.append(a)
        d_list.append(None)
        cache_bn_list.append(None)

        loss = None

        if mode == 'training':
            loss = self.last_layer_loss(parameters, N, z, y_onehot, reg_alpha=reg_alpha,reg_lambda=reg_lambda,use_reg=use_reg)

        return z_list, a_list, d_list, cache_bn_list,loss

    def last_layer_loss(self, parameters, N, z, y_onehot,
                        reg_alpha=0.5,reg_lambda=0.5,use_reg=0):
        """
        最后一层 计算损失函数

        :param parameters:
        :param N:  N 个样本
        :param z: 最后一层的 z
        :param y_onehot:
        :param reg_alpha: L1 正则化参数 (默认 0.5)
        :param reg_lambda: L2 正则化参数 (默认 0.5 )
        :param use_reg: 正则化类型选择,
                        2: L2 正则化
                        1: L1 正则化
                        0: 不使用正则化(默认)

        :return:
        """

        W_list = parameters['W']
        b_list = parameters['b']

        loss = np.sum(-y_onehot * Activation.log_softmax(z))  # shape:(1,)
        # y_onehot shape (K,N) , z shape (K,N)

        loss = loss / N

        # 加上 L2 正则化
        if use_reg == 2:

            s = 0
            for W in W_list:  # 遍历所的 W, 并求平方和
                s += np.sum(np.square(W))

            loss += (reg_lambda / (2 * N)) * s

        elif use_reg == 1:  # 加上 L1 正则化

            s = 0
            for W in W_list:
                s += np.sum(np.abs(W))

            loss += (reg_alpha / (2 * N)) * s

        return loss

    def __backwoard_last_layer(self, N, W, a, y_onehot, a_prev, d_prev=None,
                               reg_alpha=0.5, reg_lambda=0.5, use_reg=0,
                               keep_prob=1.0, use_dropout=False
                               ):
        """
        最后一层(输出层)的反向传播

        :param N:
        :param W:
        :param a:
        :param y_onehot:
        :param a_prev:
        :param d_prev:
        :param reg_alpha: L1 正则化参数
        :param reg_lambda: L2 正则化参数
        :param use_reg: 正则化类型选择,
                        2: L2 正则化
                        1: L1 正则化
                        0: 不使用正则化

        :param keep_prob:  dropout 正则化中, 每一层的神经元有效的比例  (默认 0.8)
        :param use_dropout: 开启 dropout 正则化 (默认 False 不开启)

        :return:
        """

        grad_z = a - y_onehot  # shape:(K,N)
        # a shape:(K,N) , y_onehot shape:(K,N)

        grad_W = np.dot(grad_z, a_prev.T)
        # grad_z shape:(K,N) , a_prev.T shape:(N,n_h)
        grad_W = grad_W / N

        grad_b = np.sum(grad_z, axis=1, keepdims=True)  # shape:(K,1)
        grad_b = grad_b / N

        grad_a = np.dot(W.T, grad_z)
        #  W.T shape:(n_h, K) ,  grad_z shape:(K,1)

        # 考虑正则化项
        # 加上 L2 正则化
        if use_reg == 2:
            grad_W += (reg_lambda / N) * W

        # 加上 L1 正则化
        elif use_reg == 1:
            I = np.ones(np.shape(W))
            I[W < 0] = -1
            grad_W += (reg_alpha / N) * I

        # dropout 正则化
        if use_dropout:
            grad_a[d_prev > keep_prob] = 0  # grad_a 以一定的概率失活

        return grad_z, grad_W, grad_b, grad_a

    def __backwoard_layer(self, N, W, grad_a, z, a_prev, d_prev,
                          gama=None,beta=None,cache_bn=None,
                          activation='sigmoid',
                          reg_alpha=0.5, reg_lambda=0.5, use_reg=0,
                          keep_prob=1.0, use_dropout=False,
                          use_batchnorm=False):
        """
        单层的反向传播

        :param N: 样本个数

        :param W: shape (n_current, n_prev)
                n_current -当前层向量的维度
                n_prev    -上一层向量的维度
        :param grad_a: shape(n_current,N)
        :param z: shape (n_current,N)
        :param a_prev: shape (n_prev,N)

        :param d_prev: shape (n_prev,N)

        :param gama:
        :param beta:
        :param cache_bn: BN 正向传播过程中的中间变量

        :param activation: 选择的激活函数, 要与前向传播对于此层的设置相匹配

        :param reg_alpha: L1 正则化参数
        :param reg_lambda: L2 正则化参数
        :param use_reg: 正则化类型选择,
                        2: L2 正则化
                        1: L1 正则化
                        0: 不使用正则化

        :param keep_prob:  dropout 正则化中, 每一层的神经元有效的比例  (默认 0.8)
        :param use_dropout: 开启 dropout 正则化 (默认 False 不开启)

        :param use_batchnorm: 开启 batch normalizaion
                              (默认 False 不开启)

        :return:
        """
        grad_activation = None

        if activation == 'sigmoid':
            grad_activation = Activation.grad_sigmoid(z)  # shape: (n_current,N)

        elif activation == 'relu':
            grad_activation = Activation.grad_relu(z)  # shape: (n_current,N)


        grad_z_ba = grad_a * grad_activation  # shape: (n_current,N)

        if use_batchnorm: # 开启 batch normalizaion
            grad_z,grad_gama,grad_beta = BatchNormalization.batchnorm_bakward(N,gama,beta,grad_z_ba,cache_bn)

        else: # 关闭 batch normalizaion
            grad_z = grad_z_ba
            grad_gama = 0
            grad_beta = 0

        grad_W = np.dot(grad_z, a_prev.T)  #
        # grad_z shape:(n_current,N) ,a_prev.T shape:(N,n_prev)
        grad_W = grad_W / N

        grad_b = np.sum(grad_z, axis=1, keepdims=True)  # shape:(n_current,1)
        grad_b = grad_b / N

        grad_a = np.dot(W.T, grad_z)  # shape: (n_prev, N)
        # W.T shape:(n_prev, n_current) ,  grad_z shape:(n_current,N)

        # 考虑正则化项

        # 加上 L2 正则化
        if use_reg == 2:
            grad_W += (reg_lambda / N) * W

        # 加上 L1 正则化
        elif use_reg == 1:
            I = np.ones(np.shape(W))
            I[W < 0] = -1
            grad_W += (reg_alpha / N) * I

        # dropout 正则化

        if use_dropout:
            grad_a[d_prev > keep_prob] = 0  # grad_a 以一定的概率失活


        return grad_z, grad_W, grad_b, grad_a,grad_gama,grad_beta

    def backwoard_propagation(self, activation,
                                    parameters,
                                    z_list, a_list, d_list,
                                    cache_bn_list,
                                    X, y_onehot,
                                    reg_alpha=0.5,
                                    reg_lambda=0.5,
                                    use_reg=0,
                                    keep_prob=1.0,
                                    use_dropout=False,
                                    bn_param = None,
                                    use_batchnorm = False):
        """
        计算反向传播

        :param activation: 输入层和隐藏层的激活函数
                   'sigmoid'
                   'relu'
        :param parameters:模型参数
        :param z_list:
        :param a_list:
        :param d_list:
        :param cache_bn_list: BN 正向传播过程中的中间变量
        :param X:样本特征 shape (N,m) N- 样本个数 , m-特征维度
        :param y_onehot: 样本标签

        :param reg_alpha: L1 正则化参数 (默认 0.5)
        :param reg_lambda: L2 正则化参数 (默认 0.5 )
        :param use_reg: 正则化类型选择,
                        2: L2 正则化
                        1: L1 正则化
                        0: 不使用正则化(默认)

        :param keep_prob:  dropout 正则化中, 每一层的神经元有效的比例  (默认 1.0 )
        :param use_dropout: 开启 dropout 正则化 (默认 False 不开启)

        :param bn_param:  batch normalizaion 的超参数
        :param use_batchnorm: 开启 batch normalizaion
                              (默认 False 不开启)

        :return:
        """
        N = np.shape(X)[0]

        W_list = parameters['W']
        b_list = parameters['b']

        gama_list = parameters['gama']
        beta_list = parameters['beta']

        L = len(W_list)  # MLP 的层数 L=2

        grad_W_list = []
        grad_b_list = []
        grad_a_list = []
        grad_z_list = []  # delta

        grad_gama_list=[]
        grad_beta_list=[]

        # 输出层
        l = L - 1
        grad_z, grad_W, grad_b, grad_a = self.__backwoard_last_layer(N, W=W_list[l], a=a_list[l], y_onehot=y_onehot,
                                                                     a_prev=a_list[l - 1], d_prev=d_list[l - 1],
                                                                     reg_alpha=reg_alpha, reg_lambda=reg_lambda,use_reg=use_reg,
                                                                     keep_prob=keep_prob, use_dropout=use_dropout)
                                                                    # 输出层未使用 BN

        grad_W_list.append(grad_W)
        grad_b_list.append(grad_b)
        grad_a_list.append(grad_a)
        grad_z_list.append(grad_z)

        # 输出层未使用 BN, 梯度记为0
        grad_gama_list.append(0)
        grad_beta_list.append(0)

        # 从后往前遍历 隐藏层  l = L-2 ,...,1
        for l in range(L - 2, 0, -1):
            grad_z, grad_W, grad_b, grad_a,grad_gama,grad_beta = self.__backwoard_layer(N, W=W_list[l], grad_a=grad_a, z=z_list[l],a_prev=a_list[l - 1],
                                                                                        d_prev=d_list[l - 1],
                                                                                        gama = gama_list[l], beta=beta_list[l],cache_bn = cache_bn_list[l],
                                                                                        activation = activation,
                                                                                        reg_alpha=reg_alpha, reg_lambda=reg_lambda,use_reg=use_reg,
                                                                                        keep_prob=keep_prob,use_dropout=use_dropout,
                                                                                        use_batchnorm = use_batchnorm
                                                                                        )

            grad_W_list.append(grad_W)
            grad_b_list.append(grad_b)
            grad_a_list.append(grad_a)
            grad_z_list.append(grad_z)

            grad_gama_list.append(grad_gama)
            grad_beta_list.append(grad_beta)

        # 输入层
        l = 0
        grad_z, grad_W, grad_b, grad_a,grad_gama,grad_beta = self.__backwoard_layer(N, W=W_list[l], grad_a=grad_a, z=z_list[l], a_prev=X.T,
                                                                d_prev=None,
                                                                gama = gama_list[l], beta=beta_list[l],cache_bn = cache_bn_list[l],
                                                                activation=activation,
                                                                reg_alpha=reg_alpha, reg_lambda=reg_lambda, use_reg=use_reg,
                                                                use_dropout=False,
                                                                use_batchnorm=use_batchnorm
                                                                ) # 因为前面已经没有层了, 输入层 无需计算 grad_a, 也不用考虑dropout

        grad_W_list.append(grad_W)
        grad_b_list.append(grad_b)
        grad_a_list.append(grad_a)
        grad_z_list.append(grad_z)

        grad_gama_list.append(grad_gama)
        grad_beta_list.append(grad_beta)

        # 梯度列表全部反向, 接下来都要从前向后遍历
        grad_W_list.reverse()
        grad_b_list.reverse()
        grad_a_list.reverse()
        grad_z_list.reverse()

        grad_gama_list.reverse()
        grad_beta_list.reverse()

        return grad_W_list, grad_b_list,grad_gama_list,grad_beta_list


    def fit(self, X, y, layers_dims,init_mode='Random', learning_rate=1.0, max_iter=500, mini_batch_size=64, optimize_mode='BGD',
            print_log=True, print_log_step=100):
        """
        训练模型

        :param X:
        :param y:
        :param layers_dims: 从前往后 MLP各个层的向量维度

        layers_dims= [m,h,K]
        m: 输入层向量的维度
        h: 隐藏层向量的维度
        K: 输出层向量的维度

        :param init_mode: 初始化模型参数的模式
                     'Zero'  初始化为 0
                     'Random' 随机初始化 (默认)
                     'Xavier' 配合 Relu 使用的一种随机初始化

        :param learning_rate: 学习率
        :param max_iter: 迭代次数

        :param mini_batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 64)
        :param optimize_mode: 优化算法
                  'BGD' : 批量梯度下降 BGD (默认)
                  'MinBatch': min-Batch梯度下降
                  'Momentum': 带动量的 Mini-batch 梯度下降
                  'Adam': Adam Mini-batch 梯度下降

        :param print_log: 是否打印训练误差
        :param print_log_step: 打印训练误差的步长, 训练多少个 epcho 就打印一次训练误差
        :return:
        """

        # y=y.flatten() # y 的维度转换为 (N,)

        N, m = np.shape(X)  # N 个样本, m 个特征

        print('train data num:{} , feature dimension:{}'.format(N, m))

        print(' K={} classifier '.format(self.K))

        assert self.K == len(set(y))  # 设置的分类类别必须和训练数据的标签的类别相同

        # 模型参数初始化
        class_name = init_mode + 'Initializer' #  'Random' + 'Initializer' = 'RandomInitializer'

        # Initializer = getattr(importlib.import_module('lib.Initializer_xrh'), class_name)

        if not hasattr( initial , class_name):
            raise ValueError('Invalid init_mode "%s"' % init_mode)
        Initializer = getattr(initial, class_name)

        initializer = Initializer()
        parameters = initializer.initialize_parameters(layers_dims)

        # 将标签 y one-hot 化, shape: (K,N)
        y_onehot = (y == np.array(range(self.K)).reshape(-1, 1)).astype(
            np.int8)

        loss_list = []  # 记录每次梯度下降的损失, 然后可以画出模型的学习曲线

        # 配置优化算法
        class_name = optimize_mode + 'Optimizer'  # 'BGD' + 'Optimizer' = 'BGDOptimizer'
        if not hasattr( optim , class_name):
            raise ValueError('Invalid optimize_mode "%s"' % optimize_mode)
        Optimizer = getattr(optim, class_name)

        # class_name = optimize_mode + 'Optimizer' #  'BGD' + 'Optimizer' = 'BGDOptimizer'
        # Optimizer = getattr(importlib.import_module('lib.Optimizer_xrh'), class_name)

        optimizer = Optimizer(parameters)

        count_gd = 1  # 计数器, 记录梯度下降的次数

        for epoch in range(max_iter):

            # 得到所有的批量(batch)
            batches = optimizer.get_batches(X, y_onehot, mini_batch_size=mini_batch_size)

            for batch in batches:  # 遍历所有的 batch

                X_batch, y_onehot_batch = batch

                # 把一个batch的数据喂给模型, 进行正向和反向传播, 并更新参数
                z_list, a_list, d_list,cache_bn_list, loss = self.forwoard_propagation(parameters=parameters,
                                                                         X=X_batch,
                                                                         y_onehot=y_onehot_batch,
                                                                         activation=self.activation,
                                                                         reg_alpha=self.reg_alpha,
                                                                         reg_lambda=self.reg_lambda,
                                                                         use_reg=self.use_reg,
                                                                         keep_prob=self.keep_prob,
                                                                         use_dropout=self.use_dropout,
                                                                         bn_param = self.bn_param,
                                                                         use_batchnorm = self.use_batchnorm,
                                                                         mode = 'training'
                                                                                       )

                grad_W_list, grad_b_list,grad_gama_list,grad_beta_list = self.backwoard_propagation(activation=self.activation,
                                                                      parameters=parameters,
                                                                      z_list=z_list,
                                                                      a_list=a_list,
                                                                      d_list=d_list,
                                                                      cache_bn_list=cache_bn_list,
                                                                      X=X_batch,
                                                                      y_onehot=y_onehot_batch,
                                                                      reg_alpha=self.reg_alpha,
                                                                      reg_lambda=self.reg_lambda,
                                                                      use_reg=self.use_reg,
                                                                      keep_prob=self.keep_prob,
                                                                      use_dropout=self.use_dropout,
                                                                      bn_param=self.bn_param,
                                                                      use_batchnorm=self.use_batchnorm,
                                                                      )

                parameters = optimizer.update_parameters(learning_rate, parameters, grad_W_list, grad_b_list,grad_gama_list,grad_beta_list,t=count_gd)

                count_gd += 1  # 梯度下降的次数 +1

                # if count_gd % 1 == 0: # 每 1 次梯度下降记录到 loss_list 中
                loss_list.append(loss)

            # Print the cost every 10 epoch
            if print_log and epoch % print_log_step == 0:
                print('epcho: {} , loss:{}'.format(epoch, loss))



        # 保存训练参数, 用于推理
        self.parameters = parameters

        self.save(self.model_path)

        return loss_list

    def save(self, model_dir):
        """
        保存训练好的 MLP 模型

        :param train_data_dir:
        :return:
        """

        save_dict = {}

        save_dict['K'] = self.K
        save_dict['activation'] = self.activation
        save_dict['reg_alpha'] = self.reg_alpha
        save_dict['reg_lambda'] = self.reg_lambda
        save_dict['use_reg'] = self.use_reg

        save_dict['keep_prob'] = self.keep_prob
        save_dict['use_dropout'] = self.use_dropout

        save_dict['bn_param'] = self.bn_param
        save_dict['use_batchnorm'] = self.use_batchnorm

        save_dict['parameters'] = self.parameters

        with open(model_dir, 'wb') as f:
            pickle.dump(save_dict, f)

        print("Save model successful!")

    def load(self, file_path):
        """
        读取预训练的 MLP 模型

        :param file_path:
        :return:
        """

        with open(file_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.K = save_dict['K']
        self.activation = save_dict['activation']
        self.reg_alpha = save_dict['reg_alpha']
        self.reg_lambda = save_dict['reg_lambda']
        self.use_reg = save_dict['use_reg']

        self.keep_prob = save_dict['keep_prob']
        self.use_dropout = save_dict['use_dropout']

        self.bn_param = save_dict['bn_param']
        self.use_batchnorm = save_dict['use_batchnorm']

        self.parameters = save_dict['parameters']

        print("Load model successful!")

    def predict_prob(self, X, parameters):
        """
        推理测试数据集, 返回样本的分值 (概率值),
        我们可以利用此分值得出 P-R 曲线

        :param X:
        :return:
        """
        z_list, a_list, _, _, _ = self.forwoard_propagation(parameters, X, activation=self.activation,
                                                         bn_param=self.bn_param,
                                                         use_batchnorm=self.use_batchnorm,
                                                         mode='inference')  # 前向传播

        p = a_list[-1]

        return p

    def predict(self, X):
        """
        推理测试数据集，返回样本标签

        :param X:
        :param threshold:判断样本标签正负的阈值
        :return:
        """

        parameters = self.parameters

        P = self.predict_prob(X, parameters)  # p:shape(N,1)
        # X: shape(N,m)

        res = np.argmax(P, axis=0)  # axis=0 干掉第0个维度, shape: (N,)

        return res


class Test:

    def loadData_2classification(self, fileName, n=1000):
        """
        加载文件

        将 数据集 的标签 转换为 二分类的标签

        :param fileName:要加载的文件路径
        :param n: 返回的数据集的规模
        :return: 数据集和标签集

        """

        # 存放数据及标记
        dataArr = []
        labelArr = []
        # 读取文件
        fr = open(fileName)

        cnt = 0  # 计数器

        # 遍历文件中的每一行
        for line in fr.readlines():

            if cnt == n:
                break

            # 获取当前行，并按“，”切割成字段放入列表中
            # strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
            # split：按照指定的字符将字符串切割成每个字段，返回列表形式
            curLine = line.strip().split(',')
            # 将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
            # 在放入的同时将原先字符串形式的数据转换为整型
            # 此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
            dataArr.append([int(int(num) > 128) for num in curLine[1:]])

            # 将标记信息放入标记集中
            # 转换成二分类任务
            # 标签0设置为1，反之为0
            if int(curLine[0]) == 0:
                labelArr.append(1)
            else:
                labelArr.append(0)

            cnt += 1

        fr.close()

        # 返回数据集和标记
        return dataArr, labelArr

    def test_Mnist_dataset_2classification(self, n_train, n_test):
        """
        将 Mnist (手写数字) 数据集 转变为 二分类 数据集

        测试 MLP

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData_2classification('../dataset/Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

        N, m = np.shape(trainDataArr)  # N 个样本, m 个特征

        # 开始时间
        print('start training model....')
        start = time.time()

        clf = MLP_2Classifier(reg_lambda=0.7, use_reg=2)
        clf.fit(X=trainDataArr, y=trainLabelArr, layers_dims=[m, 50, 10, 1], learning_rate=1.0, max_iter=500)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData_2classification('../dataset/Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        y_pred = clf.predict(testDataArr)

        print('test dataset accuracy: {} '.format(accuracy_score(y_pred, testLabelArr)))

        K = 2

        # 查看每一种类别 的评价指标
        print('print the classification report: ')

        report = classification_report(testLabelArr, y_pred)

        print(report)

        # 打印混淆矩阵
        print('print the confusion matrix')

        confusion = confusion_matrix(testLabelArr, y_pred)
        print(confusion)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=list(range(K)))
        disp.plot()

        plt.show()

    def test_planar_dataset_2classification(self):
        """
        对于二分类问题, 检查损失(loss)是否随着迭代而不断降低

        :return:
        """

        X, Y = load_planar_dataset()

        X = X.T
        Y = Y.T.flatten()

        # Visualize the data:
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)

        # plt.show() # 展示图片(使用 jupyter notebook 不用指明需要显示)

        shape_X = X.shape
        shape_Y = Y.shape

        N = shape_X[1]  # training set size

        print('The shape of X is: ' + str(shape_X))
        print('The shape of Y is: ' + str(shape_Y))
        print('I have m = %d training examples!' % (N))

        clf = MLP_2Classifier(use_reg=0)
        clf.fit(X=X, y=Y, layers_dims=[2, 4, 1], learning_rate=1, max_iter=10000)

    def loadData(self, fileName, n=1000, binaryzation=True):
        """
        加载文件

        加载文件
        :param fileName:要加载的文件路径
        :param n: 返回的数据集的规模
        :param binaryzation: 对样本特征进行二值化处理(大于128的转换成1，小于的转换成0);
                             若进行了二值化处理, 则后续无需再对特征进行归一化(normalization)

        :return: 数据集和标签集
        """

        # 存放数据及标记
        dataArr = []
        labelArr = []
        # 读取文件
        fr = open(fileName)

        cnt = 0  # 计数器

        # 遍历文件中的每一行
        for line in fr.readlines():

            if cnt == n:
                break

            # 获取当前行，并按“，”切割成字段放入列表中
            # strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
            # split：按照指定的字符将字符串切割成每个字段，返回列表形式
            curLine = line.strip().split(',')
            # 将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
            # 在放入的同时将原先字符串形式的数据转换为整型

            if binaryzation:  # 此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
                dataArr.append([int(int(num) > 128) for num in curLine[1:]])
            else:
                dataArr.append([int(num) for num in curLine[1:]])

            # 将标记信息放入标记集中
            # 放入的同时将标记转换为整型
            labelArr.append(int(curLine[0]))

            cnt += 1

        fr.close()

        # 返回数据集和标记
        return dataArr, labelArr

    def test_Mnist_dataset(self, n_train, n_test):
        """
        利用 Mnist 数据集 测试

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        binaryzation = False  # 是否对样本特征进行二值化处理

        # 获取训练集
        trainDataList, trainLabelList = self.loadData('../dataset/Mnist/mnist_train.csv', n=n_train,
                                                      binaryzation=binaryzation)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

        K = 10  # 10分类

        # 开始时间
        print('start training model....')

        start = time.time()

        # X = Normalizer.tow_norm_normalize(trainDataArr) # 二范数归一化

        X = trainDataArr
        y = trainLabelArr

        clf = MLP_MultiClassifier(K=K,
                                  activation='relu',
                                  reg_lambda=0.1,
                                  use_reg=0,
                                  keep_prob=0.8,
                                  use_dropout=False,
                                  use_batchnorm=True,
                                  model_path='model/Mnist.model',
                                  use_pre_train=False)

        loss_list = clf.fit(X=X, y=y, init_mode='Xavier',layers_dims=[784,100,100,100,100,10], mini_batch_size=256, optimize_mode='Adam',
                            max_iter=50, learning_rate=0.01, print_log=True, print_log_step=10)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 打印模型的学习曲线(损失函数的下降曲线)
        plt.plot(loss_list)
        plt.ylabel('loss')
        plt.xlabel('gradient descent times')
        plt.title("Learning rate = " + str(0.5))
        plt.show()

        # 获取测试集
        testDataList, testLabelList = self.loadData('../dataset/Mnist/mnist_test.csv', n=n_test,
                                                    binaryzation=binaryzation)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        # X_test = normalizeRow(testDataArr)
        X_test = testDataArr
        y_test = testLabelArr

        # 读取预训练好的模型, 并进行推理
        clf_pre_train = MLP_MultiClassifier(
            model_path='model/Mnist.model',
            use_pre_train=True)

        y_predict = clf_pre_train.predict(X_test)

        print('test accuracy :', accuracy_score(y_predict, y_test))

        # 对比训练集和测试集的 accuracy, 判断模型是否出现过拟合
        y_predict_train = clf_pre_train.predict(X)
        print('train accuracy :', accuracy_score(y_predict_train, y))

        # 查看每一种类别 的评价指标
        print('print the classification report: ')

        report = classification_report(y_test, y_predict)

        print(report)

        # 打印混淆矩阵
        print('print the confusion matrix')

        confusion = confusion_matrix(y_test, y_predict)
        print(confusion)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=list(range(K)))
        disp.plot()

        plt.show()

    def test_iris_dataset(self):

        # 使用iris数据集，其中有三个分类， y的取值为0,1，2
        X, y = datasets.load_iris(True)  # 包括150行记录
        # 将数据集一分为二，训练数据占80%，测试数据占20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)

        K = 3
        N, m = np.shape(X)  # N 个样本, m 个特征

        clf = MLP_MultiClassifier(K=K, activation='sigmoid', reg_lambda=0.7, use_reg=0, use_pre_train=False)
        clf.fit(X=X_train, y=y_train, layers_dims=[m, 50, K], learning_rate=1.0, max_iter=500)

        y_predict = clf.predict(X_test)

        print('test accuracy :', accuracy_score(y_predict, y_test))

        # 查看每一种类别 的评价指标
        print('print the classification report')

        report = classification_report(y_test, y_predict)

        print(report)

        # 打印混淆矩阵
        print('print the confusion matrix')

        confusion = confusion_matrix(y_test, y_predict)
        print(confusion)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=list(range(K)))
        disp.plot()

        plt.show()  # 显示图片


if __name__ == '__main__':

    np.random.seed(0)  # we set up a seed so that your output matches ours although the initialization is random.

    test = Test()

    # test.test_planar_dataset()

    # test.test_Mnist_dataset_2classification(60000, 10000)

    test.test_Mnist_dataset(60000, 10000)

    # test.test_iris_dataset()
