#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

import time
from deprecated import deprecated

from sklearn import datasets

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression


def normalize(X):
    """
    对所有特征进行归一化

    :param X:
    :return:
    """

    # N, m = np.shape(X)  # N 个样本, m 个特征

    mu = np.mean(X,axis=0) # 每一个特征的均值 shape:(m,)

    s = np.std(X, axis=0) # 每一个特征的标准差 shape:(m,)

    X = (X-mu)/s

    return X

class LinerReg:

    """
    线性回归

    1.通过 数值优化方法 梯度下降 找到最优解

    2.直接找到解析解

    Author: xrh
    Date: 2021-07-04

    ref:

    test0: 回归 任务
    数据集：boston房价 数据集
    参数:  max_iter=100,learning_rate=0.1
    训练集数量：455
    测试集数量：51
    测试集的 MSE： 16.9
    模型训练时长：0.4s

    """

    def __init__(self, reg_alpha=0.1 , reg_lambda=0.1, use_reg=2):
        """

        :param reg_alpha: L1 正则化参数
        :param reg_lambda: L2 正则化参数
        :param use_reg: 正则化类型选择, 2: L2 正则化 ; 1: L1 正则化 ; 0: 不使用正则化
        """

        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.use_reg=use_reg

        # 模型的参数
        self.W=None
        self.b = None

    def lossfunc(self,W,b,X,y):
        """
        计算 带正则化的损失函数

        :param W:
        :param b:
        :param X:
        :param y:
        :return:
        """
        loss =0

        N, m = np.shape(X)  # N 个样本, m 个特征

        for i in range(N): # 遍历 N 个样本

            loss += np.square( np.dot( W, X[i] )+b -y[i] )

        loss = loss/(2*N)

        # 加上 L2 正则化
        if self.use_reg ==2:
            loss += (self.reg_lambda/2) * np.sum( np.square(W) )

        elif self.use_reg ==1:# 加上 L1 正则化
            loss += self.reg_alpha * np.sum(np.abs(W))


        return loss

    def gradient(self,W,b,X,y):
        """
        计算梯度

        :param w:
        :param X:
        :param y:
        :return:
        """
        N, m = np.shape(X)  # N 个样本, m 个特征

        grad_W = 0
        grad_b = 0

        for i in range(N):  # 遍历 N 个样本

            diff = np.dot(W,X[i])+b - y[i]
            grad_W += diff * X[i]
            grad_b += diff

        grad_W = grad_W / N
        grad_b = grad_b / N

        # 加上 L2 正则化
        if self.use_reg ==2:
            grad_W += self.reg_lambda*W

        elif self.use_reg ==1:# 加上 L1 正则化
            I = np.ones(m)
            I[W<0]=-1
            grad_W += self.reg_alpha * I

        return grad_W,grad_b

    def fit(self, X ,y,learning_rate=0.1, max_iter=100,use_BGD=True):
        """
        训练模型

        1.必须对输入数据做归一化, 否则 计算梯度会发生向上溢出

        :param X:
        :param y:
        :param learning_rate: 学习率
        :param max_iter: 迭代次数
        :param use_BGD: 是否使用梯度下降求解
        :return:
        """

        N,m = np.shape(X) # N 个样本, m 个特征

        print('train data num:{}'.format(N))

        W = np.zeros(m) # 模型参数初始化
        b = 0

        if use_BGD: # 使用 批量梯度下降 求解

            for epoch in range(max_iter):

                loss = self.lossfunc(W,b,X,y)

                grad_w,grad_b = self.gradient(W,b,X,y)

                W -= learning_rate*grad_w
                b -= learning_rate * grad_b

                print('epcho: {} , loss:{}'.format(epoch,loss))

        else: # 直接求解析解
            # 懒得推公式, 忽略偏置b, 导致模型无法收敛,
            # 解决: 对 y 进行归一化

            X_square = np.dot(X.T, X)

            part1 = np.linalg.inv(X_square + X_square.T + self.reg_lambda * np.identity(m) )
            part2 = np.dot(X.T,y)

            W = np.dot(part1,part2)


        # 保存训练参数
        self.W = W
        self.b = b

    def __predict(self,x):
        """
        预测 单个样本的标签值

        :param x:
        :return:
        """

        return np.dot(self.W,x)+self.b

    def predict(self,X):
        """
        预测 测试 数据集，返回预测结果

        :param X:
        :return:
        """

        return np.array([self.__predict(x) for x in X])

class Test:


    def test_regress_dataset(self):
        """
        利用 boston房价 数据集
        测试  GBDT  回归

        :return:
        """

        # 加载sklearn自带的波士顿房价数据集
        dataset = load_boston()

        # 提取特征数据和目标数据
        X = dataset.data
        y = dataset.target

        X = normalize(X) # 回归问题 特征X 必须做归一化, 否则梯度会出现溢出

        # y 和 特征X 的差距较大, 我们发现线性回归模型不收敛, 解决方案:
        #  M1.可以对 y 进行归一化
        #  M2.在 h(x) 中加入偏置项 b, 一般线性回归都要考虑偏置项
        y = normalize(y)

        # 将数据集以9:1的比例随机分为训练集和测试集，为了重现随机分配设置随机种子，即random_state参数
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=188)


        linreg = LinearRegression()
        linreg.fit(X_train, y_train)

        start = time.time()
        print('start create model')

        lr = LinerReg(use_reg=0)

        # lr = LinerReg(reg_alpha=0.5,use_reg=1)

        # lr = LinerReg(reg_lambda=0.1,use_reg=2)

        # lr.fit(X_train, y_train, max_iter=50,learning_rate=0.1)

        lr.fit(X_train, y_train, use_BGD=False)

        #  当 L1 正则化系数 reg_alpha=1 时, 可以观察到特征的权重 W 中出现很小的值,
        #  说明发生了 特征选择的作用;
        #  W:  [ 4.54462493e-04  9.67745714e-02 -1.17590863e-01  1.05890221e-01
        #    -5.46992763e-02  2.77627850e+00  5.45168872e-02 -1.20214803e-01
        #    7.06164990e-03 -1.22548461e-01 -1.32898216e+00  1.09109089e-01
        #   -3.23601631e+00]
        #    但是另一方面,  我们发现在训练时损失在很早的 epcho就不再下降, 说明模型训练并没有出现过拟合, 反而是欠拟合的
        print('W: {}'.format(lr.W))

        print(' model complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        y_pred = linreg.predict(X_test)

        y_pred_test = lr.predict(X_test)

        print('by sklearn , the squared_error:', mean_squared_error(y_test, y_pred))  # the squared_error: 8.46788133276128

        print('by xrh , the squared_error:', mean_squared_error(y_test, y_pred_test))  #


if __name__ == '__main__':

    test = Test()

    test.test_regress_dataset()
