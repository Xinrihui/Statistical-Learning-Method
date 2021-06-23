#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import random
import numpy as np
from sklearn import datasets,cluster
import matplotlib.pyplot as plt

import time


def loadGaussianData(N,mu_arr, sigma_arr, alpha_arr):
    """
    初始化数据集
    这里通过服从高斯分布的随机函数来 构造数据集

    :param N: 生成数据集的样本个数
    :param mu: 高斯 的均值
    :param sigma: 高斯 的方差
    :param alpha: 高斯 的系数
    :return: 混合了两个高斯分布的数据

    :return:
    """
    assert sum(alpha_arr)==1.0 # alpha系数 求和必须为1

    # np.random.seed(0) # 设置随机数种子, 保证每次运行代码都生成一样的数据

    K=len(mu_arr)

    #初始化总数据集
    #多个高斯分布的数据混合后会放在该数据集中返回
    dataSet = []

    for i in range(K):

        # 初始化第一个高斯分布，生成数据，数据长度为length * alpha系数，以此来
        # 满足alpha的作用
        data = np.random.normal(mu_arr[i], sigma_arr[i], int(N * alpha_arr[i]))

        dataSet.extend(data)

    #对总的数据集进行打乱（其实不打乱也没事，只不过打乱一下直观上让人感觉已经混合了
    # 读者可以将下面这句话屏蔽以后看看效果是否有差别）
    # random.shuffle(dataSet)

    #返回伪造好的数据集
    return dataSet


class GMM_1D:
    """
    高斯混合模型
    (基于一元高斯分布)

    ref:
    《统计学习方法 第二版》李航

    Author: xrh
    Date: 2021-06-18



    """

    def __init__(self, K=2, max_iter=10 ):

        # 高斯分布的个数
        self.K = K

        # EM 的迭代次数
        self.max_iter = max_iter

    def calcGauss(self,X, mu_arr, sigma_arr):
        """
        依据 公式(9.25) 计算 特征的一元高斯分布

        :param X:
        :param mu_arr:
        :param sigma_arr:
        :return:
        """
        N  = np.shape(X)[0]

        P = np.zeros((N,self.K))

        for k in range(self.K):

            s1 = np.exp( - (( X - mu_arr[k] )**2)/(2* sigma_arr[k]**2 ) )  # 公式(9.25) 分子部分
            s2 = (math.sqrt(2*math.pi)*sigma_arr[k]) # 公式(9.25) 分母部分

            P[ :,k]= s1 / s2

        return P


    def calc_gama(self,X,mu_arr,sigma_arr,alpha_arr):
        """
        计算 gama, 即 P( 隐藏变量 | 观测变量)

        :param X:
        :param mu_arr:
        :param sigma_arr:
        :param alpha_arr:
        :param gama:
        :return:
        """
        N  = np.shape(X)[0]

        P = self.calcGauss(X, mu_arr, sigma_arr) # shape: (N, K)

        gama = np.zeros((N, self.K))

        s1= np.zeros((N,self.K))

        for k in range(self.K):
            s1[:,k] = alpha_arr[k]*P[:,k] # 公式 分子部分

        s2=np.sum(s1, axis=1) # 公式 分母部分,shape: (N, )

        for k in range(self.K):
            gama[:,k]=  s1[:,k] / s2

        return gama

    def calc_mu(self,X,gama):
        """
        根据公式 (9.30) 计算均值

        :param X:
        :param gama:
        :param mu_arr:
        :return:
        """
        N  = np.shape(X)[0]

        s1= np.zeros( self.K )

        for k in range(self.K):
            s1[k] = np.dot(gama[:,k] ,X)# 公式 (9.30) 分子部分

        s2 = np.sum(gama,axis=0)  # 公式 (9.30) 分母部分

        mu_arr =  s1/s2

        return mu_arr


    def calc_sigma(self,X,gama,mu_arr):
        """
        根据公式 (9.31) 计算方差

        :param X:
        :param gama:
        :param mu_arr:
        :return:
        """
        N = np.shape(X)[0]

        # s1 = np.zeros((N, self.K))
        s1 = np.zeros(self.K)

        for k in range(self.K):
            s1[k] = np.dot(gama[:,k] , (X-mu_arr[k])**2)  # 公式 (9.31) 分子部分

        s2 = np.sum(gama, axis=0)  #shape:(K)  公式 (9.30) 分母部分

        sigma_arr = np.sqrt(s1 / s2)

        return sigma_arr

    def calc_alpha(self, X,gama):
        """
        根据公式 (9.32) 计算隐变量的概率分布

        :param X:
        :param gama:
        :return:
        """
        N  = np.shape(X)[0]

        s1 = np.sum(gama, axis=0)  # 公式 (9.32) 分子部分

        alpha_arr= s1/N

        return alpha_arr

    def init_parameter(self,mode=0):
        """
        初始化 GMM 的参数

        :param mode:
        :return:
        """

        if mode==1: # 使用推荐的固定参数
            mu_arr = np.array([0, 1])
            sigma_arr = np.array([1, 1])
            alpha_arr = np.array([0.5, 0.5])

        else: # 使用随机参数(效果不佳), 说明 EM 算法不一定能收敛到全局最优点

            # K 个高斯分布的均值
            mu_arr = np.random.rand(self.K) # 生成[0.0, 1.0)之间的随机浮点数数组,数组长度为K

            # K 个高斯分布的方差
            sigma_arr = 10*np.random.rand(self.K) # 生成[0.0, 10.0)之间的随机浮点数数组,数组长度为K

            # 隐变量 z 的概率分布
            temp = np.random.rand(self.K)
            s= np.sum(temp)
            alpha_arr = temp / s # 因为是概率分布, 所以要进行归一化


        return  mu_arr,sigma_arr,alpha_arr

    def fit(self, X,print_log=False):
        """
        算法 9.2 高斯混合模型参数估计的 EM算法

        :param X:
        :return:
        """
        N = np.shape(X)[0]

        mu_arr,sigma_arr,alpha_arr = self.init_parameter(mode=1)

        for epoch in range(self.max_iter):

            print('epoch: {}'.format(epoch))

            # E step
            gama=self.calc_gama(X,mu_arr,sigma_arr,alpha_arr)

            # M step
            sigma_arr = self.calc_sigma(X,gama,mu_arr) # mu_arr 要使用上一次的
            mu_arr = self.calc_mu(X, gama)
            alpha_arr = self.calc_alpha(X,gama)

            if print_log:
                print('alpha: {} , mu: {}, sigma:{} '.format(
                    alpha_arr, mu_arr, sigma_arr))


        return mu_arr,sigma_arr,alpha_arr



class Test:

    def test_Gauss_dataset(self):

        # 设置两个高斯模型进行混合，这里是初始化两个模型各自的参数
        # 见“9.3 EM算法在高斯混合模型学习中的应用”
        # alpha是“9.3.1 高斯混合模型” 定义9.2中的系数α
        # mu是均值μ
        # sigma是方差σ
        # 在设置上两个alpha的和必须为1，其他没有什么具体要求，符合高斯定义就可以
        alpha = [0.3, 0.7]
        mu = [-2,0.5]
        sigma = [0.5,1]

        # 打印设置的参数
        print('---------------------------')
        print('the Parameters set is:')
        print('alpha: {} , mu: {}, sigma:{} '.format(
            alpha, mu, sigma ))

        # 初始化数据集
        N = 1000
        X = loadGaussianData(N,mu, sigma, alpha)

        start = time.time()  # 保存开始时间

        K=2
        gmm = GMM_1D(K=K,max_iter=500)
        mu_arr,sigma_arr,alpha_arr = gmm.fit(X)

        # 打印参数预测结果
        print('----------------------------')
        print('the Parameters predict is:')
        print('alpha: {} , mu: {}, sigma:{} '.format(
            alpha_arr, mu_arr, sigma_arr ))

        # 实际两个高斯分布的参数: alpha = [0.3, 0.7] mu = [-2,0.5] sigma = [0.5,1]
        # 我们估计出的 两个高斯分布的参数:
        # alpha: [0.28674728 0.71325272] , mu: [-2.02549866  0.39455874], sigma:[0.47218494 0.9937545 ]
        # 效果还可以~

        end = time.time()


        print('model train time span:', end - start)



if __name__ == '__main__':

    test = Test()

    test.test_Gauss_dataset()
