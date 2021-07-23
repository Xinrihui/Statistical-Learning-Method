#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import time

from LogisticRegression_xrh import *

from PCA_xrh import *

class FisherLDA:
    """
    线性判别分析
    (适用多类别场景)


    ref:
    https://shengtao96.github.io/2017/06/20/Linear-Discriminant-Analysis/

    Author: xrh
    Date: 2021-07-20

    """

    def __init__(self,pre_n_component=2):
        """

        :param pre_n_component: 预先使用 PCA 降低的维度
        """

        # 降维后的各主成分的方差值 (排序后的特征值)
        self.explained_variance = None

        # 降维后的各主成分的方差值占总方差值的比例
        self.explained_variance_ratio = None

        # 投影矩阵(按照特征值排序后的特征向量组成的矩阵)
        self.projection_matrix = None

        # 预先使用 PCA 降低的维度
        self.pre_n_component = pre_n_component


    def fit(self,X,y):
        """
        LDA 的训练

        得到投影矩阵

        :param X: 样本特征 shape(N,m)  N-样本个数, m-样本特征数量
        :param y: 样本标签

        :return:
        """

        # 0. 预处理, 先使用 PCA 降低维度, 避免后面的 S_W 不可逆
        self.pca = basicPCA()
        self.pca.fit(X)
        X = self.pca.transform(X,n_component=self.pre_n_component)

        # 1.计算各个类的均值
        mu_N_label = {} # 各个类的 样本个数 和 均值

        # 所有类
        label_set = set([ele for ele in y])

        for label in label_set: # 遍历所有类

            X_label = X[y==label] # shape(Nk,m) Nk-此类的样本个数
            Nk,m = np.shape(X_label)

            mu_N_label[label] = (np.mean(X_label, axis=0,keepdims=True),Nk) # (shape(1,m), )

        # 2.计算类间 between class 方差(散度矩阵) S_B

        #所有样本的均值
        mu_global = np.mean(X,axis=0,keepdims=True) #  shape(1,m)

        S_B = 0
        for label in label_set:  # 遍历所有类

            diff = mu_N_label[label][0]-mu_global  #  shape(1,m)
            S_B += mu_N_label[label][1] * np.dot(diff.T,diff) # shape(m,m)
            # diff.T shape(m,1) , diff shape(1,m)

        # 3.计算类内 within class 方差(散度矩阵) S_W

        S_W = 0

        for label in label_set:  # 遍历所有类

            X_label = X[y == label]  # shape(Nk,m)
            diff = X_label - mu_N_label[label][0] # shape(Nk,m)
            # X_label shape(Nk,m),  mu_N_label[label][0] shape(1,m)

            S_W += np.dot(diff.T,diff)  # shape(m,m)
            # diff.T shape(m,Nk), diff shape(Nk,m)

        # 4.对 (S_W^(-1))S_B 特征值分解

        w, featurevector = np.linalg.eig(np.dot(np.linalg.inv( S_W ),S_B))
        # TODO:若 S_W 为奇异矩阵 ( 奇异矩阵 Singular matrix <=> 矩阵的行列式为 0 <=> 矩阵中的列向量存在线性相关 )
        #  导致无法求得 S_W 的逆矩阵
        #  Solution1:
        #  给 S_W 矩阵加上一个很小的单位矩阵:  1e-6* np.identity(np.shape(S_W)[0]), 保证它可逆
        #  Solution2:
        #  我们知道当样本数小于特征数的时候，求出的协方差矩阵不是满秩的，即协方差矩阵是不可逆的。因此，当特征数过多（大于样本数）的时候，我们就应该考虑怎么处理Sw不可逆的问题。
        #  一般在实现 LDA算法时，都会对样本进行一次 PCA算法的降维，消除样本的冗余，从而保证Sw是非奇异矩阵，当然即使Sw是奇异矩阵也是可解的，可以把Sw和Sb对角化。
        #  ref: https://shengtao96.github.io/2017/06/20/Linear-Discriminant-Analysis/

        # w 特征值
        # print(w)

        # featurevector 中第 i 列 为 w[i] 对应的特征向量
        # print(featurevector)

        # 保留实数部分, 忽略虚数部分
        w = np.real(w)
        featurevector = np.real(featurevector)

        self.explained_variance = w # 特征值作为可解释偏差
        self.explained_variance_ratio = w / np.sum(w) #可解释偏差占比

        # 对特征值和对应的特征向量进行排序 (从大到小)
        sort_idx_w = np.argsort(w)[::-1]
        w_sort = w[sort_idx_w]  # shape(m,1)
        featurevector_sort = featurevector[:, sort_idx_w]  # shape(m,m)

        # 投影矩阵
        self.projection_matrix = featurevector_sort

        return w_sort, featurevector_sort

    def transform(self,X,n_component):
        """

        1.按照特征值排序后的特征向量矩阵, 取前 n_component 列作为投影矩阵
        2.利用投影矩阵对样本特征进行降维

        :param X: 样本特征 shape(N,m) N-样本个数, m-样本特征数量
        :param y: 样本标签
        :param n_component: 返回所保留的成分个数 n
        :return:
        """
        projection = self.projection_matrix[:, :n_component]  # shape(m,n_component)

        # 先用 PCA 进行预先降维
        X = self.pca.transform(X, n_component=self.pre_n_component)

        # 经过降维的样本特征 X
        X_reduce = np.dot(X, projection)  # shape(N,n_component)
        # X shape(N,m) , n_component shape(m,n_component) -> shape(N,n_component)

        return X_reduce


class Test:


    def test_iris_dataset(self):
        """

        包含sklearn 官方的例子

        ref:
        https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py

        :return:
        """


        iris = datasets.load_iris()

        X = iris.data
        y = iris.target
        target_names = iris.target_names

        print('train data, row num:{} , column num:{} '.format(len(X), len(X[0])))

        lda = FisherLDA()
        # lda.fit(X,y)
        lda.fit(X,y)

        X_reduce = lda.transform(X,n_component=2)

        print('explained variance ratio (first two components): %s'
              % str(lda.explained_variance_ratio))

        colors = ['navy', 'turquoise', 'darkorange']
        lw = 2

        plt.figure()
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(X_reduce[y == i, 0], X_reduce[y == i, 1], alpha=.8, color=color,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('FisherLDA of IRIS dataset')


        lda2 = LinearDiscriminantAnalysis(n_components=2)
        X_r2 = lda2.fit(X, y).transform(X)

        colors = ['navy', 'turquoise', 'darkorange']

        plt.figure()
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('skleran LDA of IRIS dataset')

        plt.show()

    def loadData_2classification(self, fileName, n=1000):
        '''
        加载文件

        将 数据集 的标签 转换为 二分类的标签

        :param fileName:要加载的文件路径
        :param n: 返回的数据集的规模
        :return: 数据集和标签集
        '''
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

        将样本特征使用 LDA 降维后输入 LR中训练

        注意: 根据 LDA 的原理, LDA降维的维数只能选择 [1,类别数-1)范围之间的整数。

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData_2classification('../dataset/Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        X = np.array(trainDataList)
        y = np.array(trainLabelList)

        start = time.time()

        # 对于二分类, LDA降维的维数只能是1
        n_component = 1

        lda = FisherLDA(pre_n_component=100) # PCA 预降维的维度为100
        lda.fit(X,y)
        X_reduce = lda.transform(X,n_component=n_component)

        # lda2 = LinearDiscriminantAnalysis(n_components=n_component)
        # X_reduce = lda2.fit(X, y).transform(X)


        end = time.time()
        print(' LDA cost time:{} '.format(end-start))

        # 开始时间
        print('start training model....')
        start = time.time()

        clf = LR_2Classifier(use_reg=0)
        clf.fit(X=X_reduce, y=y, max_iter=50)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData_2classification('../dataset/Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        X_test = np.array(testDataList)
        y_test = np.array(testLabelList)

        X_test_reduce = lda.transform(X_test,n_component=n_component)

        # X_test_reduce = lda2.transform(X_test)

        y_pred = clf.predict(X_test_reduce)

        print('test dataset accuracy: {} '.format(accuracy_score(y_pred, y_test)))

    def loadData(self, fileName, n=1000):
        '''
        加载文件
        :param fileName:要加载的文件路径
        :param n: 返回的数据集的规模
        :return: 数据集和标签集
        '''
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
            # 放入的同时将标记转换为整型
            labelArr.append(int(curLine[0]))

            cnt += 1

        fr.close()

        # 返回数据集和标记
        return dataArr, labelArr

    def test_Mnist_dataset(self, n_train, n_test):
        """
        利用 Mnist 数据集 测试

        将样本特征使用 LDA 降维后输入 LR中训练

        注意: 根据 LDA 的原理, LDA降维的维数只能选择 [1,类别数-1)范围之间的整数。

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """
        # 获取训练集
        trainDataList, trainLabelList = self.loadData('../dataset/Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        X = np.array(trainDataList)
        y = np.array(trainLabelList)

        K = 10

        # 降维, 由原来的 784 -> 9
        n_component = 9

        start = time.time()

        lda = FisherLDA(pre_n_component=100) # PCA 预降维的维度为100
        lda.fit(X,y)
        X_reduce = lda.transform(X,n_component=n_component)

        end = time.time()
        print(' LDA cost time:{} '.format(end - start))

        # 开始时间
        print('start training model....')

        start = time.time()

        clf = LR_MultiClassifier(K=K,reg_lambda=0.1,use_reg=2)
        clf.fit(X_reduce, y,max_iter=50,learning_rate=0.1)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../dataset/Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        X_test = np.array(testDataList)
        y_test = np.array(testLabelList)

        # 测试集的样本特征 也要降维
        X_test_reduce = lda.transform(X_test,n_component=n_component)

        y_predict = clf.predict(X_test_reduce)

        print('test accuracy :', accuracy_score(y_predict, y_test))



if __name__ == '__main__':

    test = Test()

    # test.test_iris_dataset()

    # test.test_Mnist_dataset_2classification(60000,10000)

    test.test_Mnist_dataset(60000,10000)