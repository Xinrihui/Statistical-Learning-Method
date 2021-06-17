#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import time
from sklearn.datasets import load_iris

from collections import *

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# 定义负无穷大
infinite = -(2 ** 31)


class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器

    1.适用于 类别型特征 和 数值型特征的训练数据；
      对于类别型特征, 无需进行编码, 比较方便

    ref:
    统计学习方法 第二版》李航

    Author: xrh
    Date: 2021-06-16

    test1: 多分类任务
    数据集：Mnist
    训练集数量：60000
    测试集数量：10000
    正确率： 0.84
    训练时长：63s

    """

    def __init__(self, K,Lambda=1):
        """
        初始化 超参数
        :param K: K 分类
        :param Lambda: 平滑因子, Lambda=1 为拉普拉斯平滑
        """
        self.K=K
        self.Lambda=Lambda

        self.P_Y=None
        self.P_X_Y=None


    def fit(self, X,y):
        """
        模型训练

        :param X:
        :param y:
        :return:
        """

        dict_labels = Counter(y.flatten()) # 不同标签值的统计

        assert len(dict_labels) == self.K # 实际样本中的类别个数应该为 K,否则报错

        N,m = np.shape(X) # N : 样本个数 ; m 特征维度

        P_Y={} # 公式(4.11)

        z = np.log(N + self.K*self.Lambda) # 公式(4.11) 分母
        for y_k, y_v in dict_labels.items(): # 遍历所有的标签值

            P_Y[y_k] = np.log(y_v + self.Lambda) - z # 公式(4.11); 概率值对数化

        P_X_Y={} # 公式(4.10)

        for i in range(m): # 遍历所有的特征

            Xi = X[:, i] # 特征 Xi

            dict_Xi=defaultdict(dict)

            P_X_Y[i]=dict_Xi

            set_X_i=set(Xi) # 特征 i 的所有特征值的集合

            Si=len(set_X_i) # 特征 i 的特征值的个数
            # Si=2

            for rid in range(N): #遍历所有的样本进行统计

                if y[rid] in dict_Xi[Xi[rid]]: #
                    dict_Xi[ Xi[rid] ][ y[rid] ]+=1 # 公式(4.10) 分子

                else:#
                    dict_Xi[Xi[rid]][y[rid]]=1 # 第一次遇到计数为1

            for i_k in set_X_i: # 遍历特征i的所有特征值 i_k

                for y_k, y_v in dict_labels.items(): # 遍历所有的标签值

                    z= np.log(y_v + Si*self.Lambda) # 公式(4.10) 分母

                    # try:
                    if y_k in dict_Xi[i_k]:
                        dict_Xi[i_k][y_k] = np.log( dict_Xi[i_k][y_k] + self.Lambda ) - z #公式(4.10)
                    else: # 标签值y_k 不在 dict_Xi[i_k] 中
                        dict_Xi[i_k][y_k] = np.log(0 + self.Lambda) - z  #

                    # except Exception as err:
                    #     print(err)  # debug 时 , 在此处打断点


        # 训练完成 更新参数
        self.P_Y= P_Y
        self.P_X_Y=P_X_Y

        self.N, self.m = N,m


    def __predict(self, row):
        """
        根据 MAP 预测 样本的标签值

        :param row:
        :return:
        """
        max_label=None
        max_label_prob=float('-inf')

        for y_k, prob_y in self.P_Y.items():  # 遍历所有的标签值

            sum_logprob=0
            for i in range(self.m):  # 遍历所有的特征

                if row[i] in self.P_X_Y[i] : # 特征i 的特征值 row[i] 在训练集中不一定有
                    logprob=self.P_X_Y[i][row[i]][y_k]
                else:
                    logprob= infinite # 概率为 P=0 取对数后为 (-无穷)

                sum_logprob+=logprob #  公式 4.7 中为概率连续相乘,因为我们对概率取了对数,因此转换为连续相加

            log_prob = prob_y + sum_logprob # (公式 4.7)

            if  log_prob >= max_label_prob:
                max_label_prob=log_prob
                max_label=y_k


        return max_label


    def predict(self, X_test):
        """
        预测 测试 数据集，返回预测结果

        :param test_data:
        :return:
        """
        res_list = []
        for row in X_test:
            res_list.append( self.__predict(row) )

        return res_list


class Test:


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

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData('../dataset/Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

        # 开始时间
        print('start training model....')

        start = time.time()

        nb = NaiveBayesClassifier(K=10)
        nb.fit(trainDataArr, trainLabelArr)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../dataset/Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        y_predict= nb.predict(testDataArr)

        print('test accuracy :', accuracy_score(y_predict,testLabelArr))



    def test_iris_dataset(self):

        # 使用iris数据集，其中有三个分类， y的取值为0,1，2
        X, y = load_iris(True)  # 包括150行记录
        # 将数据集一分为二，训练数据占80%，测试数据占20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)

        nb = NaiveBayesClassifier(K=3)
        nb.fit(X_train, y_train)

        y_predict = nb.predict(X_test)

        print('test accuracy :', accuracy_score(y_predict, y_test))






if __name__ == '__main__':
    test = Test()

    # test.test_Mnist_dataset(60000,10000)

    test.test_iris_dataset()

