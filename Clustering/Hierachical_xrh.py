#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import random
import numpy as np
from sklearn import datasets,cluster
import matplotlib.pyplot as plt

import time


def Normalize(Xarray):
    """
    定义标准化函数，对每一列特征进行min-max标准化，将数据缩放到0-1之间

    标准化处理对于计算距离的机器学习方法是非常重要的，因为特征的尺度不同会导致计算出来的距离倾向于尺度大的特征，
    为保证距离对每一列特征都是公平的，必须将所有特征缩放到同一尺度范围内
    :param Xarray:
    :return:
    """

    for f in range(Xarray.shape[1]):
        maxf = np.max(Xarray[:, f])
        minf = np.min(Xarray[:, f])
        for n in range(Xarray.shape[0]):
            Xarray[n][f] = (Xarray[n][f] - minf) / (maxf - minf)
    return Xarray


class Hierarchical:
    """
    层次聚类


    ref:
    《统计学习方法 第二版》李航

    test1: 聚类任务
    数据集：iris
    训练集数量：150
    得分值(ARI)： 0.558
    模型训练时长：1.54s

    Author: xrh
    Date: 2021-06-18
    """

    def __init__(self, K):

        # 聚类的类别个数
        self.K = K

    def __distance(self,x1,x2):
        """
        计算两个样本之间的 欧式距离

        :param x1:
        :param x2:
        :return:
        """
        return np.sqrt( np.sum( (x1-x2)**2 ) )

    def __distance_all_ele(self, X ):
        """
        计算 所有样本 两两之间的距离

        :param pos:
        :param centers:
        :return:
        """
        N,m=np.shape(X)

        dist=np.zeros((N,N))

        # 遍历所有的样本
        for i in range(N):
            for j in range(i,N):

                d=self.__distance(X[i],X[j])
                dist[i][j] = d
                dist[j][i] = d

        return dist

    def __distance_group(self,dist_all,group1,group2):
        """
        计算两类的类间距离，采用 最短距离 (公式 14.4 )

        :param dist_all:
        :param group1:
        :param group2:
        :return:
        """
        min_dist = float('inf')

        for i in group1:
            for j in group2:

                if i != j:
                    dist = dist_all[i][j]
                    min_dist = min(min_dist,dist)

        return min_dist

    def fit(self, X):
        """
        算法 14.1 聚合聚类算法

        聚合（自下而上）：
        聚合法开始将每个样本各自分裂到一个类，之后将相距最近的两类合并，建立一个新的类，重复次操作知道满足停止条件，得到层次化的类别。

        时间复杂度: O( N^3 )

        :param X:
        :return:
        """
        N,m=np.shape(X)

        # 初始化
        dist_all=self.__distance_all_ele(X)

        # 所有元素单独成组
        group_list=[ [i] for i in range(N)]

        k=len(group_list)

        while k>self.K: # 聚类后的类别个数大于预设的类别,迭代继续

            print('Number of groups:', k)

            min_dist=float('inf')
            min_dist_groups=(0,0)

            # 计算所有组两两之间的距离
            for i in range(k-1):
                for j in range(i+1,k):
                    dist = self.__distance_group(dist_all,group_list[i],group_list[j])

                    if dist <= min_dist:
                        min_dist=dist
                        min_dist_groups=(i,j)

            # 合并 拥有最小距离的两个组
            (i, j) = min_dist_groups
            merge_group= group_list[i]+group_list[j]
            group_list[i] = merge_group
            del group_list[j]

            k=len(group_list)

        return group_list


class Hierarchical_v2:
    """
    使用 最小生成树(MST)算法, 优化算法的时间复杂度

    """
    pass

from scipy.special import comb


def Adjusted_Rand_Index(group_list, Ylist, k):
    """
    计算调整 兰德系数(ARI)的函数，调整兰德系数是一种聚类方法的常用评估方法

    :param group_list:
    :param Ylist:
    :param k:
    :return:
    """

    group_array = np.zeros((k, k))  # 定义一个数组，用来保存聚类所产生的类别标签与给定的外部标签各类别之间共同包含的数据数量
    y_dict = {}  # 定义一个空字典，用来保存外部标签中各类所包含的数据，结构与group_dict相同
    for i in range(len(Ylist)):
        if Ylist[i] not in y_dict:
            y_dict[Ylist[i]] = [i]
        else:
            y_dict[Ylist[i]].append(i)
    # 循环计算group_array的值
    for i in range(k):
        for j in range(k):
            for n in range(len(Ylist)):
                if n in group_list[i] and n in y_dict[list(y_dict.keys())[j]]:
                    group_array[i][j] += 1  # 如果数据n同时在group_dict的类别i和y_dict的类别j中，group_array[i][j]的数值加一
    RI = 0  # 定义兰德系数(RI)
    sum_i = np.zeros(3)  # 定义一个数组，用于保存聚类结果group_dict中每一类的个数
    sum_j = np.zeros(3)  # 定义一个数组，用于保存外部标签y_dict中每一类的个数
    for i in range(k):
        for j in range(k):
            sum_i[i] += group_array[i][j]
            sum_j[j] += group_array[i][j]
            if group_array[i][j] >= 2:
                RI += comb(group_array[i][j], 2)  # comb用于计算group_array[i][j]中两两组合的组合数
    ci = 0  # ci保存聚类结果中同一类中的两两组合数之和
    cj = 0  # cj保存外部标签中同一类中的两两组合数之和
    for i in range(k):
        if sum_i[i] >= 2:
            ci += comb(sum_i[i], 2)
    for j in range(k):
        if sum_j[j] >= 2:
            cj += comb(sum_j[j], 2)
    E_RI = ci * cj / comb(len(Ylist), 2)  # 计算RI的期望
    max_RI = (ci + cj) / 2  # 计算RI的最大值
    return (RI - E_RI) / (max_RI - E_RI)  # 返回调整兰德系数的值

class Test:

    def test_iris_dataset(self):

        iris = datasets.load_iris()

        # X = iris['data']

        # 因为计算距离采用 欧式距离, 必须对特征数据进行标准化处理
        X = Normalize(iris['data'])

        data_visual= X[:, :2] # 可视化数据

        # x = data_visual[:, 0]
        # y = data_visual[:, 1]

        # plt.scatter(x, y, color='green')
        # plt.xlim(4, 8)
        # plt.ylim(1, 5)
        # plt.show()

        start = time.time()  # 保存开始时间

        K=3
        clu = Hierarchical(K=K)
        group_list = clu.fit(X)

        end = time.time()

        y = iris['target']

        print(group_list)
        ARI = Adjusted_Rand_Index(group_list, y, K)  # 计算ARI用来评估聚类结果
        print('Adjusted Rand Index:', ARI)

        print('model train time span:', end - start)

        # visualize result

        cat1 = data_visual[group_list[0]]
        cat2 = data_visual[group_list[1]]
        cat3 = data_visual[group_list[2]]

        plt.scatter(cat1[:, 0], cat1[:, 1], color='green')
        plt.scatter(cat2[:, 0], cat2[:, 1], color='red')
        plt.scatter(cat3[:, 0], cat3[:, 1], color='blue')
        plt.title('Hierarchical clustering with k=3')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()


if __name__ == '__main__':

    test = Test()

    test.test_iris_dataset()
