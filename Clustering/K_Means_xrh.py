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

class KMeans:
    """
    K 均值聚类

    ref:
    《统计学习方法 第二版》李航

    Author: xrh
    Date: 2021-06-15

    """

    def __init__(self, K, epoch_num=20):

        # 聚类的类别个数
        self.K = K

        # 迭代的次数
        self.epoch_num = epoch_num

    def __distance(self,x1,x2):
        """
        空间中两个点的 欧式距离

        :param x1:
        :param x2:
        :return:
        """
        return np.sqrt( np.sum( (x1-x2)**2 ) )

    def update_pos(self,X,pos,centers):
        """
        更新所有样本 所属于的聚类中心

        :param pos:
        :param centers:
        :return:
        """
        N,m=np.shape(X)

        # 遍历所有的样本, 更新它们属于的聚类中心
        for i in range(N):

            min_distance = float('inf')
            min_distance_center = 0

            for k in range(self.K):  # 遍历所有的 聚类中心

                d = self.__distance(X[i], centers[k])
                if d < min_distance:
                    min_distance = d
                    min_distance_center = k

            pos[i] = min_distance_center # 样本属于离他最近的那个聚类中心


    def update_centers(self,X,pos,centers):
        """
        更新 所有聚类中心

        :param X:
        :param pos:
        :param centers:
        :return:
        """
        N, m = np.shape(X)

        for k in range(self.K): # 遍历所有聚类中心

            X_k=X[pos==k] # 属于聚类中心k 的样本点
            c = np.average(X_k, axis=0) # 中心点
            centers[k]=c

        # return centers

    def calc_loss(self,X,pos,centers):
        """
        计算 所有样本到 聚类中心的距离和
        :param X:
        :param pos:
        :param centers:
        :return:
        """
        N, m = np.shape(X)
        loss=0

        for i in range(N):
            loss+= self.__distance(X[i],centers[pos[i]])

        return loss


    def fit(self, X):
        """

        :param X:
        :return:
        """
        N,m=np.shape(X)

        # 算法(14.2)
        # 随机选取 K个聚类中心
        center_idx = random.sample(range(N), self.K )  # 从 X 间随机 生成 K个 不重复的点 ，结果以列表返回
        centers = X[center_idx]

        # 记录每个样本 属于哪个聚类中心
        pos=np.zeros(N,dtype=int) # pos[0]=1  0号样本属于聚类中心1

        loss=0

        for epoch in range(self.epoch_num):

            self.update_pos(X,pos,centers)

            self.update_centers(X,pos,centers)

            loss=self.calc_loss(X,pos,centers)

            print('epoch: {}, loss:{}'.format(epoch,loss))

        return loss,pos,centers


class Test:

    def test_iris_dataset(self):

        iris = datasets.load_iris()

        gt = iris['target']

        # 因为计算距离采用 欧式距离, 必须对特征数据进行标准化处理
        data = Normalize(iris['data'][:, :2])

        #为了可视化 只挑选2个维度
        x = data[:, 0]
        y = data[:, 1]

        # plt.scatter(x, y, color='green')
        # plt.xlim(4, 8)
        # plt.ylim(1, 5)
        # plt.show()

        start = time.time()  # 保存开始时间

        clu = KMeans(K=3)
        loss,pos, centers = clu.fit(data)

        end = time.time()
        print('model train time span:', end - start) # 0.094s


        # visualize result

        cat1 = data[pos==0]
        cat2 = data[pos==1]
        cat3 = data[pos==2]

        for ix, p in enumerate(centers):
            plt.scatter(p[0], p[1], color='C{}'.format(ix), marker='^', edgecolor='black', s=256)

        plt.scatter(cat1[:, 0], cat1[:, 1], color='green')
        plt.scatter(cat2[:, 0], cat2[:, 1], color='red')
        plt.scatter(cat3[:, 0], cat3[:, 1], color='blue')
        plt.title('Hierarchical clustering with k=3')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

        #  根据 “肘部法则” 寻找 K 值
        loss_list = []

        for i in range(1, 10):
            clu = KMeans(K=i, epoch_num=20)
            loss, pos, centers      = clu.fit(data)
            loss_list.append(loss)

        plt.title('K with loss')
        plt.plot(range(1, 10), loss_list)
        plt.show()

if __name__ == '__main__':

    test = Test()

    test.test_iris_dataset()
