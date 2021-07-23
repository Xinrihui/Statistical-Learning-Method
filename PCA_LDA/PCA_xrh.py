#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

from LogisticRegression_xrh import *

class basicPCA:
    """
    PCA 实现降维


    Author: xrh
    Date: 2021-07-20


    """

    def __init__(self,):

        # 降维后的各主成分的方差值
        self.explained_variance = None

        # 降维后的各主成分的方差值占总方差值的比例
        self.explained_variance_ratio = None

        # 投影矩阵
        self.projection = None


    def Zero_mean_centralize(self,X):
        """
        0均值中心化

        对所有特征进行 0均值中心化, 不用归一化方差,
        叫做中心化比较好理解

        :param X: shape(N,m)
                  N - 样本的个数
                  m - 样本的维度
        :return:
        """

        mu = np.mean(X, axis=0)  # 每一个特征的均值 shape:(m,)

        #s = np.std(X, axis=0)  # 每一个特征的标准差 shape:(m,)

        X = (X - mu)

        return X


    def fit(self,X):
        """

        1.对样本特征进行归一化
        2.求样本特征的协方差矩阵
        3.对协方差矩阵进行特征分解, 得到特征值和特征向量
        4.按照特征值的从大到小的排序, 得到排序后的特征向量矩阵

        :param X: 样本特征 shape(N,m)  N-样本个数, m-样本特征数量
        :return:
        """

        # 特征归一化
        X_nor = self.Zero_mean_centralize(X)

        # 尽量让协方差矩阵可逆, 去除掉 X 中全为0的列(在某个特征全部为0,)
        # mask = (X_nor == 0).all(0)
        # X_nor  = X_nor[:, ~mask]

        #  np.cov 默认将每一列作为一个样本
        # rowvar=False 将每一行作为一个样本
        cov_matrix = np.cov(X_nor,rowvar=False) # shape(m,m)

        # cov_matrix = np.dot(X_nor.T,X)

        # print(cov_matrix)

        # 协方差矩阵的对角线上为各个特征的方差, 方差能代表它的能量
        var = np.diagonal(cov_matrix)

        w, featurevector = np.linalg.eig(cov_matrix)
        # w 特征值
        # featurevector 特征值对应的特征向量, 由于cov_matrix是对称的, 因此 featurevector是单位正交矩阵
        # 验证 featurevector 是单位正交矩阵, 单位正交矩阵满足:
        # (A.T)A=I
        # print('I: ',np.dot(featurevector.T,featurevector))

        # print(w)
        # [2.93808505 0.9201649  0.14774182 0.02085386]
        # print(featurevector)
        # 第 i 列 为 w[i] 对应的特征向量
        # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html

        # 保留实数部分, 忽略虚数部分
        w = np.real(w)
        featurevector = np.real(featurevector)

        self.explained_variance = w # 特征值作为可解释偏差
        self.explained_variance_ratio = w / np.sum(w) #可解释偏差占比

        # 对特征值和对应的特征向量进行排序 (从大到小)
        sort_idx_w = np.argsort(w)[::-1]

        w_sort = w[sort_idx_w] # shape(m,1)
        featurevector_sort = featurevector[:,sort_idx_w] # shape(m,m)

        # 投影矩阵
        self.projection = featurevector_sort

        return X_nor,w_sort,featurevector_sort

    def transform(self,X,n_component):
        """

        1.利用排序后的特征向量矩阵, 取前 n_component列 作为投影矩阵
        2.利用投影矩阵对 归一化后的样本特征进行降维

        :param X: 样本特征 shape(N,m) N-样本个数, m-样本特征数量
        :param n_component: 返回所保留的成分个数 n
        :return:
        """
        # 特征归一化
        X_nor = self.Zero_mean_centralize(X)

        featurevector_sort = self.projection

        projection = featurevector_sort[:,:n_component] # shape(m,n_component)

        # 经过降维的样本特征 X
        X_projection = np.dot(X_nor,projection) #
        # X_nor shape(N,m) , n_component shape(m,n_component) -> shape(N,n_component)


        return X_projection

class Test:

    def test_tiny_dataset(self):
        """
        结果和 sklearn 不一样的原因:
        sklearn中的PCA是通过svd_flip函数实现的，sklearn对奇异值分解结果进行了一个处理，因为ui*σi*vi=(-ui)*σi*(-vi)，也就是u和v同时取反得到的结果是一样的，而这会导致通过PCA降维得到不一样的结果（虽然都是正确的）。

        ref:
        https://zhuanlan.zhihu.com/p/37777074

        :return:
        """

        X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

        pca = basicPCA()
        X_reduce = pca.transform(X, n_component=1)

        print('by xrh:')
        print(X_reduce)

        pca = PCA(n_components=1)
        pca.fit(X)
        print('by sklearn:')
        print(pca.transform(X))


    def load_cars_data(self,file):
        """

        :param file: (str) 数据文件的路径
        :return:

        df - (dataframe) 读取的数据表格
        X - (array) 特征数据数组

        """

        df = pd.read_csv(file)  # 读取csv文件
        df.drop('Pickup', axis=1, inplace=True)  # 去掉 全为0的特征
        X = np.asarray(df.values)  # 将数据转换成数组
        return df, X

    def test_cars_dataset(self):

        data_dir = 'dataset/cars.csv'
        _,X = self.load_cars_data(data_dir)

        pca = basicPCA()

        X_reduce = pca.transform(X, n_component=3)

        print(pca.explained_variance_ratio)


    def test_iris_dataset(self):

        pca = basicPCA()

        iris = datasets.load_iris()

        X = iris.data
        y = iris.target

        print('train data, row num:{} , column num:{} '.format(len(X), len(X[0])))

        X_reduce = pca.transform(X,n_component=2)

        print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio))

        plt.figure()
        colors = ['navy', 'turquoise', 'darkorange']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
            plt.scatter(X_reduce[y == i, 0], X_reduce[y == i, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of IRIS dataset')

        plt.show()

    def test_sklearn_iris_dataset(self):
        """
        使用 sklearn 的官方例子

        sklearn.decomposition.PCA的参数：

　　　　1）n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目。最常用的做法是直接指定降维到的维度数目，此时n_components是一个大于等于1的整数。当然，我们也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数。当然，我们还可以将参数设置为"mle", 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维。我们也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)。

　　　　2）whiten ：判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.对于PCA降维本身来说，一般不需要白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化。

　　　　3）svd_solver：即指定奇异值分解 SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的 PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 full则是传统意义上的SVD，使用了scipy库对应的实现。arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用了scipy库的sparse SVD实现。默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。

　　　　PCA类的成员:
       explained_variance_，它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。

       explained_variance_ratio_，它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。

        ref:
        https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py

        :return:
        """

        iris = datasets.load_iris()

        X = iris.data
        y = iris.target
        target_names = iris.target_names

        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)

        print(pca.components_)


        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))

        plt.figure()
        colors = ['navy', 'turquoise', 'darkorange']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of IRIS dataset')


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

        将样本特征使用 PCA 降维后输入 LR中训练


        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData_2classification('../dataset/Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        X = np.array(trainDataList)
        y = np.array(trainLabelList)

        # 降维, 由原来的 784 -> 200
        n_component = 200

        start = time.time()

        pca = basicPCA()
        pca.fit(X)
        X_reduce = pca.transform(X,n_component=n_component)

        end = time.time()

        print(' PCA cost time:{} '.format(end-start))

        # pca = PCA(n_components=n_component)
        # X_reduce = pca.fit(X).transform(X)

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

        # 测试集的样本特征 也要降维
        X_test_reduce = pca.transform(X_test,n_component=n_component)

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

        # 降维, 由原来的 784 -> 200
        n_component = 200

        start = time.time()

        pca = basicPCA()
        pca.fit(X)
        X_reduce = pca.transform(X, n_component=n_component)

        end = time.time()
        print(' PCA cost time:{} '.format(end - start))

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
        X_test_reduce = pca.transform(X_test,n_component=n_component)

        y_predict = clf.predict(X_test_reduce)

        print('test accuracy :', accuracy_score(y_predict, y_test))



if __name__ == '__main__':

    test = Test()

    # test.test_tiny_dataset()

    # test.test_cars_dataset()

    # test.test_iris_dataset()

    # test.test_sklearn_iris_dataset()

    # test.test_Mnist_dataset_2classification(60000,10000)

    test.test_Mnist_dataset(60000,10000)