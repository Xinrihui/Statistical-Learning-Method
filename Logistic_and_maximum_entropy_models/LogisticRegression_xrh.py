#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

import time
from deprecated import deprecated

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt
import matplotlib as mpl

def normalize(X):
    """
    对所有特征进行归一化

    :param X:
    :return:
    """

    # N, m = np.shape(X)  # N 个样本, m 个特征

    mu = np.mean(X, axis=0)  # 每一个特征的均值 shape:(m,)

    s = np.std(X, axis=0)  # 每一个特征的标准差 shape:(m,)

    X = (X - mu) / s

    return X


class LR_2Classifier:
    """
    逻辑斯蒂回归(二分类)

    1.通过 数值优化方法 梯度下降 找到最优解


    Author: xrh
    Date: 2021-07-05

    ref:
    统计学习方法 第二版》李航

    test1: 二分类任务
    数据集：Mnist
    参数: use_reg=2, reg_lambda=0.1,max_iter=50
    训练集数量：60000
    测试集数量：10000
    正确率： 0.974
    训练时长：107s


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
        self.W = None
        self.b = None

    def sigmoid(self, X):
        """
        sigmoid 激活函数

        :param X:
        :return:
        """

        return 1 / (1 + np.exp(-X))

    def lossfunc(self, W, b, X, y):
        """
        计算 带正则化的损失函数

        :param W:
        :param b:
        :param X:
        :param y:
        :return:
        """
        loss = 0

        N, m = np.shape(X)  # N 个样本, m 个特征

        for i in range(N):  # 遍历 N 个样本

            p = self.sigmoid(np.dot(W, X[i]) + b)
            loss += -(y[i] * np.log(p) + (1 - y[i]) * np.log(1 - p))

        loss = loss / N

        # 加上 L2 正则化
        if self.use_reg == 2:
            loss += (self.reg_lambda / 2) * np.sum(np.square(W))

        elif self.use_reg == 1:  # 加上 L1 正则化
            loss += self.reg_alpha * np.sum(np.abs(W))

        return loss

    def gradient(self, W, b, X, y):
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

            p = self.sigmoid(np.dot(W, X[i]) + b)
            diff = p - y[i]
            grad_W += diff * X[i]
            grad_b += diff

        grad_W = grad_W / N
        grad_b = grad_b / N

        # 加上 L2 正则化
        if self.use_reg == 2:
            grad_W += self.reg_lambda * W

        elif self.use_reg == 1:  # 加上 L1 正则化
            I = np.ones(m)
            I[W < 0] = -1
            grad_W += self.reg_alpha * I

        return grad_W, grad_b

    def fit(self, X, y, learning_rate=0.1, max_iter=50):
        """
        训练模型

        :param X:
        :param y:
        :param learning_rate: 学习率
        :param max_iter: 迭代次数
        :param use_BGD: 是否使用梯度下降求解
        :return:
        """

        N, m = np.shape(X)  # N 个样本, m 个特征

        print('train data num:{}'.format(N))

        W = np.zeros(m)  # 模型参数初始化
        b = 0

        for epoch in range(max_iter):
            loss = self.lossfunc(W, b, X, y)

            grad_w, grad_b = self.gradient(W, b, X, y)

            W -= learning_rate * grad_w
            b -= learning_rate * grad_b

            print('epcho: {} , loss:{}'.format(epoch, loss))

        # 保存训练参数
        self.W = W
        self.b = b

    def get_score(self, X):
        """
        推理测试数据集, 返回样本的分值 (概率值),
        我们可以利用此分值得出 P-R 曲线

        :param X:
        :return:
        """

        p = self.sigmoid(np.dot(X, self.W) + self.b)  # X: shape(N,m) W:shape(m,)

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

        p = self.sigmoid(np.dot(X, self.W) + self.b)  # X: shape(N,m) W:shape(m,)

        res[p > threshold] = 1

        return res


class LR_MultiClassifier:
    """
    逻辑斯蒂回归(多分类)

    1.通过 数值优化方法 梯度下降 找到最优解


    Author: xrh
    Date: 2021-07-05

    ref:
    统计学习方法 第二版》李航

    test1: 多分类任务
    数据集：Mnist
    参数: use_reg=2, reg_lambda=0.1,max_iter=50
    训练集数量：60000
    测试集数量：10000
    正确率：
    训练时长： s

    """

    def __init__(self, K, reg_alpha=0.1, reg_lambda=0.1, use_reg=2):
        """

        :param reg_alpha: K 分类的个数
        :param reg_alpha: L1 正则化参数
        :param reg_lambda: L2 正则化参数
        :param use_reg: 正则化类型选择, 2: L2 正则化 ; 1: L1 正则化 ; 0: 不使用正则化
        """
        self.K = K

        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.use_reg = use_reg

        # 模型的参数
        self.W = None
        self.b = None

    def softmax(self, X):
        """
        softmax处理，将结果转化为概率

        解决了 softmax的 上溢出 和 下溢出的问题

        ref: https://www.cnblogs.com/guoyaohua/p/8900683.html

        :param X: shape (K,N)
        :return: shape (K,N)
        """

        X_max = np.max(X, axis=0)
        X = X - X_max

        return np.exp(X) / np.sum(np.exp(X), axis=0)  # softmax处理，将结果转化为概率

    def log_softmax(self, X):
        """
        对 softmax 取对数

        解决了  softmax 的上下溢出问题

        :param X: shape (K,N)
        :return:
        """

        Max = np.max(X, axis=0)  # shape (N,)

        s = np.sum(np.exp(X - Max), axis=0)  # shape (N,)

        log_s = np.log(s)  # shape (N,)

        return X - Max - log_s  # shape (K,N)

    def calc_F(self,W, b, X):
        """

        :param W:
        :param b:
        :param X:
        :return:
        """
        N = np.shape(X)[0]

        F = np.dot(W, X.T) + np.tile(b.reshape((-1, 1)), N)  # shape:(K,N)
        # W shape(K,m), X.T shape(m,N) , b shape(K,) -> shape(K,N)

        return F

    def lossfunc(self, W, b, X, y_one_hot):
        """
        计算 带正则化的损失函数

        :param W:  shape(K,m)
        :param b:  shape(K)
        :param X:  shape(N,m)
        :param y_one_hot: shape: (K,N)
        :return:
        """

        N, m = np.shape(X)  # N 个样本, m 个特征

        F = self.calc_F(W,b,X) # shape:(K,N)

        loss = np.sum(-y_one_hot * self.log_softmax(F))  # shape:(1,)

        loss = loss / N

        # 加上 L2 正则化
        if self.use_reg == 2:
            loss += (self.reg_lambda / 2) * np.sum(np.square(W))

        elif self.use_reg == 1:  # 加上 L1 正则化
            loss += self.reg_alpha * np.sum(np.abs(W))

        return loss

    def gradient(self, W, b, X, y_one_hot):
        """
        计算梯度

        :param W:  shape(K,m)
        :param b:  shape(K,)
        :param X:  shape(N,m)
        :param y_one_hot: shape: (K,N)
        :return:
        """
        N, m = np.shape(X)  # N 个样本, m 个特征

        F = self.calc_F(W, b, X) # shape:(K,N)


        s = self.softmax(F) - y_one_hot # shape(K,N)

        grad_W = np.dot( s , X )  # shape: (K,m)
        #  s shape(K,N) , X shape(N,m)

        grad_b = np.sum(s,axis=1) # axis=1 干掉第1个维度, shape: (K,)

        grad_W = grad_W / N
        grad_b = grad_b / N

        # 加上 L2 正则化
        if self.use_reg == 2:
            grad_W += self.reg_lambda * W

        elif self.use_reg == 1:  # 加上 L1 正则化
            I = np.ones((self.K, m))
            I[W < 0] = -1
            grad_W += self.reg_alpha * I

        return grad_W, grad_b

    def fit(self, X, y, learning_rate=0.1, max_iter=50):
        """
        训练模型

        :param X:
        :param y:
        :param learning_rate: 学习率
        :param max_iter: 迭代次数
        :param use_BGD: 是否使用梯度下降求解
        :return:
        """

        N, m = np.shape(X)  # N 个样本, m 个特征

        print('train data num:{}'.format(N))

        print(' K={} classifier '.format(self.K))

        assert self.K == len(set(y))  # 设置的分类类别必须和训练数据的标签的类别相同

        # 模型参数初始化
        W = np.zeros((self.K, m))
        b = np.zeros((self.K,))

        # 将标签y one-hot 化, shape: (K,N)
        y_one_hot = (y == np.array(range(self.K)).reshape(-1, 1)).astype(
            np.int8)

        for epoch in range(max_iter):

            loss = self.lossfunc(W, b, X, y_one_hot)

            grad_w, grad_b = self.gradient(W, b, X, y_one_hot)

            W -= learning_rate * grad_w
            b -= learning_rate * grad_b

            print('epcho: {} , loss:{}'.format(epoch, loss))

        # 保存训练参数
        self.W = W
        self.b = b

    def predict_prob(self, X):
        """
        推理测试数据集, 返回样本的分值 (概率值),
        我们可以利用此分值得出 P-R 曲线

        :param X:
        :return:
        """
        N = np.shape(X)[0]

        F = self.calc_F(self.W, self.b, X) # shape:(K,N)


        P = self.softmax( F )  # shape:(K,N)

        return P

    def predict(self, X ):
        """
        推理测试数据集，返回样本标签

        :param X:

        :return:
        """

        P = self.predict_prob(X) # shape:(K,N)

        res = np.argmax(P, axis=0) #  axis=0 干掉第0个维度, shape: (N,)

        return res




class Test:

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

        测试 LR

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData_2classification('../dataset/Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

        # 开始时间
        print('start training model....')
        start = time.time()

        clf = LR_2Classifier(use_reg=2)
        clf.fit(X=trainDataArr, y=trainLabelArr, max_iter=50)

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

        K = 10

        # 开始时间
        print('start training model....')

        start = time.time()

        clf = LR_MultiClassifier(K=K,reg_lambda=0.1,use_reg=2)
        clf.fit(trainDataArr, trainLabelArr,max_iter=50,learning_rate=0.1)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../dataset/Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        y_predict = clf.predict(testDataArr)

        print('test accuracy :', accuracy_score(y_predict, testLabelArr))

        # 查看每一种类别 的评价指标
        print('print the classification report: ')

        report = classification_report(testLabelArr, y_predict)

        print(report)

        # 打印混淆矩阵
        print('print the confusion matrix')

        confusion = confusion_matrix(testLabelArr, y_predict)
        print(confusion)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=list(range(K)))
        disp.plot()

        plt.show()



    def test_iris_dataset(self):

        K =3

        # 使用iris数据集，其中有三个分类， y的取值为0,1，2
        X, y = datasets.load_iris(True)  # 包括150行记录
        # 将数据集一分为二，训练数据占80%，测试数据占20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)

        clf = LR_MultiClassifier(K=K,reg_lambda=0.1,use_reg=2)
        clf.fit(X_train, y_train,max_iter=50,learning_rate=0.05)

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

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels = list(range(K)))
        disp.plot()

        plt.show() # 显示图片

if __name__ == '__main__':
    test = Test()

    # test.test_Mnist_dataset_2classification(60000, 10000)

    test.test_Mnist_dataset(60000,10000)

    # test.test_iris_dataset()