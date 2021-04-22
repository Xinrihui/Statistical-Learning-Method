
import numpy as np

import time


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn import datasets

from sklearn.model_selection import train_test_split

class AdaBoost:
    """
    
    适用于 二分类 的 AdaBoost

    基分类器为 决策树树桩 ( stump )
    
    Author: xrh
    Date: 2021-03-24
    
    ref: 
    
    test1: 二分类任务
    
    数据集：Mnist
    参数: error_rate_threshold =0.05, max_iter=10
    训练集数量：60000
    测试集数量：10000
    正确率：0.9535
    模型训练时长：103s   

    """

    def __init__(self, error_rate_threshold =0.05, max_iter=10):

        # 训练中止条件 error_rate <self.error_rate_threshold ( 若当前得到的基分类器的组合 错误率 小于阈值, 则停止训练)
        self.error_rate_threshold = error_rate_threshold

        # 最大迭代次数
        self.max_iter = max_iter

        self.G = []  # 弱分类器 集合

    def build_stump(self, X, y, w):
        """
        以带权重的分类误差最小为目标，选择最佳 分类特征 和 分类特征的阈值

        best_stump['feature'] 合适的特征所在维度
        best_stump['value']  合适特征的阈值
        best_stump['flag']  树桩分类的标识lt,rt

        """

        best_stump = {}

        feature_Num = np.shape(X)[1]  # 特征的数目

        N = np.shape(X)[0]  # 样本的个数

        min_em = float('inf')  # 最小的 分类误差率

        min_em_y_predict = None  # 最小分类误差率 对应的 y 的预测值

        for feature in range(feature_Num):  # 遍历 所有特征

            feature_value_set = {v for v in X[:, feature]}

            for value in feature_value_set:  # 遍历特征的所有取值

                for flag in ['lt', 'rt']:  # 树桩的分类 标识,
                    #  若为 'lt' , 则 左边( <= ) 为负例 -1, 右边( > )为正例 +1
                    #  若为 'rt' , 则 左边( <= ) 为正例 +1, 右边( > )为 负例 -1

                    y_predict = self.base_estimator( X[:, feature], value, flag ) # 生成了一个 弱学习器

                    err_arr = np.ones(N, dtype=int)
                    err_arr[y_predict == y] = 0  # y_predict 与 y 相等为0, 不等为1

                    em = np.dot(w, err_arr)  # 公式 8.1 ; 加权的分类误差率 ( 样本权重的作用 )

                    if em < min_em:
                        min_em = em
                        min_em_y_predict = y_predict
                        best_stump['feature'] = feature
                        best_stump['value'] = value
                        best_stump['flag'] = flag

        return min_em, min_em_y_predict, best_stump

    def updata_w(self, y, w, y_predict, alpha):
        """
        更新样本权重w
        """
        # 根据公式8.4 8.5 更新样本权重

        p_arr = w * np.exp(-alpha * y_predict * y)

        zm = np.sum(p_arr)

        w = p_arr / zm

        return w

    def base_estimator(self, X_feature, value, flag):
        """
        计算单个弱分类器（决策树桩）预测输出
        
        """
        y_predict = np.zeros(len(X_feature), dtype=int)

        if flag == 'lt':

            y_predict[X_feature <= value] = -1
            y_predict[X_feature > value] = 1

        else:  # flag == 'rt':

            y_predict[X_feature <= value] = 1
            y_predict[X_feature > value] = -1

        return y_predict

    def fit(self, X, y):

        """
        对训练数据进行学习
        """
        N = np.shape(X)[0] # 样本的个数

        # 初始化样本权重w
        w = np.ones((N)) * (1 / N)

        f = 0  # 基分类器 的加权和

        for m in range(self.max_iter):  # 进行 第 m 轮迭代

            em, y_predict, best_stump = self.build_stump(X, y, w)

            alpha = (1 / 2) * np.log((1 - em) / em)

            self.G.append((alpha, best_stump))  # 存储 基分类器

            # 当前 所有弱分类器加权 得到的 最终分类器 的 分类错误率

            f += alpha * y_predict
            G = np.sign(f)

            err_arr = np.ones(N, dtype=int)
            err_arr[G == y] = 0
            err_rate = np.mean(err_arr)

            print('round:{}, err_rate:{}'.format(m, err_rate))

            if err_rate < self.error_rate_threshold:  # 错误率 已经小于 阈值, 则停止训练
                break

                # 更新 w
            w = self.updata_w(y, w, y_predict, alpha)

    def predict(self, X):
        """
        对新数据进行预测

        """

        f = 0  # 最终分类器

        for alpha, best_stump in self.G:
            y_predict = self.base_estimator(X[:, best_stump['feature']], best_stump['value'], best_stump['flag'])

            f += alpha * y_predict

        G = np.sign(f)

        return G.astype(int)

    def score(self, X, y):
        """对训练效果进行评价"""

        G = self.predict(X)

        N = np.shape(X)[0]  # 样本的个数

        err_arr = np.ones(N, dtype=int)
        err_arr[G == y] = 0
        err_rate = np.mean(err_arr)

        accuracy = 1 - err_rate

        return accuracy


from CartTree_classification_xrh import *


class AdaBoost_SAMME:
    """

    SAMME 算法 实现 AdaBoost 的多分类

    基分类器为 CART分类树

    Author: xrh
    Date: 2021-03-30

    ref:

    test1: 多分类任务

    数据集：Mnist
    参数: n_estimators=20,learning_rate=0.8
    训练集数量：60000
    测试集数量：10000
    正确率：0.8577
    模型训练时长：1099s

    """

    def __init__(self, error_rate_threshold=0.05, n_estimators=10, learning_rate=1.0, algorithm='SAMME',max_depth=5):
        """

        :param error_rate_threshold:
        :param n_estimators:
        :param learning_rate:
        :param algorithm:
        :param max_depth:

        """

        # 训练中止条件 error_rate <self.error_rate_threshold ( 若当前得到的基分类器的组合 错误率 小于阈值, 则停止训练)
        self.error_rate_threshold = error_rate_threshold

        # 弱分类器的个数 即 最大迭代次数
        self.n_estimators = n_estimators

        self.learning_rate = learning_rate  # 学习率

        self.algorithm = algorithm

        self.max_depth=max_depth # 基分类器(决策树)的最大深度

        self.G = []  # 弱分类器(决策树) 的集合


    def fit(self, X, y):
        """
        对训练数据进行学习

        :param X:
        :param y:
        :return:
        """

        N = np.shape(X)[0]

        self.K = len({ele for ele in y})  # y 中有多少种不同的标签,  K分类

        print('according to the training dataset : K={} classification task'.format(self.K))

        # 初始化样本权重w
        w = np.full(shape=N,fill_value=1/N )
        # w = np.ones((N)) * (1 / N)

        F = 0  # 强分类器

        feature_value_set = CartTree.get_feature_value_set(X) # 可供选择的特征集合 , 包括 (特征, 切分值)

        for m in range(self.n_estimators):  # 进行 第 m 轮迭代

            print('round:{}'.format(m))

            # 使用 自己 的 CartTree
            CT = CartTree(max_depth=self.max_depth)  # 第 m 个弱分类器
            CT.fit(X, y, feature_value_set=feature_value_set, sample_weight=w)

            # 使用 sklearn 的 CartTree
            # CT = DecisionTreeClassifier(max_depth=self.max_depth)
            # CT.fit(X, y,sample_weight=w)

            y_predict = CT.predict(X)

            em = ( ( y_predict != y ) * w ).sum()  # 计算弱分类器误差

            print('em:',em)

            alpha = self.learning_rate * ( np.log((1 - em) / em) + np.log(self.K - 1) )  # 弱分类器权重

            self.G.append( ( alpha, CT ) )  # 存储 基分类器

            # 当前 所有弱分类器加权 得到的 最终分类器 的 分类错误率

            y_predict_one_hot = (y_predict == np.array(range(self.K)).reshape(-1, 1)).T.astype(
                np.int8)  # 将 预测向量 扩展为 one-hot

            F += alpha * y_predict_one_hot  # 弱分类器组合成强分类器

            G = np.e ** F / ((np.e ** F).sum(axis=1).reshape(-1, 1))  # softmax处理，将结果转化为概率

            G_label = np.argmax(G, axis=1)  # 取 概率最大的 作为 预测的标签

            err_arr = np.ones(N, dtype=int)
            err_arr[G_label == y] = 0
            err_rate = np.mean(err_arr)  # 计算训练误差

            print('alpha:{}, err_rate:{}'.format(alpha, err_rate))

            if err_rate < self.error_rate_threshold:  # 错误率 已经小于 阈值, 则停止训练
                break

            # 更新 w
            w = w * np.exp( alpha * (y_predict != y) )
            w /= np.sum(w)

            print('w[0:10]:',w[0:10])

    def predict(self, X):

        """
        对新数据进行预测

        :param X:
        :return:
        """

        F = 0  # 最终分类器

        for alpha,CT in self.G:

            y_predict = CT.predict(X)

            y_predict_one_hot = (y_predict == np.array(range(self.K)).reshape(-1, 1)).T.astype(
                np.int8)  # 将 预测向量 扩展为 one-hot

            F += alpha * y_predict_one_hot  # 弱分类器组合成强分类器

        G = np.e ** F / ((np.e ** F).sum(axis=1).reshape(-1, 1))  # softmax处理，将结果转化为概率

        G_label = np.argmax(G, axis=1)  # 取 概率最大的 作为 预测的标签

        return G_label

    def score(self, X, y):
        """
        对训练效果进行评价
        :param X:
        :param y:
        :return:
        """
        G_label = self.predict(X)

        N = np.shape(X)[0]  # 样本的个数

        err_arr = np.ones(N, dtype=int)
        err_arr[G_label == y] = 0
        err_rate = np.mean(err_arr)

        accuracy = 1 - err_rate

        return accuracy


class Test:


    def test_small_dataset(self):
        """

        利用《统计学习方法》 表 8.1 中的数据集 测试  AdaBoost

        :return:
        """

        # 获取训练集

        X = np.array([[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2],
                      [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]])
        y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])

        # 开始时间
        start = time.time()

        # 创建决策树
        print('start create model')

        clf = AdaBoost()
        clf.fit(X, y)

        print(clf.G)

        print(' model complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        # 测试数据集

        y_predict = clf.predict(X)
        score = clf.score(X, y)

        print("原始输出:", y)
        print("预测输出:", y_predict)
        print("预测正确率：{:.2%}".format(score))


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
            # 标签0设置为1，反之为-1
            if int(curLine[0]) == 0:
                labelArr.append(1)
            else:
                labelArr.append(-1)

            cnt += 1

        fr.close()

        # 返回数据集和标记
        return dataArr, labelArr

    def test_Mnist_dataset_2classification(self, n_train, n_test):
        """
        将 Mnist (手写数字) 数据集 转变为 二分类 数据集 
        
        测试 AdaBoost 
        
        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData_2classification('../Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

        # 开始时间
        print('start training model....')
        start = time.time()

        clf = AdaBoost( )
        clf.fit(trainDataArr, trainLabelArr)

        print(clf.G)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData_2classification('../Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        print('test dataset accuracy: {} '.format(clf.score(testDataArr, testLabelArr)) )



    def loadData(self,fileName, n=1000):
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
            labelArr.append(int(curLine[0]))
            cnt += 1

        fr.close()

        # 返回数据集和标记
        return dataArr, labelArr


    def test_Mnist_dataset(self, n_train, n_test):
        """
         Mnist (手写数字) 数据集 

        测试 AdaBoost  的 多分类

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData('../Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

        # 开始时间
        print('start training model....')
        start = time.time()


        # 使用 sklearn 库测试, max_depth=5, n_estimators=50, learning_rate=0.8, 准确率为 0.9 ,
        # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)
        #                          , algorithm="SAMME", n_estimators=50, learning_rate=0.8)
        #
        # clf.fit(trainDataArr, trainLabelArr)

        clf = AdaBoost_SAMME(n_estimators=50,learning_rate=0.8,max_depth=5)
        clf.fit(trainDataArr, trainLabelArr)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format( len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        print('test dataset accuracy: {} '.format(clf.score(testDataArr, testLabelArr)))


    def test_iris_dataset(self ):

        # 使用iris数据集，其中有三个分类， y的取值为0,1，2
        X, y = datasets.load_iris(True)  # 包括150行记录
        # 将数据集一分为二，训练数据占80%，测试数据占20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 )


        clf = AdaBoost_SAMME(n_estimators=3, learning_rate=0.8, max_depth=1)
        clf.fit(X_train, y_train)

        print(clf.score(X_test,y_test))


if __name__ == '__main__':

    test=Test()


    # test.test_small_dataset()

    # test.test_Mnist_dataset_2classification(60000,10000)

    # test.test_Mnist_dataset(n_train=6000,n_test=1000)

    test.test_iris_dataset()

