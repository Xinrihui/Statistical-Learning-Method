import numpy as np

import time

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import datasets

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

from scipy.special import expit, logsumexp


from CartTree_regression_xrh import *


from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve


class XGBoost_v1:
    """

    实现 基础的 XGBoost , 功能包括:
    1. 回归
    2.

    Author: xrh
    Date: 2021-04-18

    ref: https://zhuanlan.zhihu.com/p/91652813

    test1: 多分类任务

    数据集：Mnist
    参数: error_rate_threshold=0.01, max_iter=20, max_depth=3 , learning_rate=0.5
    训练集数量：60000
    测试集数量：10000
    正确率： 0.915
    模型训练时长：1542s

    """

    def __init__(self, error_rate_threshold=0.05, max_iter=10, max_depth=1):
        """

        :param error_rate_threshold: 训练中止条件, 若当前得到的基分类器的组合 的错误率 小于阈值, 则停止训练
        :param max_iter: 最大迭代次数
        :param max_depth: CART 回归树 的最大深度
        """

        # 训练中止条件 error_rate  < self.error_rate_threshold ( 若当前得到的基分类器的组合 的错误率 小于阈值, 则停止训练)
        self.error_rate_threshold = error_rate_threshold

        # 最大迭代次数
        self.max_iter = max_iter

        # CART 回归树 的最大深
        self.max_depth = max_depth

        self.G = []  # 弱分类器 集合

    def sigmoid( self , X ):
        """
        sigmoid 激活函数
        :param X:
        :return:
        """
        return 1 / (1 + np.exp(-X))

    # def softmax_deprecated(self,X):
    #     """
    #     softmax处理，将结果转化为概率
    #
    #     :param X:
    #     :return:
    #     """
    #     #TODO: 导致 上溢出 和 下溢出 问题
    #
    #     return  np.exp(X) / np.sum( np.exp(X) , axis=0 )  # softmax处理，将结果转化为概率

    def softmax_deprecated(self,X):
        """
        softmax处理，将结果转化为概率

        解决了 softmax的 上溢出 和 下溢出的问题

        ref: https://www.cnblogs.com/guoyaohua/p/8900683.html

        :param X: shape (K,N)
        :return: shape (N,)
        """

        X_max= np.max( X, axis=0)
        X= X-X_max

        return  np.exp(X) / np.sum( np.exp(X) , axis=0 )  # softmax处理，将结果转化为概率

    def softmax(self,X):
        """
        softmax处理，将结果转化为概率

        解决了 softmax的 溢出问题

        np.nan_to_num : 使用0代替数组x中的nan元素，使用有限的数字代替inf元素

        ref: sklearn 源码
            MultinomialDeviance -> def negative_gradient

        :param X: shape (K,N)
        :return: shape (N,)
        """

        return  np.nan_to_num( np.exp(X - logsumexp(X, axis=0)) )  # softmax处理，将结果转化为概率

    def fit(self, X, y, learning_rate):
        """

        用 训练数据 拟合模型

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :param y: 标签数据 , shape=(N_sample,)
        :param learning_rate: 学习率
        :return:
        """

        N = np.shape(X)[0]  # 样本的个数

        self.K = len({ele for ele in y})  # y 中有多少种不同的标签,  K分类

        print('according to the training dataset : K={} classification task'.format(self.K))

        F_0 = np.zeros( (self.K ),dtype=float)  # shape : (K,)

        for k in range(self.K): # 遍历 所有的 类别

            F_0[k] = len(y[y == k]) / len(y)

        self.G.append(F_0)

        F = np.transpose([F_0] * N) # 对 F_0 进行复制,  shape : (K, N)

        feature_value_set = RegresionTree.get_feature_value_set(X)  # 可供选择的特征集合 , 包括 (特征, 切分值)

        y_one_hot = (y == np.array(range(self.K)).reshape(-1, 1)).astype(
            np.int8)  # 将 预测向量 扩展为 one-hot , shape: (K,N)

        for m in range(1,self.max_iter):  # 进行 第 m 轮迭代

            p = self.softmax( F ) #  shape: (K,N)

            DT_list=[]

            for k in range(self.K): #  依次训练 K 个 二分类器

                print( '======= train No.{} 2Classifier ======='.format(k) )

                r = y_one_hot[k] - p[k]   # 残差 shape:(N,)

                # 训练 用于 2分类的 回归树
                DT = RegresionTree_GBDT(min_square_loss=0.1, max_depth=self.max_depth,print_log=True)

                DT.fit(X, r, y_one_hot[k], feature_value_set=feature_value_set)

                y_predict = (self.K / (self.K-1)) * ( DT.predict(X) ) #  shape:(N,)

                DT_list.append(DT)

                F[k] += learning_rate * y_predict  # F[k]  shape:(N,)

                # print('======= end =======')


            self.G.append( (learning_rate, DT_list) )  # 存储 基分类器

            # 计算 当前 所有弱分类器加权 得到的 最终分类器 的 分类错误率

            G = self.softmax( F )

            G_label = np.argmax( G, axis=0 )  # 取 概率最大的 作为 预测的标签

            err_arr = np.ones( N, dtype=int )
            err_arr[G_label == y] = 0
            err_rate = np.mean(err_arr)  # 计算训练误差

            print('round:{}, err_rate:{}'.format(m, err_rate))
            print('======================')

            if err_rate < self.error_rate_threshold:  # 错误率 已经小于 阈值, 则停止训练
                break

    def predict(self, X):
        """
        对 测试 数据进行预测, 返回预测的标签

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :return:
        """
        N = np.shape(X)[0]  # 样本的个数

        F_0 = self.G[0]  # G中 第一个 存储的是 初始化情况

        F = np.transpose([F_0] * N)  # shape : (K, N)

        for alpha, DT_list in self.G[1:]:

            for k in range(self.K):

                DT = DT_list[k]

                y_predict = (self.K / (self.K - 1)) * (DT.predict(X))  # shape:(N,)

                F[k] += alpha * y_predict  # F[k]  shape:(N,)

        G = self.softmax(F)

        G_label = np.argmax(G, axis=0)

        return G_label

    def predict_proba(self, X):
        """
        对 测试 数据进行预测, 返回预测的 概率值

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :return:
        """

        F = self.G[0]  # 第一个 存储的是 初始化情况

        for alpha, DT_list in self.G[1:]:

            for k in range(self.K):
                DT = DT_list[k]

                y_predict = (self.K / (self.K - 1)) * (DT.predict(X))  # shape:(N,)

                DT_list.append(DT)

                F[k] += alpha * y_predict  # F[k]  shape:(N,)

        G = self.softmax(F)

        return G


    def score(self, X, y):
        """
        使用 测试数据集 对模型进行评价, 返回正确率

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :param y: 标签数据 , shape=(N_sample,)
        :return:  正确率 accuracy
        """

        N = np.shape(X)[0]  # 样本的个数

        G = self.predict(X)

        err_arr = np.ones(N, dtype=int)
        err_arr[G == y] = 0
        err_rate = np.mean(err_arr)

        accuracy = 1 - err_rate

        return accuracy


class Test:

    def test_tiny_regress_dataset(self):
        """

        利用 https://blog.csdn.net/zpalyq110/article/details/79527653 中的数据集
        测试  GBDT  回归

        :return:
        """

        # 获取训练集

        dataset = np.array(
            [[5, 20, 1.1],
             [7, 30, 1.3],
             [21, 70, 1.7],
             [30, 60, 1.8],
             ])
        columns = ['id', 'age', 'weight', 'label']

        X = dataset[:, 0:2]
        y = dataset[:, 2]

        # 开始时间
        start = time.time()

        # 创建决策树
        print('start create model')

        clf = GBDT_Regressor(max_iter=10, max_depth=3)
        clf.fit(X, y, learning_rate=0.1)

        print(' model complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        # 测试数据集
        X_test = np.array([
            [25, 65]
        ])
        y_predict = clf.predict(X_test)

        print('res: ', y_predict)

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

        # 将数据集以9:1的比例随机分为训练集和测试集，为了重现随机分配设置随机种子，即random_state参数
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=188)

        # 实例化估计器对象
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        gbr = ensemble.GradientBoostingRegressor(**params)

        # 估计器拟合训练数据
        gbr.fit(X_train, y_train)

        # 训练完的估计器对测试数据进行预测
        y_pred = gbr.predict(X_test)

        # 输出特征重要性列表
        # print(gbr.feature_importances_)

        start = time.time()
        print('start create model')

        clf = GBDT_Regressor(max_iter=250, max_depth=4)
        clf.fit(X_train, y_train, learning_rate=0.01)

        print(' model complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        y_pred_test = clf.predict(X_test)

        print('by sklearn , the squared_error:', mean_squared_error(y_test, y_pred))  # 8

        print('by xrh , the squared_error:', mean_squared_error(y_test, y_pred_test))  #

    def test_tiny_2classification_dataset(self):
        """

        利用 https://blog.csdn.net/zpalyq110/article/details/79527653 中的数据集
        测试  GBDT  回归

        :return:
        """

        dataset = np.array(
            [[5, 20, 0],
             [7, 30, 0],
             [21, 70, 1],
             [30, 60, 1],
             ])
        columns = ['age', 'weight', 'label']

        X = dataset[:, 0:2]
        y = dataset[:, 2]

        clf = GBDT_2Classifier(error_rate_threshold=0.0, max_iter=5, max_depth=3)
        clf.fit(X, y, learning_rate=0.1)

        X_test = np.array(
            [[25, 65]])

        print('y predict:', clf.predict(X_test))

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

            # 显然这会导致 正负 样本的 分布不均衡, 1 的样本很少(10%), 而0 的很多
            if int(curLine[0]) == 0:
                labelArr.append(1)
            else:
                labelArr.append(0)

            # if int(curLine[0]) <= 5:
            #     labelArr.append(1)
            # else:
            #     labelArr.append(0)

            cnt += 1

        fr.close()

        # 返回数据集和标记
        return dataArr, labelArr


    def test_Mnist_dataset_2classification(self, n_train, n_test):
        """
        将 Mnist (手写数字) 数据集 转变为 二分类 数据集

        测试 GBDT, 并对 模型效果做出评估

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

        '''
        sklearn GradientBoostingClassifier 调参：

        loss：损失函数。有deviance和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
        n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
        learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
        subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
        init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管

        由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
        max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
        max_depth：CART最大深度，默认为None
        min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
        min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
        min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
        min_leaf_nodes：最大叶子节点数
        '''

        """
        sklearn 性能指标
        参数 learning_rate=0.1, n_estimators=50 , max_depth=3
        
        train data, row num:6000 , column num:784 
        training cost time : 9.30972957611084
        test data, row num:1000 , column num:784 
        test dataset accuracy: 0.976 
        
        """

        # clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50
        #                                   , max_depth=3
        #                                 )
        # clf.fit(trainDataArr, trainLabelArr)

        clf = GBDT_2Classifier( error_rate_threshold=0.01, max_iter=30, max_depth=3 )
        clf.fit(trainDataArr, trainLabelArr,learning_rate=0.2)


        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData_2classification('../Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        # print('test dataset accuracy: {} '.format(clf.score(testDataArr, testLabelArr)))

        # 模型评估

        y_pred = clf.predict(testDataArr)
        y_true = testLabelArr

        # 1.正确率
        print('test dataset accuracy: {} '.format(accuracy_score(y_true, y_pred)))

        print('====================')

        # 2.精确率

        # print(precision_score(y_true, y_pred, average='macro'))  #
        # print(precision_score(y_true, y_pred, average='micro'))  #
        # print(precision_score(y_true, y_pred, average='weighted'))  #

        print('pos-1 precision: ', precision_score(y_true, y_pred, average='binary'))

        precision_list = precision_score(y_true, y_pred, average=None)

        print('neg-0 precision:{}, pos-1 precision:{}  '.format(precision_list[0], precision_list[1]))

        print('====================')

        # 3. 召回率

        # print(recall_score(y_true, y_pred, average='macro'))  #
        # print(recall_score(y_true, y_pred, average='micro'))  #
        # print(recall_score(y_true, y_pred, average='weighted'))  #

        print('pos-1 recall: ', recall_score(y_true, y_pred, average='binary'))

        recall_list = recall_score(y_true, y_pred, average=None)

        print('neg-0 recall:{}, pos-1 recall:{}  '.format(recall_list[0], recall_list[1]))

        print('====================')

        # 4. F1-score

        # print(f1_score(y_true, y_pred, average='macro'))
        # print(f1_score(y_true, y_pred, average='micro'))
        # print(f1_score(y_true, y_pred, average='weighted'))

        print('pos-1 f1_score: ', f1_score(y_true, y_pred, average='binary'))

        f1_score_list = f1_score(y_true, y_pred, average=None)

        print('neg-0 f1_score:{}, pos-1 f1_score:{}  '.format(f1_score_list[0], f1_score_list[1]))

        print('====================')

        # 画出 P-R 曲线

        # sklearn 的 GBDT 作为基线
        clf2 = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=30
                                          , max_depth=3
                                          )

        clf2.fit(trainDataArr, trainLabelArr)
        y_pred = clf.predict(testDataArr)

        y_scores = clf.predict_proba(testDataArr)

        y_true = testLabelArr

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        y_scores2 = clf2.predict_proba(testDataArr)[:, 1]  # 第 1 列 , 表示为 正例的概率

        precision2, recall2, thresholds2 = precision_recall_curve(y_true, y_scores2)

        # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot()

        plt.plot(recall, precision, label="GDBT_2Classifier(xrh)", color='navy')  #

        plt.plot(recall2, precision2, label="GradientBoostingClassifier(sklearn)", color='turquoise')

        plt.title(' Precision-Recall curve ')

        # plt.ylim([0.0, 1.05]) # Y 轴的取值范围
        # plt.xlim([0.0, 1.0]) # X 轴的取值范围

        plt.xlabel("recall")
        plt.ylabel("precision")

        plt.legend(loc=(0, -.38), prop=dict(size=14))  # 图例

        plt.show()

        # ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        fpr2, tpr2, _ = roc_curve(y_true, y_scores2)

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

        plt.plot(fpr, tpr, label="GDBT_2Classifier(xrh)", color='darkorange')  #

        plt.plot(fpr2, tpr2, label="GradientBoostingClassifier(sklearn)", color='turquoise')

        # plt.xlim( [0.0, 1.0] )
        # plt.ylim( [0.0, 1.05] )

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend(loc=(0, -.38), prop=dict(size=14))  # 图例

        plt.show()


    def test_tiny_multiclassification_dataset(self):
        """
        使用 https://zhuanlan.zhihu.com/p/91652813 中的数据测试 GBDT-多分类

        :return:
        """

        X_train =np.array( [[6],
                   [12],
                   [14],
                   [18],
                   [20],
                   [65],
                   [31],
                   [40],
                   [1],
                   [2],
                   [100],
                   [101],
                   [65],
                   [54],
                   ])

        y_train = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2], [2], [2]]).ravel()

        clf = GBDT_MultiClassifier( error_rate_threshold=0.01, max_iter=5, max_depth=1 )
        clf.fit(X_train, y_train,learning_rate=1)


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

        """
        调参：
        loss：损失函数。有deviance和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
        n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
        criterion: 切分叶子节点时, 选择切分特征考虑的误差函数, 默认是 “ friedman_mse”（ Friedman 均方误差），“ mse”（均方误差）和“ mae”（均绝对误差）
        learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
        subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
        init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管
        由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
        max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
        max_depth：CART最大深度，默认为None
        min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
        min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
        min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
        min_leaf_nodes：最大叶子节点数
        
        ref: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        
        测试1: 
        max_depth=3, n_estimators=30, learning_rate=0.8, 
        n_train=60000
        n_test=10000
        训练时间 : 795.5719292163849
        准确率: 0.8883 
        
        测试2:
        max_depth=3, n_estimators=20, learning_rate=0.5, 
        n_train=60000
        n_test=10000
        训练时间 : 589 s
        准确率: 0.9197 
        
        """

        clf = GradientBoostingClassifier(loss='deviance',criterion='mse', n_estimators=20, learning_rate=0.5,
                                         max_depth=3)

        clf.fit(trainDataArr, trainLabelArr)



        # clf = GBDT_MultiClassifier( error_rate_threshold=0.01, max_iter=20, max_depth=3 )
        # clf.fit( trainDataArr, trainLabelArr,learning_rate= 0.5 ) #

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        print('test dataset accuracy: {} '.format(clf.score(testDataArr, testLabelArr)))

    def test_iris_dataset(self):

        # 使用iris数据集，其中有三个分类， y的取值为0,1，2
        X, y = datasets.load_iris(True)  # 包括150行记录
        # 将数据集一分为二，训练数据占80%，测试数据占20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # clf = GradientBoostingClassifier(loss='deviance',n_estimators=3, learning_rate=0.1,
        #                                  max_depth=2)
        # clf.fit(X_train, y_train)

        clf = GBDT_MultiClassifier( error_rate_threshold=0.01, max_iter=5, max_depth=3 )
        clf.fit(X_train, y_train,learning_rate=0.8)

        print(clf.score(X_test, y_test))


if __name__ == '__main__':
    test = Test()

    # test.test_regress_dataset()

    # test.test_Mnist_dataset_2classification(60000,10000)

    # test.test_tiny_2classification_dataset()

    # test.test_tiny_multiclassification_dataset()

    test.test_Mnist_dataset(60000,10000)

    # test.test_iris_dataset()


