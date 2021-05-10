import numpy as np

import time
from deprecated import deprecated

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

    1. 回归 ( 使用平方损失函数)
    2. 二分类 ( 使用 交叉熵损失函数)
    3. 多分类 ( 使用 对数损失函数 )

    ref: https://microstrong.blog.csdn.net/article/details/103655906

    实现细节：
    (1) 使用  完全贪心算法 划分子节点 , 未实现 近似算法
    (2) 未实现 稀疏性感知（Sparsity Awareness）
    (3) 加权分位数略图（Weighted Quantile Sketch）
    (4) 未实现 分块并行

    Author: xrh
    Date: 2021-05-07


    test1: 多分类任务

    数据集：Mnist
    参数: error_rate_threshold=0.01, max_iter=20, max_depth=3 , learning_rate=0.5
    训练集数量：
    测试集数量：
    正确率：
    模型训练时长：

    """

    def __init__(self,  print_log=False,
                        error_rate_threshold=0.05 ,
                        max_iter=10,
                        max_depth=1,
                        objective='binary:logistic',
                        num_class=None ,
                        base_score=0.0,
                        gama=0.0,
                        reg_lambda=1.0 ):
        """
        :param print_log: 打印 Cart树
        :param error_rate_threshold: 训练中止条件, 若当前得到的基分类器的组合 的错误率 小于阈值, 则停止训练;
                                     若为 回归问题, 错误率为 MSE (平均 平方损失)
        :param max_iter: 最大迭代次数
        :param max_depth: CART 回归树 的最大深度

        :param objective:  目标函数选择
                            (1) reg:squarederror：损失平方回归
                            (2) binary:logistic：二元分类的逻辑回归，输出概率
                            (3) multi:softmax：使用softmax目标函数 进行多类分类，还需要设置 num_class（类数）
        :param num_class: 多分类时的 分类数目
        :param base_score: 所有实例的 初始预测得分 F_0
        :param gama:  损失函数中 树的总叶子个数 T 的系数, 可以控制模型的复杂度
        :param reg_lambda: 目标函数中使用 L2 正则化时控制 正则化的强度


        """
        self.N=0 # 训练集的样本个数

        self.print_log = print_log

        # 训练中止条件 error_rate  < self.error_rate_threshold ( 若当前得到的基分类器的组合 的错误率 小于阈值, 则停止训练)
        self.error_rate_threshold = error_rate_threshold

        # 最大迭代次数
        self.max_iter = max_iter

        # CART 回归树 的最大深度
        self.max_depth = max_depth

        # 目标函数选择
        self.objective=objective

        #  多分类时的 分类数目
        self.num_class=num_class

        # 所有实例的 初始预测得分 F_0
        self.base_score= base_score

        # 损失函数中 树的总叶子个数 T 的系数, 可以控制模型的复杂度
        self.gama = gama

        # 目标函数中使用 L2 正则化时控制 正则化的强度
        self.reg_lambda= reg_lambda

        self.G = []  # 弱分类器 集合

    def sigmoid( self , X ):
        """
        sigmoid 激活函数
        :param X:
        :return:
        """
        return 1 / (1 + np.exp(-X))


    def softmax(self,X):
        """
        softmax处理，将结果转化为概率

        解决了 softmax的 溢出问题

        np.nan_to_num : 使用 0代替数组x中的nan元素，使用有限的数字代替 inf元素

        ref: sklearn 源码
            MultinomialDeviance -> def negative_gradient

        :param X: shape (K,N)
        :return: shape (N,)
        """

        return  np.nan_to_num( np.exp(X - logsumexp(X, axis=0)) )  # softmax处理，将结果转化为概率

    def init_y_predict(self,F_0):
        """
        m=0 :
        初始化 y_predict
        :param F_0:
        :return:
        """
        y_predict=None

        if self.objective ==  "reg:squarederror": # 回归
            y_predict = np.array( [ F_0 ] * self.N )

        elif self.objective == "binary:logistic": # 二分类
            y_predict = np.array( [ self.sigmoid(F_0) ] * self.N)

        return y_predict

    def cal_g_h(self,y , y_predict):
        """
        计算损失函数 一阶梯度 和 二阶梯度

        g : 损失函数 对 F 的一阶梯度 , 相当于 GBDT 中的残差 r
        h: 损失函数 对 F 的 二阶梯度

        ref: https://www.cnblogs.com/nxf-rabbit75/p/10440805.html

        :param y:
        :param y_predict:
        :return:
        """
        g,h=0,0

        if self.objective ==  "reg:squarederror": # 回归

        # TODO: 若采用:
        #     g =  y - y_predict
        #     h=  -np.ones_like(g)
        #   将导致 计算的 max_gain<0 ,  由于 max_gain < self.gama  所以 节点不再继续分裂,导致树的高度过低, 模型欠拟合

            g = y_predict - y
            h=  np.ones_like(g)


        elif self.objective == "binary:logistic": # 二分类
            g = y_predict - y
            h = y_predict*(1-y_predict)

        return g,h

    def update_y_predict(self,F):
        """

        更新 本轮迭代的 y_predict

        :param F:
        :return:
        """
        y_predict=None

        if self.objective ==  "reg:squarederror": # 回归
            y_predict = F

        elif self.objective == "binary:logistic": # 二分类
            y_predict = self.sigmoid(F)

        return y_predict

    def model_error_rate(self,y,y_predict):
        """
        计算 当前 所有弱分类器加权 得到的 最终分类器 的 误差率

        若为 回归问题, 误差率 为 平均平方误差损失 ( MSE = mean_squared_error )

        :param y:
        :param y_predict:
        :return:
        """
        N = len(y)

        error_rate=None

        if self.objective == "reg:squarederror":  # 回归

            error_rate = np.average(np.square(y_predict - y)) #  error_rate  为 平均平方误差损失 ( mean_squared_error )

        elif self.objective == "binary:logistic":  # 二分类

            y_predict[y_predict >= 0.5] = 1  # 概率 大于 0.5 被标记为 正例
            y_predict[y_predict < 0.5] = 0  # 概率 小于 0.5 被标记为 负例

            err_arr = np.ones(N, dtype=int)
            err_arr[y_predict == y] = 0

            error_rate = np.mean(err_arr) # loss 为 分类错误率

        return error_rate

    def fit(self, X, y ,learning_rate,train_error=True):
        """

        用 训练数据 拟合模型

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :param y: 标签数据 , shape=(N_sample,)
        :param learning_rate: 学习率
        :param train_error:  在训练模型时, 计算模型在 训练集上的误差率
        :return:
        """

        self.N = np.shape(X)[0]  # 样本的个数

        if self.objective == "binary:logistic" or 'reg:squarederror': # 二分类 or 回归

            F_0 = self.base_score

            self.G.append(F_0)

            F = np.array( [self.base_score] * self.N ) # shape: (N, )

            y_predict= self.init_y_predict(F_0)

            feature_value_set = RegresionTree_XGBoost.get_feature_value_set(X)  # 可供选择的特征集合 , 包括 (特征, 切分值)

            for m in range(self.max_iter):  # 进行 第 m 轮迭代

                g,h = self.cal_g_h( y,y_predict )

                RT = RegresionTree_XGBoost(gama=self.gama,reg_lambda=self.reg_lambda, max_depth=self.max_depth, print_log=self.print_log)

                RT.fit( X, g, h, feature_value_set=feature_value_set )

                f_m = RT.predict(X) # 第 m 颗树

                self.G.append( (learning_rate, RT) )  # 存储 基分类器

                F = F + learning_rate * f_m

                y_predict = self.update_y_predict(F)

                if train_error:

                    y_predict_copy = y_predict.copy() # y_predict 下一轮迭代 还要使用, 不能被修改

                    error_rate= self.model_error_rate(y,y_predict_copy)

                    print( 'round:{}, error_rate :{}'.format( m, error_rate ) )
                    print( '======================' )

                    if error_rate < self.error_rate_threshold:  # 错误率 已经小于 阈值, 则停止训练
                        break


    def predict(self, X):
        """
        对 测试 数据进行预测, 返回预测值

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :return:
        """

        f = 0  # 最终分类器

        f += self.G[0]  # 第一个 存储的是 初始化情况

        y_predict=None # 输出的标签

        for alpha, RT in self.G[1:]:
            f_m = RT.predict(X)
            f += alpha * f_m

        if self.objective == "reg:squarederror":  # 回归

            y_predict=f

        elif self.objective == "binary:logistic":

            y_predict = self.sigmoid(f)

            y_predict[y_predict >= 0.5] = 1  # 概率 大于 0.5 被标记为 正例
            y_predict[y_predict < 0.5] = 0  # 概率 小于 0.5 被标记为 负例

        return y_predict


    def score(self, X, y):
        """
        使用 测试数据集 对模型进行评价, 返回正确率

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :param y: 标签数据 , shape=(N_sample,)
        :return:  错误率 error

        """

        y_predict = self.predict(X)

        loss=self.model_error_rate(y,y_predict)

        return loss


class Test:

    def test_tiny_regress_dataset(self):
        """

        利用 https://blog.csdn.net/zpalyq110/article/details/79527653 中的数据集
        测试  回归 问题

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

        clf = XGBoost_v1( error_rate_threshold=0.01,objective='reg:squarederror', max_iter=5, max_depth=3,gama=0.0,reg_lambda=0.0 )
        clf.fit( X, y,learning_rate=0.1 )

        print(' model complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        # 测试数据集
        X_test = np.array([
            [25, 65]
        ])
        y_predict = clf.predict( X_test )

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=188)


        # 输出特征重要性列表

        start = time.time()
        print('start create model')

        clf = XGBoost_v1( error_rate_threshold=0.1,
                          objective='reg:squarederror',
                          max_iter=100,
                          max_depth=3,
                          gama=1.0,
                          reg_lambda=0.0 )
        clf.fit( X_train, y_train,learning_rate=0.1 )

        print(' model complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        y_pred_test = clf.predict(X_test)

        print('by xrh , the squared_error:', mean_squared_error(y_test, y_pred_test))  #

    def test_tiny_2classification_dataset(self):
        """

        利用 https://blog.csdn.net/anshuai_aw1/article/details/82970489 中的数据集
        测试  xgboost

        :return:
        """

        dataset = np.array(
            [[1, -5, 0],
             [2, 5, 0],
             [3, -2, 1],
             [1, 2, 1],
             [2, 0, 1],
             [6, -5, 1],
             [7, 5, 1],
             [6, -2, 0],
             [7, 2, 0],
             [6, 0, 1],
             [8, -5, 1],
             [9, 5, 1],
             [10, -2, 0],
             [8, 2, 0],
             [9, 0, 1]
             ])

        X = dataset[:, 0:2]
        y = dataset[:, 2]

        clf = XGBoost_v1(error_rate_threshold=0.0, max_iter=2, max_depth=3,objective="binary:logistic")
        clf.fit(X, y, learning_rate=0.1)

        X_test = np.array(
            [[9, 0]])

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


        clf = XGBoost_v1( error_rate_threshold=0.01,objective='binary:logistic' ,max_iter=30, max_depth=3 )
        clf.fit(trainDataArr, trainLabelArr,learning_rate=0.5)


        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData_2classification('../Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        accuracy=1-clf.score(testDataArr, testLabelArr)
        print('test dataset accuracy: {} '.format( accuracy ))

        # 模型评估

        # y_pred = clf.predict(testDataArr)
        # y_true = testLabelArr
        #
        # # 1.正确率
        # print('test dataset accuracy: {} '.format(accuracy_score(y_true, y_pred)))
        #
        # print('====================')
        #
        # # 2.精确率
        #
        # # print(precision_score(y_true, y_pred, average='macro'))  #
        # # print(precision_score(y_true, y_pred, average='micro'))  #
        # # print(precision_score(y_true, y_pred, average='weighted'))  #
        #
        # print('pos-1 precision: ', precision_score(y_true, y_pred, average='binary'))
        #
        # precision_list = precision_score(y_true, y_pred, average=None)
        #
        # print('neg-0 precision:{}, pos-1 precision:{}  '.format(precision_list[0], precision_list[1]))
        #
        # print('====================')
        #
        # # 3. 召回率
        #
        # # print(recall_score(y_true, y_pred, average='macro'))  #
        # # print(recall_score(y_true, y_pred, average='micro'))  #
        # # print(recall_score(y_true, y_pred, average='weighted'))  #
        #
        # print('pos-1 recall: ', recall_score(y_true, y_pred, average='binary'))
        #
        # recall_list = recall_score(y_true, y_pred, average=None)
        #
        # print('neg-0 recall:{}, pos-1 recall:{}  '.format(recall_list[0], recall_list[1]))
        #
        # print('====================')
        #
        # # 4. F1-score
        #
        # # print(f1_score(y_true, y_pred, average='macro'))
        # # print(f1_score(y_true, y_pred, average='micro'))
        # # print(f1_score(y_true, y_pred, average='weighted'))
        #
        # print('pos-1 f1_score: ', f1_score(y_true, y_pred, average='binary'))
        #
        # f1_score_list = f1_score(y_true, y_pred, average=None)
        #
        # print('neg-0 f1_score:{}, pos-1 f1_score:{}  '.format(f1_score_list[0], f1_score_list[1]))
        #
        # print('====================')
        #
        # # 画出 P-R 曲线
        #
        # # sklearn 的 GBDT 作为基线
        # clf2 = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=30
        #                                   , max_depth=3
        #                                   )
        #
        # clf2.fit(trainDataArr, trainLabelArr)
        # y_pred = clf.predict(testDataArr)
        #
        # y_scores = clf.predict_proba(testDataArr)
        #
        # y_true = testLabelArr
        #
        # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        #
        # y_scores2 = clf2.predict_proba(testDataArr)[:, 1]  # 第 1 列 , 表示为 正例的概率
        #
        # precision2, recall2, thresholds2 = precision_recall_curve(y_true, y_scores2)
        #
        # # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        # # disp.plot()
        #
        # plt.plot(recall, precision, label="GDBT_2Classifier(xrh)", color='navy')  #
        #
        # plt.plot(recall2, precision2, label="GradientBoostingClassifier(sklearn)", color='turquoise')
        #
        # plt.title(' Precision-Recall curve ')
        #
        # # plt.ylim([0.0, 1.05]) # Y 轴的取值范围
        # # plt.xlim([0.0, 1.0]) # X 轴的取值范围
        #
        # plt.xlabel("recall")
        # plt.ylabel("precision")
        #
        # plt.legend(loc=(0, -.38), prop=dict(size=14))  # 图例
        #
        # plt.show()
        #
        # # ROC 曲线
        # fpr, tpr, _ = roc_curve(y_true, y_scores)
        #
        # fpr2, tpr2, _ = roc_curve(y_true, y_scores2)
        #
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        #
        # plt.plot(fpr, tpr, label="GDBT_2Classifier(xrh)", color='darkorange')  #
        #
        # plt.plot(fpr2, tpr2, label="GradientBoostingClassifier(sklearn)", color='turquoise')
        #
        # # plt.xlim( [0.0, 1.0] )
        # # plt.ylim( [0.0, 1.05] )
        #
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        #
        # plt.legend(loc=(0, -.38), prop=dict(size=14))  # 图例
        #
        # plt.show()


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

    # test.test_tiny_regress_dataset()

    # test.test_regress_dataset()

    test.test_Mnist_dataset_2classification(6000,1000)

    # test.test_tiny_2classification_dataset()

    # test.test_tiny_multiclassification_dataset()

    # test.test_Mnist_dataset(60000,10000)

    # test.test_iris_dataset()


