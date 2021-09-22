#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import time
import os
import pickle
from deprecated import deprecated

from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from updater_colmaker_xrh import *
from activation_xrh import *
from sparse_vector_xrh import *
from utils_xrh import *

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


class XGBoost:
    """

    简版 XGBoost 模型

    功能包括:

    1. 回归 ( 使用平方损失函数 )
    2. 二分类 ( 使用 交叉熵损失函数 )
    3. 多分类 ( 使用 对数损失函数 )

    ref:
    1. XGBoost: A Scalable Tree Boosting System
    2. https://github.com/dmlc/xgboost/

    实现细节：
    (1) 对 样本数据 按照特征分块, 在块内按照特征值的大小排序

    (2) 使用 完全贪心算法 寻找最优切分点

    (3) 实现了 稀疏感知 (Sparsity-aware Split Finding) , 在寻找 最优切分点时跳过特征值为缺失值的样本行,
        提升了模型在高维稀疏特征下的性能

    待实现:
    (4) 近似算法寻找最优切分点: 在切分时我们只考虑特征值的分位点即可, 而不是每一个特征值都考虑

    (5) 获取分位点时, 需要 将所有值读入内存并排序 ;当内存不够时, 无法做到对所有特征值排序, 此时可采用分位点估计算法
        即 加权分位数速写 (Weighted Quantile Sketch), 来估算出分位点的值

    (6) 寻找最优分裂点时, 可以使用多线程并行

    Author: xrh
    Date: 2021-05-29

    """

    def __init__(self,
                 use_pretrained=False, model_path='models/xgboost.model',
                 print_log=False,
                 max_iter=10,
                 max_depth=3,
                 objective='binary:logistic',
                 num_class=None,
                 base_score=0.0,
                 gama=0.1,
                 reg_lambda=1.0,
                 min_child_weight=0,
                 tree_method='exact',
                 sketch_eps=0.3,
                 missing={np.nan, 0}
                 ):
        """
        :param print_log: 打印 Cart树

        :param max_iter: 最大迭代次数
        :param max_depth: CART 回归树 的最大深度

        :param objective:  目标函数选择
                            (1) reg:squarederror：平方损失, 适用于回归
                            (2) binary:logistic：二元分类的逻辑回归，输出概率
                            (3) multi:softmax：使用softmax目标函数 进行多类分类，还需要设置 num_class（类数）

        :param num_class: 多分类时的 分类数目
        :param base_score: 所有实例的 初始预测得分 F_0
        :param gama:  损失函数中 树的总叶子个数 T 的系数, 可以控制模型的复杂度
        :param reg_lambda: 目标函数中使用 L2 正则化时控制 正则化的强度
        :param min_child_weight: 搜索最佳切分点时, 若 min_child_weight < Min(HL, HR) 则放弃此切分点

        :param tree_method： 指定了构建树的算法，可以为下列的值：
                            (1)'exact'： 使用 exact greedy 完全贪心算法分裂节点
                            (2)'approx'： 使用近似算法分裂节点

        :param sketch_eps： 设定分桶的步长为: 二阶梯度的区间和 * sketch_eps ;
                            取值范围为 (0,1), 默认值为 0.3, 此时 每一个特征划分 3个桶 ;
                            它仅仅用于 tree_medhodd='approx'

        :param missing： 一个集合，代表发生了数据缺失。默认为 {np.nan, 0}

        """

        self.model_path = model_path

        if not use_pretrained:

            self.print_log = print_log

            # 最大迭代次数
            self.max_iter = max_iter

            # CART 回归树 的最大深度
            self.max_depth = max_depth

            # 目标函数选择
            self.objective = objective

            #  多分类时的 分类数目
            self.num_class = num_class

            # 所有实例的 初始预测得分 F_0
            self.base_score = base_score

            # 损失函数中 树的总叶子个数 T 的系数, 可以控制模型的复杂度
            self.gama = gama

            # 目标函数中使用 L2 正则化时控制 正则化的强度
            self.reg_lambda = reg_lambda

            # 搜索最佳切分点时, 若 min_child_weight < Min(HL, HR) 则放弃此切分点
            self.min_child_weight = min_child_weight

            # 构建树时 切分子节点的算法
            self.tree_method = tree_method

            # 分桶的步长
            self.sketch_eps = sketch_eps

            # 缺失值集合
            self.missing = missing

            self.G = []  # 弱分类器 集合

        else:  # 载入预训练的模型

            self.load()


    def save(self):
        """
        保存训练好的模型

        :return:
        """

        save_dict = {}

        save_dict['model_path'] = self.model_path
        save_dict['print_log'] = self.print_log
        save_dict['max_iter'] = self.max_iter
        save_dict['max_depth'] = self.max_depth
        save_dict['objective'] = self.objective
        save_dict['num_class'] = self.num_class

        save_dict['base_score'] = self.base_score
        save_dict['gama'] = self.gama
        save_dict['reg_lambda'] = self.reg_lambda
        save_dict['min_child_weight'] = self.min_child_weight

        save_dict['tree_method'] = self.tree_method
        save_dict['sketch_eps'] = self.sketch_eps
        save_dict['missing'] = self.missing
        save_dict['G'] = self.G

        with open(self.model_path, 'wb') as f:
            pickle.dump(save_dict, f)

        print("Save model successful!")

    def load(self):
        """
        读取预训练的模型

        :param file_path:
        :return:
        """

        with open(self.model_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.model_path = save_dict['model_path']
        self.print_log = save_dict['print_log']
        self.max_iter = save_dict['max_iter']
        self.max_depth = save_dict['max_depth']
        self.objective = save_dict['objective']
        self.num_class = save_dict['num_class']

        self.base_score = save_dict['base_score']
        self.gama = save_dict['gama']
        self.reg_lambda = save_dict['reg_lambda']
        self.min_child_weight = save_dict['min_child_weight']

        self.tree_method = save_dict['tree_method']
        self.sketch_eps = save_dict['sketch_eps']
        self.missing = save_dict['missing']
        self.G = save_dict['G']

        print("Load model successful!")

    def init_y_predict(self, F_0, N):
        """

        初始化 y_predict

        :param F_0:
        :param N: 样本个数
        :return:
        """
        y_predict = None

        if self.objective == "reg:squarederror":  # 回归
            y_predict = np.array([F_0] * N)

        elif self.objective == "binary:logistic":  # 二分类
            y_predict = np.array([Activation.sigmoid(F_0)] * N)

        elif self.objective == "multi:softmax":

            y_predict = np.transpose([Activation.softmax(F_0)] * N)  # shape : (K, N)

        return y_predict

    def cal_g_h(self, y, y_predict):
        """
        计算损失函数的 一阶梯度 和 二阶梯度

        g - 损失函数 对 打分函数 F 的一阶梯度 , 相当于 GBDT 中的残差 r
        h - 损失函数 对 打分函数 F 的 二阶梯度

        ref: https://www.cnblogs.com/nxf-rabbit75/p/10440805.html

        :param y:
        :param y_predict:
        :return:
        """
        g, h = 0, 0

        if self.objective == "reg:squarederror":  # 回归

            g = y_predict - y
            h = np.ones_like(g)

        elif self.objective == "binary:logistic":  # 二分类

            g = y_predict - y
            h = y_predict * (1 - y_predict)

        elif self.objective == "multi:softmax":  # 二分类

            g = (y_predict - y)

            # 以下实现 参考 xgboost 源码
            # ref:
            # https://github.com/dmlc/xgboost/blob/master/src/objective/multiclass_obj.cu
            # class SoftmaxMultiClassObj
            # -> void GetGradient(

            h = 2 * y_predict * (1 - y_predict)
            h[h < 1e-16] = 1e-16  # h 不能小于0

        return g, h

    def update_y_predict(self, F):
        """

        更新 本轮迭代的 y_predict

        :param F:
        :return:
        """
        y_predict = None

        if self.objective == "reg:squarederror":  # 回归
            y_predict = F

        elif self.objective == "binary:logistic":  # 二分类
            y_predict = Activation.sigmoid(F)

        elif self.objective == "multi:softmax":

            y_predict = Activation.softmax(F)

        return y_predict

    def model_error_rate(self, y, y_predict):
        """
        计算 当前 所有弱分类器加权 得到的 最终分类器 的 误差率

        若为 回归问题, 误差率 为 平均平方误差损失 ( MSE = mean_squared_error )

        :param y:
        :param y_predict:
        :return:
        """
        N = len(y)

        error_rate = None

        if self.objective == "reg:squarederror":  # 回归

            error_rate = np.average(np.square(y_predict - y))  # error_rate  为 平均平方误差损失 ( mean_squared_error )

        elif self.objective == "binary:logistic":  # 二分类

            y_predict[y_predict >= 0.5] = 1  # 概率 大于 0.5 被标记为 正例
            y_predict[y_predict < 0.5] = 0  # 概率 小于 0.5 被标记为 负例

            err_arr = np.ones(N, dtype=int)
            err_arr[y_predict == y] = 0

            error_rate = np.mean(err_arr)  # loss 为 分类错误率

        elif self.objective == "multi:softmax":  # 多分类

            y_label = np.argmax(y_predict, axis=0)  # 取 概率最大的 作为 预测的标签

            err_arr = np.ones(N, dtype=int)
            err_arr[y_label == y] = 0
            error_rate = np.mean(err_arr)  # 计算训练误差

        return error_rate

    def fit(self, X, y, learning_rate, error_rate_threshold=0.01, print_error_rate=True):
        """

        用训练数据拟合模型

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :param y: 标签数据 , shape=(N_sample,)
        :param learning_rate: 学习率
        :param error_rate_threshold: 训练中止条件, 若当前得到的基分类器的组合的在训练集上的错误率 小于阈值, 则停止训练;
                                     若为 回归问题, 错误率为 MSE (平均 平方损失)
        :param print_error_rate: 在训练模型时, 计算并输出模型在训练集上的误差率
        :return:
        """

        N = np.shape(X)[0]  # 样本的个数

        X_DMatrix = DMatrix(X, missing=self.missing)

        if self.objective == "multi:softmax":  # 多分类

            F_0 = np.array([self.base_score] * self.num_class)  # shape : (K,)

            self.G.append(F_0)

            F = np.transpose([F_0] * N)  # 对 F_0 进行复制, F shape : (K, N)

            y_predict = self.init_y_predict(F_0, N)  # shape : (K, N)

            y_one_hot = Utils.convert_to_one_hot(x=y, class_num=self.num_class).T  # shape: (K,N)

            for m in range(1, self.max_iter+1):  # 进行 第 m 轮迭代

                DT_list = []

                for k in range(self.num_class):  # 依次训练 K 个 二分类器

                    print('======= train No.{} 2Classifier ======='.format(k))

                    g, h = self.cal_g_h(y_one_hot[k], y_predict[k])  # y_predict[k]  shape:(N,)

                    # 训练 用于 2分类的 回归树
                    RT = Builder(gama=self.gama,
                                 reg_lambda=self.reg_lambda,
                                 max_depth=self.max_depth,
                                 min_child_weight=self.min_child_weight,
                                 tree_method=self.tree_method,
                                 sketch_eps=self.sketch_eps,
                                 print_log=self.print_log)

                    RT.fit(X_DMatrix, g, h)

                    f_m = RT.inference(X)  # shape:(N,)

                    DT_list.append(RT)

                    F[k] = F[k] + learning_rate * f_m  # F shape : (K, N), F[k]  shape:(N,)

                y_predict = self.update_y_predict(F)

                self.G.append((learning_rate, DT_list))  # 存储 基分类器

                # 计算 当前 所有弱分类器加权 得到的 最终分类器 的 分类错误率
                if print_error_rate:

                    y_predict_copy = y_predict.copy()  # y_predict 下一轮迭代 还要使用, 不能被修改

                    error_rate = self.model_error_rate(y, y_predict_copy)

                    print('round:{}, error_rate :{}'.format(m, error_rate))
                    print('======================')

                    if error_rate < error_rate_threshold:  # 错误率 已经小于 阈值, 则停止训练
                        break

        elif self.objective in ('binary:logistic', 'reg:squarederror'):  # 二分类 or 回归

            F_0 = self.base_score

            self.G.append(F_0)

            F = np.array([self.base_score] * N)  # shape: (N, )

            y_predict = self.init_y_predict(F_0, N)

            for m in range(1, self.max_iter+1):  # 进行 第 m 轮迭代

                g, h = self.cal_g_h(y, y_predict)

                RT = Builder(gama=self.gama,
                             reg_lambda=self.reg_lambda,
                             max_depth=self.max_depth,
                             min_child_weight=self.min_child_weight,
                             tree_method=self.tree_method,
                             sketch_eps=self.sketch_eps,
                             print_log=self.print_log)

                RT.fit(X_DMatrix, g, h)

                f_m = RT.inference(X)  # 第 m 颗树

                self.G.append((learning_rate, RT))  # 存储 基分类器

                F = F + learning_rate * f_m

                y_predict = self.update_y_predict(F)

                if print_error_rate:

                    y_predict_copy = y_predict.copy()  # y_predict 下一轮迭代 还要使用, 不能被修改

                    error_rate = self.model_error_rate(y, y_predict_copy)

                    print('')
                    print('round:{}, error_rate :{}'.format(m, error_rate))
                    print('======================')

                    if error_rate < error_rate_threshold:  # 错误率 已经小于 阈值, 则停止训练
                        break

        self.save()

    def predict_prob(self, X):
        """
        对 测试 数据进行预测, 返回预测的概率( 若为回归, 返回的是回归值 )

        :param X: 特征数据 , shape=(N_sample, N_feature)

        :return:
        """
        N = np.shape(X)[0]

        y_prob = None  # 输出的标签

        F_0 = self.G[0]  # 最终分类器

        if self.objective == "multi:softmax":

            F = np.transpose([F_0] * N)  # shape : (K, N)

            for alpha, DT_list in self.G[1:]:

                for k in range(self.num_class):
                    DT = DT_list[k]

                    f_m = DT.inference(X)

                    F[k] += alpha * f_m  # F[k]  shape:(N,)

            y_prob = Activation.softmax(F)  # F shape:(K,N) ;  y_prob shape:(K,N)

        elif self.objective in ("reg:squarederror", "binary:logistic"):

            F = F_0  # 第一个 存储的是 初始化情况

            for alpha, RT in self.G[1:]:
                f_m = RT.inference(X)
                F += alpha * f_m

            if self.objective == "reg:squarederror":  # 回归

                y_prob = F

            elif self.objective == "binary:logistic":  # 二分类

                y_prob = Activation.sigmoid(F)

        return y_prob

    def predict(self, X):
        """
        对 测试 数据进行预测, 返回预测标签

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :return:
        """
        N = np.shape(X)[0]

        y_predict = None  # 输出的标签

        F_0 = self.G[0]  # 最终分类器

        if self.objective == "multi:softmax":

            F = np.transpose([F_0] * N)  # shape : (K, N)

            for alpha, DT_list in self.G[1:]:

                for k in range(self.num_class):
                    DT = DT_list[k]

                    f_m = DT.inference(X)

                    F[k] += alpha * f_m  # F[k]  shape:(N,)

            prob = Activation.softmax(F)  # F shape:(K,N) ;  prob shape:(K,N)

            y_predict = np.argmax(prob, axis=0)  # y_predict shape:(N,)

        elif self.objective in ("reg:squarederror", "binary:logistic"):

            F = F_0  # 第一个 存储的是 初始化情况

            for alpha, RT in self.G[1:]:
                f_m = RT.inference(X)
                F += alpha * f_m

            if self.objective == "reg:squarederror":  # 回归

                y_predict = F

            elif self.objective == "binary:logistic":  # 二分类

                y_predict = Activation.sigmoid(F)

                y_predict[y_predict >= 0.5] = 1  # 概率 大于 0.5 被标记为 正例
                y_predict[y_predict < 0.5] = 0  # 概率 小于 0.5 被标记为 负例

        return y_predict

    def score(self, X, y):
        """
        使用 测试数据集 对模型进行评价, 返回 正确率(accuracy), 仅适用于分类任务

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :param y: 标签数据 , shape=(N_sample,)
        :return:  错误率 error

        """
        N = X.shape[0]

        y_predict = self.predict(X)

        err_arr = np.ones(N, dtype=int)
        err_arr[y_predict == y] = 0
        error_rate = np.mean(err_arr)

        accuracy = 1 - error_rate

        return accuracy


class EvaluateModel:

    def tow_classify_evaluate(self, y_true, y_pred, y_prob):
        """
        模型评估

        :param y_true: 样本的真实标签
        :param y_pred: 模型预测的标签
        :param y_prob: 模型预测的概率

        :return:
        """

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

        # 5. 画出 P-R 曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

        # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot()

        plt.plot(recall, precision, label="GDBT_2Classifier(xrh)", color='navy')  #


        plt.title(' Precision-Recall curve ')

        # plt.ylim([0.0, 1.05]) # Y 轴的取值范围
        # plt.xlim([0.0, 1.0]) # X 轴的取值范围

        plt.xlabel("recall")
        plt.ylabel("precision")

        plt.legend(loc=(0, -.38), prop=dict(size=14))  # 图例

        plt.show()

        # 6. ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

        plt.plot(fpr, tpr, label="GDBT_2Classifier(xrh)", color='darkorange')  #


        # plt.xlim( [0.0, 1.0] )
        # plt.ylim( [0.0, 1.05] )

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend(loc=(0, -.38), prop=dict(size=14))  # 图例

        plt.show()




class Test:


    def test_boston_regress_dataset(self):
        """

        利用 boston房价 数据集
        测试  xgboost  回归

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

        clf = XGBoost(
                      objective='reg:squarederror',
                      max_iter=100,
                      max_depth=3,
                      gama=1.0,
                      min_child_weight=1,
                      reg_lambda=1.0,
                      print_log=False
                      )

        clf.fit(X_train, y_train, learning_rate=0.1, print_error_rate=True)

        print(' model complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        y_pred_test = clf.predict(X_test)

        print('by xrh , the squared_error:', mean_squared_error(y_test, y_pred_test))  #


    def loadData_2classification(self, fileName, n=1000):
        """
        加载文件
        将 数据集 的标签 转换为 二分类的标签

        :param fileName:要加载的文件路径
        :param n: 返回的数据集的规模
        :return: 数据集和标签集

        """

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

        测试模型, 并对 模型效果做出评估

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """
        Mnist_dir = '../../dataset/Mnist'

        # 获取训练集
        trainDataList, trainLabelList = self.loadData_2classification(os.path.join(Mnist_dir, 'mnist_train.csv'), n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

        # 开始时间
        print('start training model....')
        start = time.time()

        model_path = 'models/xgboost_two_classify.model'

        clf = XGBoost(
                      model_path=model_path,
                      objective='binary:logistic',
                      max_iter=20,
                      max_depth=3,
                      gama=0.5,
                      reg_lambda=0.5,
                      print_log=True
                      )
        clf.fit(trainDataArr, trainLabelArr, learning_rate=1.0)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData_2classification(os.path.join(Mnist_dir, 'mnist_test.csv'), n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        clf2 = XGBoost(use_pretrained=True, model_path=model_path)

        print('test dataset accuracy: {} '.format(clf2.score(testDataArr, testLabelArr)))

        # 模型评估
        y_prob = clf2.predict_prob(testDataArr)
        y_pred = clf2.predict(testDataArr)
        y_true = testLabelArr

        eval_model = EvaluateModel()

        # eval_model.tow_classify_evaluate(y_true=y_true,y_pred=y_pred,y_prob=y_prob)

    def test_Higgs_dataset(self, n=10000):
        """

        :return:
        """
        Higgs_dataset_path = '../../dataset/higgs/kaggle'

        data = np.loadtxt(os.path.join(Higgs_dataset_path, 'training.csv'), delimiter=',', skiprows=1, max_rows=n,
                          converters={32: lambda x: int(x == 's'.encode('utf-8'))})

        # max_rows 设置读取的行数
        # converters 对最后一列进行转换

        X = data[:, 1:31]
        y = data[:, 32]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # 陈天奇 xgboost 参数
        # param1 = {'objective': 'binary:logistic', "eta": 0.1, "max_depth": 3, "nthread": 16}
        # num_round = 120
        # Accuracy:0.8335

        start = time.time()

        clf = XGBoost(
                      objective='binary:logistic',
                      max_iter=20,
                      max_depth=3,
                      gama=0.5,
                      reg_lambda=0.5,
                      print_log=False
                      )

        clf.fit(X_train, y_train, learning_rate=1.0)

        end = time.time()

        ypred = clf.predict(X_test)

        print("Accuracy:{}".format(accuracy_score(y_test, ypred)))

        print('training cost time :', end - start)


    def loadData(self, fileName, n=1000):
        """
        加载文件

        :param fileName:要加载的文件路径
        :param n: 返回的数据集的规模
        :return: 数据集和标签集
        """
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

        测试  模型的 多分类

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """
        Mnist_dir = '../../dataset/Mnist'

        # 获取训练集
        trainDataList, trainLabelList = self.loadData(os.path.join(Mnist_dir, 'mnist_train.csv'), n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

        # 开始时间
        print('start training model....')
        start = time.time()

        model_path = 'models/xgboost_multi_classify.model'

        # clf = XGBoost(model_path=model_path,
        #               objective="multi:softmax",
        #               num_class=10,
        #               max_iter=20,
        #               max_depth=3,
        #               gama=0,
        #               reg_lambda=0,
        #               print_log=True
        #               )
        # clf.fit(trainDataArr, trainLabelArr, learning_rate=1.0)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData(os.path.join(Mnist_dir, 'mnist_test.csv'), n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        clf2 = XGBoost(use_pretrained=True, model_path=model_path)

        print('test dataset accuracy: {} '.format(clf2.score(testDataArr, testLabelArr)))

    def test_iris_dataset(self):

        # 使用iris数据集，其中有三个分类， y的取值为0 , 1 , 2

        X, y = datasets.load_iris(True)  # 包括150行记录
        # 将数据集一分为二，训练数据占80%，测试数据占20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=188)

        clf = XGBoost(
                      objective="multi:softmax",
                      num_class=3,
                      max_iter=3,
                      max_depth=2,
                      print_log=False)

        clf.fit(X_train, y_train, learning_rate=0.5)

        clf = XGBoost(use_pretrained=True)

        print('test dataset accuracy: {} '.format(clf.score(X_test, y_test)))


if __name__ == '__main__':
    test = Test()

    # test.test_boston_regress_dataset()

    test.test_Mnist_dataset_2classification(6000, 1000)

    # test.test_Mnist_dataset_2classification(60000, 10000)

    # test.test_Higgs_dataset()

    # test.test_tiny_multiclassification_dataset()

    # test.test_Mnist_dataset(6000,1000)

    # test.test_Mnist_dataset(60000, 10000)

    # test.test_iris_dataset()
