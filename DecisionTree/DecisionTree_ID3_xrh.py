
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import *
import pprint

import time


class DecisonTree_MathLib:
    """
    决策树 相关的 数学函数库

    Author: xrh
    Date: 2021-03-12

    """

    def calc_ent(self,datasets):
        """
        计算数据集的 熵
        
        """

        label = datasets.columns[-1]  # 最后一列 为标签

        # pandas 分组 https://www.yiibai.com/pandas/python_pandas_groupby.html
        grouped = datasets.groupby(label)

        D = datasets.shape[0]  # 数据集的总行数

        H_D = 0

        for name, group in grouped:

            C_k = group.shape[0]
            p = C_k / D
            H_D += -p*log2(p)

        return H_D

    def calc_cond_ent(self,datasets, axis=0):
        """
        计算数据集的 条件熵

        axis 为选择的 特征

        """
        label = datasets.columns[-1]  # 最后一列 为标签 Y

        H_D_A = 0  # 条件熵

        A = datasets.columns[axis]

        D = datasets.shape[0]  # 数据集的总行数

        grouped = datasets.groupby(A)

        for name, group in grouped:

            D_i = group.shape[0]

            p_i = D_i/D

            sub_grouped = group.groupby(label)

            for sub_name, sub_group in sub_grouped:

                D_ik = sub_group.shape[0]

                p_ik = D_ik/D_i

                H_D_A += -p_i*p_ik*log2(p_ik)

        return H_D_A

    def info_gain(self,ent, cond_ent):
        """
        信息增益
        :param ent:
        :param cond_ent:
        :return:
        """

        return ent-cond_ent

    def info_gain_train(self,datasets, feature_set=None):
        """

        选择 信息增益 最大的特征

        :param datasets:
        :param feature_set:  可供选择 的特征集合
        :return:
        """

        if feature_set == None:
            feature_Num = len(datasets.columns) - 1  # 特征的总数（最后一列为标签, 不是特征）

            feature_set = set(range(feature_Num))

        ent = self.calc_ent(datasets)  # 整个数据集的 熵

        max_info_gain = float('-inf')  # 最大信息增益
        max_info_gain_feature = 0  # 取得最大信息增益的特征

        for i in feature_set:

            cond_ent = self.calc_cond_ent(datasets, i)  # 选择第i个特征作为划分特征时的条件熵

            current = self.info_gain(ent, cond_ent)

            #         print('g(D,A{})={}'.format(i,current))

            if current > max_info_gain:
                max_info_gain = current
                max_info_gain_feature = i

        return max_info_gain_feature, max_info_gain


# 树节点
class Node:
    def __init__(self, label=None, curr_dataset=None, feature=None, feature_name=None, prev_feature_name=None,prev_feature_value=None, childs=None):
        self.label = label  # 叶子节点才有标签

        self.curr_dataset = curr_dataset

        self.feature = feature  # 非叶子节点, 划分 子节点的特征
        self.feature_name = feature_name

        self.prev_feature_name=prev_feature_name
        self.prev_feature_value = prev_feature_value

        self.childs = childs


# ID3 决策树
class DecisonTree_ID3(DecisonTree_MathLib):
    """
    决策树的 ID3 算法

    未实现剪枝

    运行速度较慢

    test1:
    数据集：Mnist
    训练集数量：6000
    测试集数量：1000
    正确率：72.6%
    运行时长：>10min

    test2:
    数据集：Mnist
    训练集数量：60000
    测试集数量：10000
    正确率：--
    运行时长：-- 

    Author: xrh
    Date: 2021-03-12

    """

    def __init__(self, root=None, threshold=0.0):

        self.root = root
        self.threshold = threshold  # 信息增益的 阈值

        self.MathLib=DecisonTree_MathLib

    def __pure_dataset(self, datasets):
        """
        判断 数据集 D 是否纯净
        """

        label = datasets.columns[-1]  # 最后一列 为标签 Y

        grouped = datasets.groupby(label)

        return len(grouped) == 1

    def __major_class(self, datasets):
        """
        拿到 数据集 D 中数量最多的 标签
        """

        label = datasets.columns[-1]  # 最后一列 为标签 Y

        grouped = datasets.groupby(label)

        max_nums = float('-inf')
        max_nums_group_name = None

        for name, group in grouped:

            if group.shape[0] > max_nums:
                max_nums = group.shape[0]
                max_nums_group_name = name

        return max_nums_group_name

    def __build_tree(self, datasets, feature_set, prev_feature_name=None,prev_feature_value=None):

        T = Node(curr_dataset=datasets)

        T.prev_feature_name=prev_feature_name
        T.prev_feature_value=prev_feature_value

        if self.__pure_dataset(datasets) == True:  # 数据集 已经纯净, 无需往下划分, 形成叶子节点

            T.label = datasets.iloc[0, -1]

        elif len(feature_set) == 0:  # 所有特征已经用完, 形成叶子节点

            # 选取 数据集 中最多的样本标签值作为  叶子节点的标签
            T.label = self.__major_class(datasets)

        else:

            Ag, max_info_gain = self.info_gain_train(datasets,feature_set)
            T.feature = Ag
            T.feature_name = datasets.columns[Ag]

            if max_info_gain < self.threshold:  # 信息增益 小于 阈值
                T.label = self.__major_class(datasets)

            else:

                grouped = datasets.groupby(datasets.columns[Ag])

                T.childs = dict()

                for value, group in grouped:
                    T.childs[value] = self.__build_tree(group, feature_set - { Ag },prev_feature_name=T.feature_name,
                                                        prev_feature_value=value)

        print('T.feature_name:{}'.format(T.feature_name))
        print('T.prev_feature_name:{},T.prev_feature_value:{} '.format(T.prev_feature_name,T.prev_feature_value))

        print('T.childs:{}'.format(T.childs))
        print('T.label:{}'.format(T.label))
        # print('T.curr_dataset:{}'.format(T.curr_dataset))

        print('-----------')

        return T

    def fit(self, train_data):

        feature_set = set(range(len(train_data.columns) - 1))  # 特征的总数（最后一列为标签, 不是特征）

        self.root = self.__build_tree(train_data, feature_set)


    def __predict(self, row):
        """
        预测 一个样本

        :param row:
        :return:
        """

        p = self.root

        while p.label==None: # 到达 叶子节点 退出循环

            judge_feature = p.feature # 当前节点划分的 特征
            # judge_feature_name= p.feature_name

            p= p.childs[ row[judge_feature] ]

        return p.label

    def predict(self, test_data):
        """
        预测 测试 数据集，返回预测结果 和 正确率

        :param test_data:
        :return:
        """

        res_list=[]

        for idx,row in test_data.iterrows():

            res_list.append(self.__predict(row))

        label_list= test_data.iloc[:,-1]

        accuracy=np.mean(np.equal(res_list,label_list)) # 快速计算 正确率

        return res_list, accuracy

class Test:

    def __create_tarin_data(self):
        """
        《统计学习方法》 表5.1 中的数据集
        :return:
        """
        datasets = [['青年', '否', '否', '一般', '否'],
                    ['青年', '否', '否', '好', '否'],
                    ['青年', '是', '否', '好', '是'],
                    ['青年', '是', '是', '一般', '是'],
                    ['青年', '否', '否', '一般', '否'],
                    ['中年', '否', '否', '一般', '否'],
                    ['中年', '否', '否', '好', '否'],
                    ['中年', '是', '是', '好', '是'],
                    ['中年', '否', '是', '非常好', '是'],
                    ['中年', '否', '是', '非常好', '是'],
                    ['老年', '否', '是', '非常好', '是'],
                    ['老年', '否', '是', '好', '是'],
                    ['老年', '是', '否', '好', '是'],
                    ['老年', '是', '否', '非常好', '是'],
                    ['老年', '否', '否', '一般', '否'],
                    ]
        labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
        # 返回数据集和每个维度的名称
        return datasets, labels

    def __create_test_data(self):

        datasets = [['青年', '否', '是', '一般', '是'],
                    ['老年', '否', '否', '好', '否']
                    ]
        labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']

        # 返回数据集和每个维度的名称
        return datasets, labels

    def test_DecisonTree_MathLib(self):
        """
        DecisonTree_MathLib  测试

        :return:
        """
        # 获取训练集
        datasets, labels = self.__create_tarin_data()
        train_data = pd.DataFrame(datasets, columns=labels)

        MathLib=DecisonTree_MathLib()
        print('H(D)= ', MathLib.calc_ent(train_data))
        print('H(D|A)= ', MathLib.calc_cond_ent(
            train_data, 0))  # 选择 年龄 作为 划分的特征

        print('g(D,A)=', MathLib.info_gain(
            MathLib.calc_ent(train_data), MathLib.calc_cond_ent(train_data, 0)))

        max_info_gain_feature, max_info_gain = MathLib.info_gain_train(
            train_data)

        print('best feature:{}, max_info_gain:{}'.format(
            train_data.columns[max_info_gain_feature], max_info_gain))


    def test_small_dataset(self):
        """
        利用《统计学习方法》 表5.1 中的数据集 测试 决策树ID3

        :return:
        """

        # 获取训练集
        datasets, labels = self.__create_tarin_data()

        #train_data = pd.DataFrame(datasets, columns=labels)
        train_data = pd.DataFrame(datasets)

        # 开始时间
        start = time.time()

        # 创建决策树
        print('start create tree')

        ID3 = DecisonTree_ID3(threshold=0.1)
        ID3.fit(train_data)

        print(' tree complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        # 测试数据集
        datasets, labels = self.__create_test_data()

        # test_data = pd.DataFrame(datasets, columns=labels)

        test_data = pd.DataFrame(datasets)

        print('res:', ID3.predict(test_data))

    def loadData(self,fileName,n=1000):
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

        cnt=0 # 计数器

        # 遍历文件中的每一行
        for line in fr.readlines():

            if cnt==n:
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

            cnt+=1

        fr.close()

        # 返回数据集和标记
        return dataArr, labelArr


    def test_Mnist_dataset(self ,n_train,n_test):
        """
        利用 Mnist 数据集 测试 决策树ID3


        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData('../Mnist/mnist_train.csv',n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList),len(trainDataList[0])))

        trainData = pd.DataFrame(trainDataList)
        trainLabel = pd.DataFrame(trainLabelList, columns=[len(trainDataList[0]) + 1])  # 标签列要加上列名,
                                                                                        # 否则 columns 默认为0 与 trainData 拼接时, trainData 本身有 columns=0

        trainData = pd.concat([trainData, trainLabel], axis=1)

        # 开始时间
        print('start training model....')
        start = time.time()

        ID3 = DecisonTree_ID3(threshold=0.1)
        ID3.fit(trainData)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../Mnist/mnist_test.csv',n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        testData = pd.DataFrame(testDataList)
        testLabel = pd.DataFrame(testLabelList, columns=[len(trainDataList[0]) + 1])

        testData = pd.concat([testData, testLabel], axis=1)

        print('res:', ID3.predict(testData))





if __name__ == '__main__':

    test=Test()
    # test.test_small_dataset()

    test.test_Mnist_dataset(60000,10000)



