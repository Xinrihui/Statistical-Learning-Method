
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



class DecisonTree_Lib:
    """
    决策树 相关的 函数库

    不使用 pandas 的 库函数( eg. groupby() ) , 而使用 numpy, 提升 训练速度
    
    Author: xrh
    Date: 2021-03-14
    
    ref: https://github.com/Dod-o/Statistical-Learning-Method_Code.git

    """

    def calcH_D(self, trainLabelArr):
        """
        计算数据集D的经验熵，参考公式5.7 经验熵的计算
        
        
        :param trainLabelArr:当前数据集的标签集
        :return: 经验熵

        """

        D = len(trainLabelArr)  # 数据集的总行数

        LabelSet = {ele for ele in trainLabelArr}  # trainLabelArr 中所有标签的类别

        H_D = 0

        for label in LabelSet:
            C_k = len(trainLabelArr[trainLabelArr == label])
            p = C_k / D
            H_D += -p * log2(p)

        return H_D

    def calcH_D_A(self, trainDataArr_DevFeature, trainLabelArr):
        """
        计算 经验条件熵
        
        只对 所关心的 特征的那一列 进行计算，提升训练速度
        
        :param trainDataArr_DevFeature: 切割后只有feature那列数据的数组
        :param trainLabelArr: 标签集数组
        :return: 经验条件熵

        """

        A_set = {A_i for A_i in trainDataArr_DevFeature}  # trainDataArr_DevFeature 中的 所有取值

        D = len(trainLabelArr)  # 数据集的总行数

        H_D_A = 0

        for i in A_set:
            D_i = len(trainDataArr_DevFeature[trainDataArr_DevFeature == i])  # 特征值为 i 的 样本的总个数

            p_i = D_i / D

            H_D_A += p_i * self.calcH_D(trainLabelArr[trainDataArr_DevFeature == i])

        return H_D_A

    def info_gain(self, H_D, H_D_A):
        """
        信息增益
        :param H_D:经验熵
        :param H_D_A:经验条件熵
        :return:
        """

        return H_D - H_D_A

    def select_max_info_gain(self, trainDataArr, trainLabelArr, feature_set=None):

        """
        选择 信息增益 最大的特征

        :param trainDataArr: shape=(60000,784)
        :param trainLabelArr: shape=(60000,1)
        :param feature_set:  可供选择 的特征集合
        :return:
        """

        if feature_set == None:
            feature_Num = len(trainDataArr[0])  # 特征的总数
            feature_set = set(range(feature_Num))

        H_D = self.calcH_D(trainLabelArr)  # 整个数据集的 熵

        max_info_gain = float('-inf')  # 最大信息增益
        max_info_gain_feature = 0  # 取得最大信息增益的特征

        for i in feature_set:

            H_D_A = self.calcH_D_A(trainDataArr[:, i], trainLabelArr)  # 选择第i个特征作为划分特征时的条件熵

            current = self.info_gain(H_D, H_D_A)

            #         print('g(D,A{})={}'.format(i,current))

            if current > max_info_gain:
                max_info_gain = current
                max_info_gain_feature = i

        return max_info_gain_feature, max_info_gain



# 树节点
class Node:
    def __init__(self, label=None, curr_dataset=None, feature=None, feature_name=None, prev_feature=None,
                 prev_feature_value=None, childs=None):
        self.label = label  # 叶子节点才有标签

        self.curr_dataset = curr_dataset

        self.feature = feature  # 非叶子节点, 划分 子节点的特征
        self.feature_name = feature_name

        self.prev_feature = prev_feature
        self.prev_feature_value = prev_feature_value

        self.childs = childs


# ID3 决策树
class DecisonTree_ID3(DecisonTree_Lib):
    """
    决策树的 ID3 算法

    未实现剪枝
    
    1.使用 numpy 而不是 pandas 
    2. 计算 经验条件熵 时, 只计算 关注的特征列 和 最后的标签值列 , 减少了计算的数据规模
    
    优化训练速度
        
    test1: 多分类任务
    数据集：Mnist
    训练集数量：60000
    测试集数量：10000
    正确率：0.8589
    运行时长：173s  
    
    Author: xrh
    Date: 2021-03-14

    """

    def __init__(self, root=None, threshold=0.0):

        self.root = root
        self.threshold = threshold  # 信息增益的 阈值

    def __pure_dataset(self, trainLabelArr):
        """
        判断 数据集 D 是否纯净
        """
        dict_labels = Counter(trainLabelArr.flatten())

        return len(dict_labels) == 1

    def major_class(self, trainLabelArr):
        """
        拿到 数据集 D 中数量最多的 标签

        """
        dict_labels = Counter(trainLabelArr.flatten())

        max_num = float('-inf')
        max_num_label = None

        for k, v in dict_labels.items():

            if v > max_num:
                max_num = v
                max_num_label = k

        return max_num_label

    def __build_tree(self, trainDataArr, trainLabelArr, feature_set, prev_feature=None, prev_feature_value=None):

        T = Node()

        T.prev_feature = prev_feature
        T.prev_feature_value = prev_feature_value

        if self.__pure_dataset(trainLabelArr) == True:  # 数据集 已经纯净, 无需往下划分, 形成叶子节点

            T.label = trainLabelArr[0]

        elif len(feature_set) == 0:  # 所有特征已经用完, 形成叶子节点

            # 选取 数据集 中最多的样本标签值作为  叶子节点的标签
            T.label = self.major_class(trainLabelArr)

        else:

            Ag, max_info_gain = self.select_max_info_gain(trainDataArr, trainLabelArr, feature_set)
            T.feature = Ag

            if max_info_gain < self.threshold:  # 信息增益 小于 阈值
                T.label = self.major_class(trainLabelArr)

            else:

                T.childs = dict()

                trainDataArr_DevFeature = trainDataArr[:, Ag]
                A_set = {A_i for A_i in trainDataArr_DevFeature}  # trainDataArr_DevFeature 中的 所有取值

                for A_i in A_set:
                    T.childs[A_i] = self.__build_tree(trainDataArr[trainDataArr_DevFeature == A_i],
                                                      trainLabelArr[trainDataArr_DevFeature == A_i], feature_set - {Ag},
                                                      prev_feature=T.feature,
                                                      prev_feature_value=A_i)

        print('T.feature:{}'.format(T.feature))
        print('T.prev_feature:{},T.prev_feature_value:{} '.format(T.prev_feature, T.prev_feature_value))

        print('T.childs:{}'.format(T.childs))
        print('T.label:{}'.format(T.label))

        print('-----------')

        return T

    def fit(self, trainDataArr, trainLabelArr):

        feature_set = set(range(len(trainDataArr[0])))  #  特征的总数: len(trainDataArr[0])

        self.root = self.__build_tree(trainDataArr, trainLabelArr, feature_set)

    def __predict(self, row):
        """
        预测 一个样本

        :param row:
        :return:
        """

        p = self.root

        while p.label == None:  # 到达 叶子节点 退出循环

            judge_feature = p.feature  # 当前节点划分的 特征
            # judge_feature_name= p.feature_name

            p = p.childs[row[judge_feature]]

        return p.label


    def predict(self, testDataArr, testLabelArr):
        """
        预测 测试 数据集，返回预测结果 和 正确率

        :param test_data:
        :return:
        """

        res_list = []

        for row in testDataArr:
            res_list.append( self.__predict(row) )

        # accuracy = np.mean( np.equal( res_list, testLabelArr ) )  # 快速计算 正确率

        err_arr = np.ones( len(res_list), dtype=int)
        res_arr=np.array(res_list)
        err_arr[res_arr == testLabelArr] = 0
        err_rate = np.mean(err_arr)

        accuracy=1-err_rate

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

    def test_DecisonTree_Lib(self):
        """
        DecisonTree_MathLib  测试

        :return:
        """
        # DecisonTree_Lib  测试

        datasets, label_name = self.__create_tarin_data()

        datasetsArr = np.array(datasets)

        MathLib = DecisonTree_Lib()
        print('H(D)= ', MathLib.calcH_D(datasetsArr[:, -1]))

        print('H(D|A)= ', MathLib.calcH_D_A(datasetsArr[:, 0], datasetsArr[:, -1]))  # 选择 年龄 作为 划分的特征

        print('g(D,A)=', MathLib.info_gain(
            MathLib.calcH_D(datasetsArr[:, -1]), MathLib.calcH_D_A(datasetsArr[:, 0], datasetsArr[:, -1])))

        max_info_gain_feature, max_info_gain = MathLib.select_max_info_gain(
            datasetsArr[:, 0:-1], datasetsArr[:, -1])

        print('best feature:{}, max_info_gain:{}'.format(
            max_info_gain_feature, max_info_gain))


    def test_small_dataset(self):
        """
        
        利用《统计学习方法》 表 5.1 中的数据集 测试 决策树ID3

        :return:
        """

        # 获取训练集
        datasets, labels = self.__create_tarin_data()

        datasetsArr=np.array(datasets)
        # 开始时间
        start = time.time()

        # 创建决策树
        print('start create tree')

        ID3 = DecisonTree_ID3(threshold=0.1)
        ID3.fit(datasetsArr[:, 0:-1], datasetsArr[:, -1])

        print(' tree complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        # 测试数据集
        datasets_test, labels = self.__create_test_data()

        datasetsArr_test = np.array( datasets_test )


        print('res:', ID3.predict(datasetsArr_test[:, 0:-1], datasetsArr_test[:, -1]))


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

        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)


        # 开始时间
        print('start training model....')
        start = time.time()

        ID3 = DecisonTree_ID3(threshold=0.1)
        ID3.fit(trainDataArr, trainLabelArr)

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../Mnist/mnist_test.csv',n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)


        print('res:', ID3.predict(testDataArr,testLabelArr))



if __name__ == '__main__':

    test=Test()

    # test.test_DecisonTree_Lib()

    # test.test_small_dataset()

    test.test_Mnist_dataset(6000,1000)



