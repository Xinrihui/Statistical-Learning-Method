import os
import shutil
import logging
import argparse
import pandas as pd
from GBDT_ref.gbdt import GradientBoostingRegressor
from GBDT_ref.gbdt import GradientBoostingBinaryClassifier
from GBDT_ref.gbdt import GradientBoostingMultiClassifier

import numpy as np

import sklearn.metrics as metrics

import time


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
            # 标签0设置为1，反之为-1

            # 显然这会导致 正负 样本的 分布不均衡, 1 的样本很少, 而-1 的很多
            if int(curLine[0]) == 0:
                labelArr.append(1)
            else:
                labelArr.append(0)

            # if int(curLine[0]) <= 5:
            #     labelArr.append(1)
            # else:
            #     labelArr.append(-1)

            cnt += 1

        fr.close()

        # 返回数据集和标记
        return dataArr, labelArr

    def test_Mnist_dataset_2classification(self, n_train, n_test):
        """
        将 Mnist (手写数字) 数据集 转变为 二分类 数据集

        测试 GBDT

        :param n_train: 使用训练数据集的规模
        :param n_test: 使用测试数据集的规模
        :return:
        """

        # 获取训练集
        trainDataList, trainLabelList = self.loadData_2classification('../../../Mnist/mnist_train.csv', n=n_train)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array ( trainDataList )
        trainLabelArr = np.array ( trainLabelList )

        data= np.concatenate( (trainDataArr,trainLabelArr.reshape(-1,1)) , axis=1 )

        data= pd.DataFrame( data=data,
                            columns= list(range(data.shape[1]-1))+['label']
                           )

        # 开始时间
        print('start training model....')
        start = time.time()

        model = GradientBoostingBinaryClassifier( learning_rate=0.1, n_trees=10, max_depth=3,is_log=False, is_plot=False)

        model.fit(data)


        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData_2classification('../../../Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        test_data= np.concatenate( (testDataArr,testLabelArr.reshape(-1,1)) , axis=1 )

        test_data= pd.DataFrame( data=test_data,
                            columns= list(range(test_data.shape[1]-1))+['label']
                           )

        model.predict(test_data)

        y_predict = test_data['predict_label']


        print('test dataset accuracy: {} '.format(metrics.accuracy_score(testLabelArr,y_predict)))






if __name__ == '__main__':
    test = Test()

    # test.test_regress_dataset()

    test.test_Mnist_dataset_2classification(6000, 1000)





