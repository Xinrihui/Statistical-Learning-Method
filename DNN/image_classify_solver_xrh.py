#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from lib import optimizer_xrh as optim

from lib.cnn_multi_classify_xrh import *

from lib.classify_dataset_xrh import *

import time

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ImageClassifySolver:
    """

    对图片进行分类的模型的包装器

    示例代码:

    data = load_Mnist_data()
    model = MultiClassifyCNN()
    solver = CaptioningSolver(model, data,
                    update_rule='BGD',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    其中模型对象(model)必须实现以下方法(API):

    - model.params 是一个字典, 包含模型所有需要更新的参数

    - model.fit_batch(batch_picture, batch_label) 计算模型的损失函数和所有参数的梯度, 它的输入输出为:

      Inputs:
      - batch_picture: 一批图片 shape (N, pict_c, pict_h, pict_w)
      - batch_label: 图片的分类标签 shape (N, )

      Returns:
      - loss: 损失函数的值
      - grads: 是一个字典, 包含了模型所有需要更新的参数的梯度, 它里面的元素和 model.params 中的一一对应

    - model.inference_batch(batch_picture) 利用训练好的模型进行推理

    Author: xrh
    Date: 2021-09-09

    """

    def __init__(self, model, dataset,
                 model_path='model/cnn_multi_classify.model',
                 **kwargs):
        """

        必要参数:
        - model: RNN 模型对象
        - dataset: 训练数据集
        - model_path: 预训练模型的路径

        可选参数:
        - optimize_mode: 可选的优化算法
                      'BGD' : 批量梯度下降 BGD
                      'MinBatch': min-Batch梯度下降 (默认)
                      'Momentum': 带动量的 Mini-batch 梯度下降
                      'Adam': Adam Mini-batch 梯度下降

        - optim_config: 字典类型, 优化算法的超参数 (详见 Optimizer_xrh.py )
                      {
                     'learning_rate': 5e-3,
                      }
        - batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 64)
        - num_epochs: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        - print_log: 是否打印日志
        - print_every: 打印训练误差的步长, 训练多少个 iteration 就打印一次训练误差

        """
        self.model = model
        self.dataset = dataset

        self.model_path = model_path

        # 解析可选参数列表
        # kwargs = dict()
        self.optimize_mode = kwargs.pop('optimize_mode', 'MinBatch')  # 将元素从字典中弹出, 若字典中没有此 key, 则赋值为默认值
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size = kwargs.pop('batch_size', 64)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_log = kwargs.pop('print_log', True)
        self.print_every = kwargs.pop('print_every', 10)

        # 如果kwargs中还有其他参数未被弹出, 则报错
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # 配置优化算法
        class_name = self.optimize_mode + 'Optimizer'  # 'BGD' + 'Optimizer' = 'BGDOptimizer'
        if not hasattr(optim, class_name):
            raise ValueError('Invalid optimize_mode "%s"' % self.optimize_mode)

        Optimizer = getattr(optim, class_name)

        self.optimizer = Optimizer(self.optim_config)
        self.optim_param = {}  # 记录优化算法中需要更新的参数

        # 记录每一次迭代(iteration)的损失函数的值
        self.loss_history = []

    def fit_one_iteration(self):
        """
        模型训练的一次迭代 (iteration) , 多次迭代组成一个 epcho

        :return:
        """

        batch_picture, batch_label = sample_dataset_minibatch(batch_size=self.batch_size, dataset=self.dataset)
        # 从 dataset 随机采样 batch_size 个样本

        loss, grads = self.model.fit_batch(batch_picture=batch_picture, batch_label=batch_label)

        self.loss_history.append(loss)  # 记录这次迭代的损失

        # 更新模型的参数
        for param_name, param_value in self.model.params.items():
            grad_param = grads['grad_' + param_name]

            next_param_value = self.optimizer.update_parameter((param_name, param_value), grad_param, self.optim_param)

            self.model.params[param_name] = next_param_value

    def fit(self):
        """
        训练模型

        :return:
        """

        N = np.shape(self.dataset['feature'])[0]  # 训练数据集中含有的样本个数 50

        iterations_per_epoch = max(N // self.batch_size, 1)  # 每一个 epoch 要迭代的次数 50/25=2

        num_iterations = self.num_epochs * iterations_per_epoch  # 总共迭代的次数

        for t in range(num_iterations):

            self.fit_one_iteration()

            # Maybe print training loss
            if self.print_log and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))

        # 存储训练好的模型
        self.model.save(self.model_path)


class Test:

    def loadData(self, fileName, n=1000, binaryzation=True):
        """
        加载文件

        加载文件
        :param fileName:要加载的文件路径
        :param n: 返回的数据集的规模
        :param binaryzation: 对样本特征进行二值化处理(大于128的转换成1，小于的转换成0);
                             若进行了二值化处理, 则后续无需再对特征进行归一化(normalization)

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

            if binaryzation:  # 此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
                dataArr.append([int(int(num) > 128) for num in curLine[1:]])
            else:
                dataArr.append([int(num) for num in curLine[1:]])

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

        binaryzation = True  # 是否对样本特征进行二值化处理

        # 训练模型

        # 1.获取训练集
        trainDataList, trainLabelList = self.loadData('../dataset/Mnist/mnist_train.csv', n=n_train,
                                                      binaryzation=binaryzation)

        print('train data, row num:{} , column num:{} '.format(len(trainDataList), len(trainDataList[0])))

        trainDataArr = np.array(trainDataList)  # shape (N, n_pict=784)
        trainLabelArr = np.array(trainLabelList)  # shape (N,)

        dataset_train = {}

        N_train = np.shape(trainDataArr)[0]  # 样本个数
        dataset_train['feature'] = trainDataArr.reshape((N_train, 1, 28, 28))
        dataset_train['label'] = trainLabelArr


        # 开始时间
        print('start training model....')
        start = time.time()

        # 2. 训练模型
        cnn_model = MultiClassifyCNN(
            picture_dim=(1, 28, 28),
            class_num=10,
            use_pre_train=False
        )

        solver = ImageClassifySolver(cnn_model, dataset_train,
                                     model_path='models/cnn_multi_classify.model',
                                     optimize_mode='Adam',
                                     num_epochs=5,
                                     batch_size=512,
                                     optim_config={
                                         'learning_rate': 5e-3,
                                         'bias_correct': True
                                     },
                                     print_log=True,
                                     print_every=10,
                                     )


        solver.fit()

        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # Plot the training losses
        # plt.plot(solver.loss_history)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Training loss history')
        # plt.show()

        # 测试模型

        cnn_model_pre_train = MultiClassifyCNN(
            use_pre_train=True,
            model_path='models/cnn_multi_classify.model'
        )

        # 1.获取测试集
        testDataList, testLabelList = self.loadData('../dataset/Mnist/mnist_test.csv', n=n_test,
                                                    binaryzation=binaryzation)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        dataset_test = {}
        N_test = np.shape(testDataArr)[0]  # 样本个数
        dataset_test['feature'] = testDataArr.reshape((N_test, 1, 28, 28))
        dataset_test['label'] = testLabelArr

        # 从 dataset 随机采样 batch_size 个样本, 将整个测试数据集输入模型进行推理
        batch_picture_test, batch_label_test = sample_dataset_minibatch(batch_size=N_test, dataset=dataset_test)


        # 2. 测试数据输入模型进行推理
        y_predict = cnn_model_pre_train.predict(batch_picture_test)

        print('test accuracy :', accuracy_score(y_predict, batch_label_test))


        # 对比训练集和测试集的 accuracy, 判断模型是否出现过拟合
        batch_picture_train, batch_label_train = sample_dataset_minibatch(batch_size=N_test, dataset=dataset_train)

        y_predict_train = cnn_model_pre_train.predict(batch_picture_train)

        print('train accuracy :', accuracy_score(y_predict_train, batch_label_train))

if __name__ == '__main__':

    test = Test()

    test.test_Mnist_dataset(60000, 10000)
