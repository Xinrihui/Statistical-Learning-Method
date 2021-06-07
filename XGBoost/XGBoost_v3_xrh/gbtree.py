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

from updater_colmaker import  *

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

from sklearn.metrics import auc



class DMatrix:

    def __init__(self,DataArr,missing=np.nan):
        """

        :param DataArr: 样本数据的特征列, 不包含标签
        :param missing: 代表缺失数据的值
        """

        # N: 样本个数 ; m 特征的总数
        self.N,self.m = np.shape(DataArr)

        # RowIndex 样本的行索引
        self.RowIndex=list(range(self.N))

        # 样本 行
        self.RowData=DataArr

        # 出现过特征缺失值的行
        self.MissingValueRow=set()

        # 所有特征对应的块集合
        self.SortedPages=[]

        for k in range(self.m):  # 遍历所有的特征

            DataFeatureK = DataArr[:, k]  # 特征 k 拎出来 shape:(N,)
            FeatureK_Index=[]

            for ridx in range(self.N):
                if DataFeatureK[ridx]!=missing:
                    FeatureK_Index.append( (DataFeatureK[ridx],ridx) ) # (特征值, 样本号)
                else:
                    self.MissingValueRow.add(ridx)

            # 按照特征值的大小排序
            SortedFeatureK_Index=sorted(FeatureK_Index ,key=lambda t: t[0])

            self.SortedPages.append(SortedFeatureK_Index)



class XGBoost:
    """

    python 重写 XGBoost , 功能包括:

    1. 回归 ( 使用平方损失函数 )
    2. 二分类 ( 使用 交叉熵损失函数 )
    3. 多分类 ( 使用 对数损失函数 )

    ref:
    https://github.com/dmlc/xgboost/

    实现细节：
    (1) 使用 完全贪心算法 和 近似算法 划分子节点;

        完全贪心算法(分块有序):


        近似算法(分块有序+局部策略)：


    (2)  获取分位点时, 需要 将所有值读入内存并排序 ;
        当内存不够时, 无法做到 对所有特征值排序, 此时可采用 分位点估计算法即 加权分位数速写-Weighted Quantile Sketch, 来估算出分位点的值 (未实现)

    (3)  稀疏感知划分节点（Sparsity-aware Split Finding）
         

    Author: xrh
    Date: 2021-05-29

    test1: 回归任务
    数据集：boston房价数据集
    参数: error_rate_threshold=0.01, max_iter=100, max_depth=3,learning_rate=0.1,gama=1.0, reg_lambda=1.0
    训练集数量：455
    测试集数量：51
    测试集的 MSE： 9.51
    模型训练时长：3.2s


    test2: 二分类任务
    数据集：Mnist
    参数:
      error_rate_threshold=0.01
      max_iter=40,
      max_depth=3,
      gama=1.0,
      reg_lambda=1.0,
      tree_method = 'approx',
      sketch_eps = 0.3
    训练集数量：6000
    测试集数量：1000
    正确率：0.981
    模型训练时长： 589s

    test3: 二分类任务
    数据集：Higgs
    参数:
      error_rate_threshold=0.01,
      objective='binary:logistic',
      max_iter=30,
      max_depth=3,
      gama=1.0,
      reg_lambda=1.0,
      tree_method='approx',
      sketch_eps=0.3,
    训练集数量：8000
    测试集数量：2000
    正确率：0.828
    模型训练时长： 101s

    test3: 二分类任务
    数据集：Higgs
    参数:
      error_rate_threshold=0.01,
      objective='binary:logistic',
      max_iter=30,
      max_depth=3,
      gama=1.0,
      reg_lambda=1.0,
    训练集数量：8000
    测试集数量：2000
    正确率：0.829
    模型训练时长： 124s


    """

    def __init__(self,  print_log=False,
                        error_rate_threshold=0.05 ,
                        max_iter=10,
                        max_depth=3,
                        objective='binary:logistic',
                        num_class=None,
                        base_score=0.0,
                        gama=0.1,
                        reg_lambda=1.0,
                        min_child_weight=1,
                        tree_method= 'exact',
                        sketch_eps = 0.3
                        ):
        """
        :param print_log: 打印 Cart树
        :param error_rate_threshold: 训练中止条件, 若当前得到的基分类器的组合 的在训练集上的错误率 小于阈值, 则停止训练;
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
        :param min_child_weight: 搜索最佳切分点时, 若 min_child_weight < Min(HL, HR) 则放弃此切分点

        :param tree_method： 指定了构建树的算法，可以为下列的值：
                            (1)'exact'： 使用 exact greedy 完全贪心算法分裂节点
                            (2)'approx'： 使用近似算法分裂节点

        :param sketch_eps： 设定分桶的步长为: 二阶梯度的区间和 * sketch_eps ;
                            取值范围为 (0,1), 默认值为 0.3, 此时 每一个特征划分 3个桶 ;
                            它仅仅用于 tree_medhodd='approx'

        """
        # self.N=0 # 训练集的样本个数

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

        # 搜索最佳切分点时, 若 min_child_weight < Min(HL, HR) 则放弃此切分点
        self.min_child_weight=min_child_weight

        # 构建树时 切分子节点的算法
        self.tree_method=tree_method

        # 分桶的步长
        self.sketch_eps=sketch_eps

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

    def init_y_predict(self , F_0 , N):
        """
        m=0 :
        初始化 y_predict
        :param F_0:
        :return:
        """
        y_predict=None

        if self.objective ==  "reg:squarederror": # 回归
            y_predict = np.array( [ F_0 ] * N )

        elif self.objective == "binary:logistic": # 二分类
            y_predict = np.array( [ self.sigmoid(F_0) ] * N)

        elif self.objective == "multi:softmax":

            y_predict = np.transpose([self.softmax(F_0)] * N) #  shape : (K, N)


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

            g = y_predict - y
            h=  np.ones_like(g)

        elif self.objective == "binary:logistic": # 二分类

            g = y_predict - y
            h = y_predict*(1-y_predict)

        elif self.objective == "multi:softmax": # 二分类

            g = (y_predict - y)

            # 以下实现 参考 xgboost 源码
            # ref:
            # https://github.com/dmlc/xgboost/blob/master/src/objective/multiclass_obj.cu
            # class SoftmaxMultiClassObj
            # -> void GetGradient(

            h =  2*y_predict*(1-y_predict)
            h[h<1e-16]= 1e-16 # h 不能小于0

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

        elif self.objective == "multi:softmax":

            y_predict = self.softmax(F)

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

        elif self.objective == "multi:softmax": # 多分类

            y_label = np.argmax( y_predict, axis=0 )  # 取 概率最大的 作为 预测的标签

            err_arr = np.ones( N, dtype=int )
            err_arr[y_label == y] = 0
            error_rate = np.mean(err_arr)  # 计算训练误差

        return error_rate

    def fit(self, X, y ,learning_rate,train_error=True):
        """

        用 训练数据 拟合模型

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :param y: 标签数据 , shape=(N_sample,)
        :param learning_rate: 学习率
        :param train_error:  在训练模型时, 计算 并输出 模型在 训练集上的误差率
        :return:
        """

        N = np.shape(X)[0]  # 样本的个数

        X_DMatrix=DMatrix(X)

        if self.objective == "multi:softmax": # 多分类

            F_0 =  np.array([self.base_score]*self.num_class)  # shape : (K,)

            self.G.append(F_0)

            F = np.transpose([F_0] * N)  # 对 F_0 进行复制,  shape : (K, N)

            y_predict = self.init_y_predict(F_0,N) #  shape : (K, N)

            y_one_hot = (y == np.array(range(self.num_class)).reshape(-1, 1)).astype(
                np.int8)  # 将 向量y 扩展为 one-hot , shape: (K,N)

            for m in range(1, self.max_iter):  # 进行 第 m 轮迭代

                DT_list = []

                for k in range(self.num_class):  # 依次训练 K 个 二分类器

                    print('======= train No.{} 2Classifier ======='.format(k))

                    g,h= self.cal_g_h(y_one_hot[k],y_predict[k]) # y_predict[k]  shape:(N,)

                    # 训练 用于 2分类的 回归树
                    RT = Builder(gama=self.gama,
                                                  reg_lambda=self.reg_lambda,
                                                  max_depth=self.max_depth,
                                                  min_child_weight=self.min_child_weight,
                                                  tree_method= self.tree_method,
                                                  sketch_eps=self.sketch_eps,
                                                  print_log=self.print_log)

                    RT.Update(X_DMatrix, g, h)

                    f_m =  RT.predict(X)

                    DT_list.append(RT)

                    F[k] = F[k] + learning_rate * f_m  # F[k]  shape:(N,)


                y_predict = self.update_y_predict(F)

                self.G.append((learning_rate, DT_list))  # 存储 基分类器

                # 计算 当前 所有弱分类器加权 得到的 最终分类器 的 分类错误率
                if train_error:

                    y_predict_copy = y_predict.copy() # y_predict 下一轮迭代 还要使用, 不能被修改

                    error_rate= self.model_error_rate( y , y_predict_copy )

                    print( 'round:{}, error_rate :{}'.format( m, error_rate ) )
                    print( '======================' )

                    if error_rate < self.error_rate_threshold:  # 错误率 已经小于 阈值, 则停止训练
                        break

        elif self.objective in ('binary:logistic' , 'reg:squarederror') : # 二分类 or 回归

            F_0 = self.base_score

            self.G.append(F_0)

            F = np.array( [self.base_score] * N ) # shape: (N, )

            y_predict= self.init_y_predict(F_0,N)


            for m in range(self.max_iter):  # 进行 第 m 轮迭代

                g,h = self.cal_g_h( y,y_predict )

                RT = Builder(gama=self.gama,
                                              reg_lambda=self.reg_lambda,
                                              max_depth=self.max_depth,
                                              min_child_weight=self.min_child_weight,
                                              tree_method=self.tree_method,
                                              sketch_eps=self.sketch_eps,
                                              print_log=self.print_log)

                RT.Update( X_DMatrix, g, h )

                f_m =  RT.predict(X) # 第 m 颗树

                self.G.append( (learning_rate, RT) )  # 存储 基分类器

                F = F + learning_rate * f_m

                y_predict = self.update_y_predict(F)

                if train_error:

                    y_predict_copy = y_predict.copy() # y_predict 下一轮迭代 还要使用, 不能被修改

                    error_rate= self.model_error_rate( y,y_predict_copy)

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
        N = np.shape(X)[0]

        y_predict = None  # 输出的标签

        F_0 = self.G[0]  # 最终分类器

        if self.objective == "multi:softmax":

            F = np.transpose([F_0] * N)  # shape : (K, N)

            for alpha, DT_list in self.G[1:]:

                for k in range(self.num_class):
                    DT = DT_list[k]

                    # f_m = (self.num_class / (self.num_class - 1)) * DT.predict(X) # shape:(N,)

                    f_m =   DT.predict(X)

                    F[k] += alpha * f_m  # F[k]  shape:(N,)

            prob = self.softmax(F) # F shape:(K,N) ;  prob shape:(K,N)

            y_predict = np.argmax(prob, axis=0) # y_predict shape:(N,)


        elif self.objective in ("reg:squarederror" , "binary:logistic"):

            F  = F_0  # 第一个 存储的是 初始化情况

            for alpha, RT in self.G[1:]:
                f_m = RT.predict(X)
                F += alpha * f_m

            if self.objective == "reg:squarederror":  # 回归

                y_predict=F

            elif self.objective == "binary:logistic":

                y_predict = self.sigmoid(F)

                y_predict[y_predict >= 0.5] = 1  # 概率 大于 0.5 被标记为 正例
                y_predict[y_predict < 0.5] = 0  # 概率 小于 0.5 被标记为 负例

        return y_predict


    def score(self, X, y):
        """
        使用 测试数据集 对模型进行评价, 返回 正确率, 仅适用于分类任务

        :param X: 特征数据 , shape=(N_sample, N_feature)
        :param y: 标签数据 , shape=(N_sample,)
        :return:  错误率 error

        """
        N= X.shape[0]

        y_predict = self.predict(X)

        err_arr = np.ones(N, dtype=int)
        err_arr[y_predict == y] = 0
        error_rate = np.mean(err_arr)

        accuracy= 1- error_rate

        return accuracy


class Test:

    def test_tiny_regress_dataset(self):
        """

        利用 https://blog.csdn.net/zpalyq110/article/details/79527653 中的数据集
        测试  xgboost 回归

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



        # 创建决策树
        print('start create model')

        clf = XGBoost( error_rate_threshold=0.01,objective='reg:squarederror', max_iter=5, max_depth=3,gama=0.0,reg_lambda=0.0 )
        clf.fit( X, y,learning_rate=0.1 )

        print(' model complete ')


        # 测试数据集
        X_test = np.array([
            [25, 65]
        ])
        y_predict = clf.predict( X_test )

        print('res: ', y_predict)

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

        clf = XGBoost( error_rate_threshold=0.1,
                          objective='reg:squarederror',
                          max_iter=40,
                          max_depth=3,
                          gama=1.0,
                          min_child_weight=1,
                          reg_lambda=1.0,
                          tree_method = 'approx',
                          sketch_eps = 0.3,
                          print_log = True
                          )

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

        clf = XGBoost(error_rate_threshold=0.0, max_iter=2, max_depth=3,objective="binary:logistic")
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


        clf = XGBoost( error_rate_threshold=0.01,
                          objective='binary:logistic',
                          max_iter=30,
                          max_depth=3,
                          gama=1.0,
                          reg_lambda=1.0,
                          tree_method='approx',
                          sketch_eps=0.3,
                          print_log=True
                          )

        clf.fit(trainDataArr, trainLabelArr,learning_rate=0.5)


        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData_2classification('../Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        print('test dataset accuracy: {} '.format( clf.score(testDataArr, testLabelArr) ))

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


    def test_Higgs_dataset(self,n=10000):
        """

        :return:
        """
        Higgs_dataset_path = '../dataset/higgs/kaggle'

        data = np.loadtxt(Higgs_dataset_path + '/training.csv', delimiter=',', skiprows=1, max_rows=n,
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

        start= time.time()

        clf = XGBoost( error_rate_threshold=0.01,
                          objective='binary:logistic',
                          max_iter=30,
                          max_depth=3,
                          gama=1.0,
                          reg_lambda=1.0,
                          tree_method='approx',
                          sketch_eps=0.3,
                          print_log=True
                          )

        clf.fit(X_train, y_train,learning_rate=0.5)

        end =time.time()

        ypred = clf.predict(X_test)

        print("Accuracy:{}".format(accuracy_score(y_test, ypred)))

        print('training cost time :', end - start)


    def test_tiny_multiclassification_dataset(self):
        """
        使用 https://zhuanlan.zhihu.com/p/91652813 中的数据测试  多分类

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

        clf = XGBoost( error_rate_threshold=0.01,objective="multi:softmax" ,num_class=3,max_iter=5, max_depth=1 , print_log=True)
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

        测试  模型的 多分类

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

        clf = XGBoost( error_rate_threshold=0.01,objective="multi:softmax" ,num_class=10 ,max_iter=20, max_depth=3 )
        clf.fit(trainDataArr, trainLabelArr,learning_rate=0.5)


        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../Mnist/mnist_test.csv', n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        print('test dataset accuracy: {} '.format( clf.score(testDataArr, testLabelArr) ))

    def test_iris_dataset(self):

        # 使用iris数据集，其中有三个分类， y的取值为0 , 1 , 2

        X, y = datasets.load_iris(True)  # 包括150行记录
        # 将数据集一分为二，训练数据占80%，测试数据占20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=188)

        clf = XGBoost( error_rate_threshold=0.01,objective="multi:softmax" ,num_class=3 ,max_iter=3, max_depth=2, print_log=True)
        clf.fit(X_train, y_train,learning_rate=0.5)

        print('test dataset accuracy: {} '.format( clf.score(X_test, y_test) ))


if __name__ == '__main__':
    test = Test()


    # test.test_boston_regress_dataset()

    test.test_Mnist_dataset_2classification(6000,1000)

    # test.test_Mnist_dataset_2classification(60000, 10000)

    # test.test_Higgs_dataset()

    # test.test_tiny_multiclassification_dataset()

    # test.test_Mnist_dataset(6000,1000)

    # test.test_Mnist_dataset(60000, 10000)

    # test.test_iris_dataset()


