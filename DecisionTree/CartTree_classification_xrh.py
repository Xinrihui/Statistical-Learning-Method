
import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split


from collections import Counter
from collections import defaultdict

from sklearn.tree import DecisionTreeClassifier,plot_tree

import time

import matplotlib.pyplot as plt




class CartTree_Category_Lib:
    """

    Cart 分类树 相关的 函数库

     ( 适用于 类别 类型 训练数据 )

    Author: xrh
    Date: 2021-04-02

    ref:

    """

    def calc_gini(self, trainLabelArr, sample_weight=None):
        """
        计算数据集 D的 基尼指数 (基尼不纯度)

        :param trainLabelArr:当前数据集的标签集
        :param sample_weight: 样本的权重 ( 适用于 AdaBoost )

        :return: 基尼指数

        """

        N = len(trainLabelArr)  # 数据集的总行数

        if sample_weight is None:
            sample_weight = np.ones(N, dtype=int)

        D = np.sum(sample_weight)  # 数据集中所有样本的 权重和

        LabelSet = {ele for ele in trainLabelArr}  # trainLabelArr 中所有标签的类别

        sum_p = 0

        for label in LabelSet:

            C_k = np.sum(sample_weight[trainLabelArr == label])  # 类型为k 的样本的权重和

            p = C_k / D

            sum_p += p ** 2

        gini = 1 - sum_p

        return gini

    def calc_gini_A(self, trainDataArr_DevFeature, trainLabelArr, split_value=None, sample_weight=None):
        """
        计算 条件 基尼指数

        只对 所关心的 特征 对应的那一列 进行计算，提升训练速度

        :param trainDataArr_DevFeature: 切割后只有feature那列数据的数组
        :param trainLabelArr: 标签集数组

        :param split_value: 切分特征的 特征值
        :param sample_weight: 样本的权重 ( 适用于 AdaBoost )
        :return: 条件基尼指数

        """
        N = len(trainLabelArr)  # 数据集的总行数

        if sample_weight is None:
            sample_weight = np.ones(N, dtype=int)

        D = np.sum(sample_weight)  # 数据集中所有样本的 权重和

        gini_A = 0

        if split_value == None:  # 未指定划分 的特征值

            A_set = {A_i for A_i in trainDataArr_DevFeature}  # trainDataArr_DevFeature 中的 所有取值

            for i in A_set:
                D_i = np.sum(sample_weight[trainDataArr_DevFeature == i])  # 特征值为 i 的 样本的权重和

                p_i = D_i / D

                gini_A += p_i * self.calc_gini(trainLabelArr[trainDataArr_DevFeature == i],
                                               sample_weight[trainDataArr_DevFeature == i])

        else:  # 指定 划分的特征值, 把集合 根据特征值划分为2个子集
            #  D_1 满足 特征值的 子集合
            #  D_2 不满足 特征值的 子集合

            D_1 = np.sum(sample_weight[trainDataArr_DevFeature == split_value])  # 特征值为 split_value 的 样本的权重和

            p_1 = D_1 / D

            p_2 = 1 - p_1

            gini_A = p_1 * self.calc_gini(trainLabelArr[trainDataArr_DevFeature == split_value],
                                          sample_weight[trainDataArr_DevFeature == split_value]) \
                     + p_2 * self.calc_gini(trainLabelArr[trainDataArr_DevFeature != split_value],
                                            sample_weight[trainDataArr_DevFeature != split_value])

        return gini_A

    def select_min_gini(self, trainDataArr, trainLabelArr, feature_value_set, sample_weight=None):

        """
        选择 条件基 尼指数 最小的特征

        :param trainDataArr: shape=(60000,784)
        :param trainLabelArr: shape=(60000,1)
        :param feature_value_set:  可供选择 的特征集合 , 包括 (特征, 切分值)
        :param sample_weight: 样本的权重 ( 适用于 AdaBoost )

        :return:
        """

        N = len(trainLabelArr)  # 数据集的总行数

        if sample_weight is None:
            sample_weight = np.ones(N, dtype=int)

        mini_gini_A = float('inf')  # 最小 条件基尼指数
        mini_gini_A_feature = 0  # 取得 最小条件基尼指数的 特征
        mini_gini_A_feature_split = None  # 取得 最小条件基尼指数的特征 的切分点

        for i, v in feature_value_set:  # 遍历 (特征, 特征的切分值)

            current = self.calc_gini_A(trainDataArr[:, i], trainLabelArr, v, sample_weight)  # 选择第i个特征作为划分特征 的条件基尼指数

            # print('gini(D,A:{},v:{})={}'.format(i,v,current))

            if current < mini_gini_A:
                mini_gini_A = current
                mini_gini_A_feature = i

                mini_gini_A_feature_split = v

        best_feature_value = (mini_gini_A_feature, mini_gini_A_feature_split)

        return best_feature_value, mini_gini_A


class CartTree_Lib:
    """

    Cart 分类树 相关的 函数库

    ( 适用于 数值 类型 训练数据 )


    Author: xrh
    Date: 2021-03-26

    ref:

    """

    def calc_gini(self, trainLabelArr, sample_weight=None):
        """
        计算数据集 D的 基尼指数 (基尼不纯度)

        :param trainLabelArr:当前数据集的标签集
        :param sample_weight: 样本的权重 ( 适用于 AdaBoost )

        :return: 基尼指数

        """

        N = len(trainLabelArr)  # 数据集的总行数

        if sample_weight is None:
            sample_weight = np.ones(N, dtype=int)

        D = np.sum(sample_weight)  # 数据集中所有样本的 权重和

        LabelSet = {ele for ele in trainLabelArr}  # trainLabelArr 中所有标签的类别

        sum_p = 0

        for label in LabelSet:
            C_k = np.sum(sample_weight[trainLabelArr == label])  # 类型为k 的样本的权重和

            p = C_k / D

            sum_p += p ** 2

        gini = 1 - sum_p

        return gini

    def calc_gini_A(self, trainDataArr_DevFeature, trainLabelArr, split_value=None, sample_weight=None):
        """
        计算 条件 基尼指数

        只对 所关心的 特征 对应的那一列 进行计算，提升训练速度

        :param trainDataArr_DevFeature: 切割后只有feature那列数据的数组
        :param trainLabelArr: 标签集数组

        :param split_value: 切分特征的 特征值
        :param sample_weight: 样本的权重 ( 适用于 AdaBoost )
        :return: 条件基尼指数

        """
        N = len(trainLabelArr)  # 数据集的总行数

        if sample_weight is None:
            sample_weight = np.ones(N, dtype=int)

        D = np.sum(sample_weight)  # 数据集中所有样本的 权重和

        gini_A = 0

        if split_value == None:  # 未指定划分 的特征值

            A_set = {A_i for A_i in trainDataArr_DevFeature}  # trainDataArr_DevFeature 中的 所有取值

            for i in A_set:
                D_i = np.sum(sample_weight[trainDataArr_DevFeature <= i])  # 特征值为 i 的 样本的权重和

                p_i = D_i / D

                gini_A += p_i * self.calc_gini(trainLabelArr[trainDataArr_DevFeature <= i],
                                               sample_weight[trainDataArr_DevFeature <= i])

        else:  # 指定 划分的特征值, 把集合 根据特征值划分为2个子集
            #  D_1 满足 <= 特征值的 子集合
            #  D_2 满足 > 特征值的 子集合

            D_1 = np.sum(sample_weight[trainDataArr_DevFeature <= split_value])  # 特征值为 split_value 的 样本的权重和

            p_1 = D_1 / D

            p_2 = 1 - p_1

            gini_A = p_1 * self.calc_gini(trainLabelArr[trainDataArr_DevFeature <= split_value],
                                          sample_weight[trainDataArr_DevFeature <= split_value]) \
                     + p_2 * self.calc_gini(trainLabelArr[trainDataArr_DevFeature > split_value],
                                            sample_weight[trainDataArr_DevFeature > split_value])

        return gini_A

    def select_min_gini(self, trainDataArr, trainLabelArr, feature_value_set, sample_weight=None):

        """
        选择 条件基尼指数 最小的特征

        :param trainDataArr: shape=(60000,784)
        :param trainLabelArr: shape=(60000,1)
        :param feature_value_set:  可供选择 的特征集合 , 包括 (特征, 切分值)
        :param sample_weight: 样本的权重 ( 适用于 AdaBoost )

        :return:
        """

        N = len(trainLabelArr)  # 数据集的总行数

        if sample_weight is None:
            sample_weight = np.ones(N, dtype=int)

        mini_gini_A = float('inf')  # 最小 条件基尼指数
        mini_gini_A_feature = 0  # 取得 最小条件基尼指数的 特征
        mini_gini_A_feature_split = None  # 取得 最小条件基尼指数的特征 的切分点

        for i, v in feature_value_set:  # 遍历 (特征, 特征的切分值)

            current = self.calc_gini_A(trainDataArr[:, i], trainLabelArr, v, sample_weight)  # 选择第i个特征作为划分特征 的条件基尼指数

            # print('gini(D,A:{},v:{})={}'.format(i,v,current))

            if current < mini_gini_A:
                mini_gini_A = current
                mini_gini_A_feature = i

                mini_gini_A_feature_split = v

        best_feature_value = (mini_gini_A_feature, mini_gini_A_feature_split)

        return best_feature_value, mini_gini_A

# 树节点
class Node:
    def __init__(self, label=None, sample_N=None, gini=None, feature=None, feature_value=None, prev_feature=None,
                 prev_feature_value=None, value=None,childs=None):

        self.label = label  # 叶子节点才有标签

        self.sample_N = sample_N # 当前节点的样本总数

        self.gini=gini

        self.feature = feature  # 非叶子节点, 划分 子节点的特征
        self.feature_value = feature_value

        self.prev_feature = prev_feature
        self.prev_feature_value = prev_feature_value

        self.value=value
        self.childs = childs


#TODO:
# 1. 权重( sampel_weight ) 的计算出现问题 , 导致 模型的 预测准确率差 (已完成)
# 2. 对于 数值 类型的 特征, 要对 它进行区间的划分 , 进而生成子节点

class CartTree_Category(CartTree_Category_Lib):

    """
    决策树的 Cart分类 算法( 适用于 类别型训练数据 )

    1.支持样本权重
    2.样本中的 所有特征 均为 类别型特征
    3.未实现剪枝
    
        
    test1: 多分类任务
    数据集：Mnist
    参数: max_depth=50
    训练集数量：60000
    测试集数量：10000
    正确率：0.86
    模型训练时长：319s
    
    Author: xrh
    Date: 2021-03-26

    """

    def __init__(self, root=None, threshold=0.1 , max_depth=1):

        self.root = root
        self.threshold = threshold  # 信息增益的 阈值

        self.max_depth=max_depth # 树的最大深度

    def __pure_dataset(self, trainLabelArr):
        """
        判断 数据集 D 是否纯净
        """
        dict_labels = Counter(trainLabelArr.flatten())

        return len(dict_labels) == 1

    def __class_value(self,trainLabelArr,sample_weight):
        """
        返回数据集D中 所有类型的 样本的 权重和

        :param trainLabelArr:
        :param sample_weight:
        :return:
        """
        dict_labels=defaultdict(int)

        for i in range(len(trainLabelArr)):

            k=trainLabelArr[i]

            dict_labels[k]+=sample_weight[i] #　某个类别的样本的权重和

        return dict_labels



    def major_class(self, trainLabelArr , sample_weight):
        """
        返回 数据集 D 中 数量最多的样本 的类别

        样本权值 sample_weight ，本质上是样本的数量分布, 因此 判断 哪种类型的样本最多时, 需要 考虑 样本权重

        """
        # dict_labels = Counter(trainLabelArr.flatten())
        # max_num_label=dict_labels.most_common(1)[0][0]

        dict_labels=defaultdict(int)

        for i in range(len(trainLabelArr)):

            k=trainLabelArr[i]

            dict_labels[k]+=sample_weight[i] #　某个类别的样本的权重和

        max_num_label= max( dict_labels.items() , key=lambda ele: ele[1] )[0] # 找 权重和　最大的　类别

        return max_num_label

    def __build_tree(self, trainDataArr, trainLabelArr, feature_value_set, tree_depth,sample_weight,
                     prev_feature=None, prev_feature_value=None,father_label=None):
        """
        递归 构建树
        
        :param trainDataArr: 
        :param trainLabelArr: 
        :param feature_value_set: 
        :param tree_depth: 
        :param sample_weight: 
        :param prev_feature: 
        :param prev_feature_value: 
        :return: 
        """

        T = Node()

        T.prev_feature = prev_feature # 标记 父节点的 划分特征
        T.prev_feature_value = prev_feature_value # 标记 通过父节点的 哪一个 分支走到 当前节点

        if len(trainLabelArr) == 0:  # 数据集已经为空, 形成叶子节点

            T.label = father_label  # 说明不能再往下划分了, 使用 上一个节点( 父亲节点 ) 给它的标签值

        else:

            gini= self.calc_gini(trainLabelArr,sample_weight)

            T.gini=gini
            T.sample_N= np.shape(trainLabelArr)[0]

            dict_labels= self.__class_value(trainLabelArr, sample_weight).items() # 数据集 D中 所有类型的 样本的 权重和 { 类型: 权重 }

            T.value=dict_labels # 所有类型的 样本的 权重和

            max_num_label = max(dict_labels, key=lambda ele: ele[1])[0]  #  权重和　最大的类别

            if self.__pure_dataset(trainLabelArr) == True:  # 数据集 已经纯净, 无需往下划分, 形成叶子节点

                T.label = trainLabelArr[0]

            elif len(feature_value_set) == 0 or tree_depth >= self.max_depth  or gini < self.threshold :  # 所有 (特征, 特征值) 的组合 已经用完,
                                                                                                            # 或者 树的深度达到最大深度 ,
                # 选取 数据集 中最多的样本标签值作为  叶子节点的标签
                T.label = max_num_label

            else:

                best_feature_value,mini_gini_A = self.select_min_gini(trainDataArr, trainLabelArr, feature_value_set, sample_weight)

                # if mini_gini_A < self.threshold:
                #     T.label = self.major_class(trainLabelArr)


                Ag, Ag_split=best_feature_value

                T.feature = Ag # 选择的 最佳特征
                T.feature_value= Ag_split # 最佳特征 的切分点

                T.childs = dict()

                trainDataArr_DevFeature= trainDataArr[:,Ag]

                # CART 树为二叉树
                # 左节点为 满足 切分特征值的 分支
                T.childs[0] = self.__build_tree(trainDataArr[trainDataArr_DevFeature == Ag_split],
                                                  trainLabelArr[trainDataArr_DevFeature == Ag_split], feature_value_set - set([best_feature_value]),
                                                  tree_depth+1,
                                                  sample_weight[trainDataArr_DevFeature == Ag_split],
                                                  prev_feature=Ag,
                                                  prev_feature_value= str(Ag_split)+'-Yes',
                                                  father_label=max_num_label )

                # 右节点为 不满足 切分特征值的 分支
                T.childs[1] = self.__build_tree(trainDataArr[trainDataArr_DevFeature != Ag_split],
                                                  trainLabelArr[trainDataArr_DevFeature != Ag_split], feature_value_set - set([best_feature_value]),
                                                  tree_depth + 1,
                                                  sample_weight[trainDataArr_DevFeature != Ag_split],
                                                  prev_feature=Ag,
                                                  prev_feature_value= str(Ag_split)+'-No',
                                                  father_label=max_num_label )

        print('T.feature:{},T.feature_value:{}, T.gini:{} , T.sample_N:{} '.format(T.feature,T.feature_value,T.gini,T.sample_N))

        print('T.prev_feature:{},T.prev_feature_value:{} '.format(T.prev_feature, T.prev_feature_value))

        print('T.childs:{}'.format(T.childs))
        print('Tree depth:{}'.format(tree_depth))
        print('T.label:{}'.format(T.label))

        print('-----------')

        return T

    @staticmethod
    def get_feature_value_set(trainDataArr):
        """
        由于 cartTree 为二叉树, 
        若某特征 有 3个特征值, 则 需要 3个切分点

        eg. 特征'年龄' 包含特征值: ['青年' , '中年' , '老年']

        切分点为:
        1. 是否为 青年
        2. 是否为 中年
        3. 是否为 老年

        若某特征 有 2个特征值, 则 需要 2 个切分点
        
        返回所有 (特征, 特征切分点) 的组合
        
        :param trainDataArr: 
        :return: 
        """
        feature_value_set = set()  # 可供选择的特征集合 , 包括 (特征, 切分值)

        for i in range(np.shape(trainDataArr)[1]):  # 遍历所有的特征

            trainDataArr_DevFeature = trainDataArr[:, i]  # 特征 i 单独抽出来

            A_set = {A_i for A_i in trainDataArr_DevFeature}  # trainDataArr_DevFeature 中的 所有取值

            if len(A_set) <= 2:  # 特征 i 的特征值的个数 小于2个

                feature_value_set.add((i, list(A_set)[0]))  #

            else:  # 特征 i 的特征值的个数 >=3 个

                for A_i in A_set:
                    feature_value_set.add((i, A_i))  #

        return feature_value_set


    def fit(self, trainDataArr, trainLabelArr, feature_value_set=None, sample_weight=None):

        N = len(trainLabelArr)  # 数据集的总行数

        if sample_weight is None:

            sample_weight= np.ones(N,dtype=int)

        if feature_value_set is None:

            feature_value_set= self.get_feature_value_set(trainDataArr) # 可供选择的特征集合 , 包括 (特征, 切分值)

        # print('feature_value_set completed')

        self.root = self.__build_tree(trainDataArr, trainLabelArr, feature_value_set,tree_depth=0,sample_weight=sample_weight)

    def __predict(self, row):
        """
        预测 一个样本

        :param row:
        :return:
        """

        p = self.root

        while p.label == None:  # 到达 叶子节点 退出循环

            judge_feature = p.feature  # 当前节点划分的 特征

            if row[judge_feature]== p.feature_value: # 样本 特征的特质值 与 切分点相同, 走左节点
                p = p.childs[0]

            else: # 走右节点
                p = p.childs[1]

        return p.label


    def predict(self, testDataArr):
        """
        预测 测试 数据集，返回预测结果

        :param test_data:
        :return:
        """

        res_list = []

        for row in testDataArr:
            res_list.append( self.__predict(row) )


        return res_list

    def score(self,testDataArr, testLabelArr):

        """
        预测 测试 数据集，返回 正确率

        :param test_data:
        :return:
        """

        res_list= self.predict(testDataArr)

        err_arr = np.ones( len(res_list), dtype=int)
        res_arr=np.array(res_list)
        err_arr[res_arr == testLabelArr] = 0
        err_rate = np.mean(err_arr)

        accuracy=1-err_rate

        return accuracy


class CartTree(CartTree_Lib):
    """
    
    决策树的 Cart分类 算法( 适用于 数值类型训练数据 )

    1.支持样本权重
    2.样本中的 所有特征 均为 数值类型特征
    3.未实现剪枝

    test1: 多分类任务

    数据集：Mnist
    训练集数量：60000
    测试集数量：10000
    参数:  max_depth=50
    正确率：0.86
    模型训练时长：565s

    Author: xrh
    Date: 2021-03-26

    """

    def __init__(self, root=None, threshold=0.1, max_depth=1):

        self.root = root
        self.threshold = threshold  # 信息增益的 阈值

        self.max_depth = max_depth  # 树的最大深度

    def __pure_dataset(self, trainLabelArr):
        """
        判断 数据集 D 是否纯净
        """
        dict_labels = Counter(trainLabelArr.flatten())

        return len(dict_labels) == 1

    def __class_value(self, trainLabelArr, sample_weight):
        """
        返回数据集 D中 所有类型的 样本的 权重和

        样本权值 sample_weight ，本质上是样本的数量分布, 因此 判断种类型的样本最多时, 需要 考虑 样本权重

        :param trainLabelArr:
        :param sample_weight:
        :return:
        """
        dict_labels = defaultdict(int)

        for i in range(len(trainLabelArr)):

            k = trainLabelArr[i]

            dict_labels[k] += sample_weight[i]  # 某个类别的样本的权重和

        return dict_labels

    def __major_class(self, trainLabelArr, sample_weight):
        """
        返回 数据集 D 中 数量最多的样本 的类别

        样本权值 sample_weight ，本质上是样本的数量分布, 因此 判断种类型的样本最多时, 需要 考虑 样本权重

        """
        # dict_labels = Counter(trainLabelArr.flatten())
        # max_num_label=dict_labels.most_common(1)[0][0]

        dict_labels = defaultdict(int)

        for i in range( len(trainLabelArr) ):
            k = trainLabelArr[i]

            dict_labels[k] += sample_weight[i]  # 某个类别的样本的权重和

        max_num_label = max(dict_labels.items(), key=lambda ele: ele[1])[0]  # 找 权重和　最大的　类别

        return max_num_label

    def __build_tree( self, trainDataArr, trainLabelArr, feature_value_set, tree_depth, sample_weight, prev_feature=None,
                     prev_feature_value=None,father_label=None ):
        """
        递归 构建树

        递归结束条件：

        (1) 当前结点包含的样本全属于同一类别，无需划分。

        (2) 当前属性集为空，或所有样本在所有属性上的取值相同，无法划分：把当前结点标记为叶节点，并将其类别设定为该结点所含样本最多的类别。

        属性集为空的情况：假设有六个特征，六个特征全部用完发现，数据集中还是存在不同类别数据的情况。

        当前特征值全都相同，在类别中有不同取值。


        (3)当前结点包含的样本集合为空，不能划分：将类别设定为父节点所含样本最多的类别。

        出现这个情况的原因是：在生成决策树的过程中，数据按照特征不断的划分，很有可能在使用这个特征某一个值之前，已经可以判断包含该特征值的类别了。所以会出现空的情况。


        :param trainDataArr:
        :param trainLabelArr:
        :param feature_value_set:
        :param tree_depth:
        :param sample_weight:
        :param prev_feature:
        :param prev_feature_value:
        :return:
        """

        T = Node()

        T.prev_feature = prev_feature  # 标记 父节点的 划分特征
        T.prev_feature_value = prev_feature_value  # 标记 通过父节点的 哪一个 分支走到 当前节点

        if len(trainLabelArr) == 0:  # 数据集已经为空, 形成叶子节点

            T.label = father_label  # 说明不能再往下划分了, 使用 上一个节点( 父亲节点 ) 给它的标签值

        else: #

            gini = self.calc_gini(trainLabelArr, sample_weight)

            T.gini = gini
            T.sample_N = np.shape(trainLabelArr)[0]

            dict_labels= self.__class_value(trainLabelArr, sample_weight).items() # 数据集 D中 所有类型的 样本的 权重和 { 类型: 权重 }

            T.value=dict_labels # 所有类型的 样本的 权重和

            max_num_label = max(dict_labels, key=lambda ele: ele[1])[0]  #  权重和　最大的类别

            if self.__pure_dataset(trainLabelArr) == True:  # 数据集 已经纯净, 无需往下划分, 形成叶子节点

                T.label = trainLabelArr[0]

            elif len(feature_value_set) == 0 or tree_depth >= self.max_depth or gini < self.threshold:  # 所有 切分(特征, 特征值) 的组合 已经用完,
                                                                                                        # 或者 树的深度达到最大深度 ,
                                                                                                        # 选取 数据集 中最多的样本标签值作为  叶子节点的标签
                T.label = max_num_label

            else:

                best_feature_value, mini_gini_A = self.select_min_gini(trainDataArr, trainLabelArr, feature_value_set,
                                                                       sample_weight)

                Ag, Ag_split = best_feature_value

                T.feature = Ag  # 选择的 最佳特征
                T.feature_value = Ag_split  # 最佳特征 的切分点

                T.childs = dict()

                trainDataArr_DevFeature = trainDataArr[:, Ag]

                # CART 树为二叉树
                # 左节点为  <= 特征值的 分支
                T.childs[0] = self.__build_tree(trainDataArr[trainDataArr_DevFeature <= Ag_split],
                                                trainLabelArr[trainDataArr_DevFeature <= Ag_split],
                                                feature_value_set - { (best_feature_value) },
                                                tree_depth + 1,
                                                sample_weight[trainDataArr_DevFeature <= Ag_split],
                                                prev_feature=Ag,
                                                prev_feature_value=' <= ' + str(Ag_split),father_label=max_num_label )

                # 右节点为 > 切分特征值的 分支
                T.childs[1] = self.__build_tree(trainDataArr[trainDataArr_DevFeature > Ag_split],
                                                trainLabelArr[trainDataArr_DevFeature > Ag_split],
                                                feature_value_set - { (best_feature_value) },
                                                tree_depth + 1,
                                                sample_weight[trainDataArr_DevFeature > Ag_split],
                                                prev_feature=Ag,
                                                prev_feature_value=' > ' + str(Ag_split) ,father_label=max_num_label)

        print('T.feature:{},T.feature_value:{}, T.gini:{} , T.sample_N:{} , T.value:{} '.format(T.feature, T.feature_value, T.gini,
                                                                                   T.sample_N,T.value))

        print('T.prev_feature:{},T.prev_feature_value:{} '.format(T.prev_feature, T.prev_feature_value))

        print('T.childs:{}'.format(T.childs))
        print('Tree depth:{}'.format(tree_depth))
        print('T.label:{}'.format(T.label))

        print('-----------')

        return T

    @staticmethod
    def get_feature_value_set(trainDataArr):
        """
        cartTree 为二叉树,

        对于离散型特征：

            若为 可比型 (数值类型特征)，比如电影评分等级，特征的所有取值为 [1, 2, 3, 4, 5]，那么按照阈值 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 分别划分即可，这里选择了 6 个划分点；

            *若为 不可比型，即Categorical类型，比如职业，对应的可能取值为 [0, 1, 2]，那么划分取值为 0, 1, 2，表示是否等于0，是否等于1，是否等于2，注意sklearn里面处理这类数据并没有对应的实现方法，而是采用的是离散型特征可比型处理策略。即：CART可以做类别型数据，但是sklearn没实现。

        对于连续型特征：
            那么取值就很多了，比如 [0.13, 0.123, 0.18, 0.23, ...]，那么要每两个值之间都取一个阈值划分点。

        综上, 排除 * 的情况, 我们可以简单 使用 特征值作为 切分 阈值

         eg. 特征'电影评分' 包含特征值: [1, 2, 3]

        切分点为:
        1.  电影评分 <=1 | 电影评分 >1
        2.  电影评分 <=2 | 电影评分 >2
        3.  电影评分 <=3 | 电影评分 >3

        返回所有 (特征, 特征切分点) 的组合

        :param trainDataArr:
        :return:
        """

        feature_value_set = set()  # 可供选择的特征集合 , 包括 (特征, 切分值)

        for i in range(np.shape(trainDataArr)[1]):  # 遍历所有的特征

            trainDataArr_DevFeature = trainDataArr[:, i]  # 特征 i 单独抽出来

            A_set = {A_i for A_i in trainDataArr_DevFeature}  # trainDataArr_DevFeature 中的 所有取值

            for A_i in A_set:
                feature_value_set.add((i, A_i))  #

        return feature_value_set

    def fit(self, trainDataArr, trainLabelArr, feature_value_set=None, sample_weight=None):

        N = len(trainLabelArr)  # 数据集的总行数

        if sample_weight is None:
            sample_weight = np.ones(N, dtype=int)

        if feature_value_set is None:
            feature_value_set = self.get_feature_value_set(trainDataArr)  # 可供选择的特征集合 , 包括 (特征, 切分值)

        # print('feature_value_set completed')

        self.root = self.__build_tree(trainDataArr, trainLabelArr, feature_value_set, tree_depth=0,
                                      sample_weight=sample_weight)

    def __predict(self, row):
        """
        预测 一个样本

        :param row:
        :return:
        """

        p = self.root

        while p.label == None:  # 到达 叶子节点 退出循环

            judge_feature = p.feature  # 当前节点划分的 特征

            if row[judge_feature] <= p.feature_value:  # 样本 特征的特质值 <= 切分点, 走左节点
                p = p.childs[0]

            else:  # 走右节点
                p = p.childs[1]

        return p.label

    def predict(self, testDataArr):
        """
        预测 测试 数据集，返回预测结果

        :param test_data:
        :return:
        """

        res_list = []

        for row in testDataArr:
            res_list.append(self.__predict(row))

        return res_list

    def score(self, testDataArr, testLabelArr):

        """
        预测 测试 数据集，返回 正确率

        :param test_data:
        :return:
        """

        res_list = self.predict(testDataArr)

        err_arr = np.ones(len(res_list), dtype=int)
        res_arr = np.array(res_list)
        err_arr[res_arr == testLabelArr] = 0
        err_rate = np.mean(err_arr)

        accuracy = 1 - err_rate

        return accuracy

class PreProcess:
    """

    对数据集进行 预处理的方法集合

    Author: xrh
    Date: 2021-04-01

    """

    @staticmethod
    def ordinal_encoding(dataset):
        """
        序号编码

        将数据集中的 类别型特征 转换为 数值型特征

        :param dataset:
        :return:
        dataset - 转换后的 数据集
        features_idx_category_dic - 所有 类别型特征 和 转换后的数值型特征的对应关系
        """

        N, feature_Num = np.shape(dataset)

        dic_features_category_value=[] # 存储 所有 类别型特征 和 转换后的数值型特征的对应关系

        for i in range(feature_Num):

            feature_i_category_set = {ele for ele in dataset[:, i]}  # 第i 个特征的 类别集合

            dic_feature_i_category_value = {}

            for category, idx in zip(feature_i_category_set, range(len(feature_i_category_set))):
                dic_feature_i_category_value[category] = idx

            dic_features_category_value.append( dic_feature_i_category_value ) # 存储 所有 类别型特征 和 转换后的数值型特征的对应关系

            dataset[:, i] = list( map( lambda x: dic_feature_i_category_value[x], dataset[:, i] ) )

        return dataset,dic_features_category_value

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

    def test_CartTree_Category_Lib(self):
        """
        CartTree_Lib  测试

        :return:
        """

        datasets, label_name = self.__create_tarin_data()

        datasetsArr = np.array(datasets)

        trainDataArr= datasetsArr[:, 0:-1]
        trainLabelArr= datasetsArr[:, -1]

        Lib = CartTree_Category_Lib()

        feature_value_set=set() # 可供选择的特征集合 , 包括 (特征, 切分值)

        for i in range(np.shape(trainDataArr)[1]):  # 遍历所有的特征

            trainDataArr_DevFeature= trainDataArr[:,i] # 特征 i 单独抽出来

            A_set = {A_i for A_i in trainDataArr_DevFeature}  # trainDataArr_DevFeature 中的 所有取值

            if len(A_set)<=2: # 特征 i 的特征值的个数 小于2个

                feature_value_set.add( (i ,list(A_set)[0]) ) #

            else: # 特征 i 的特征值的个数 >=3 个

                for A_i in A_set:

                    feature_value_set.add((i, A_i))  #



        best_feature_value, mini_gini_A= Lib.select_min_gini(trainDataArr,trainLabelArr,feature_value_set)

        print('best feature: {}, split value:{}, gini:{}  '.format(best_feature_value[0],best_feature_value[1], mini_gini_A))

    def test_CartTree_Lib(self):
        """
        CartTree_Lib  测试

        :return:
        """

        datasets, label_name = self.__create_tarin_data()

        # 将 类别型训练数据 转换为数值型训练数据
        datasetsArr, features_idx_category_dic = PreProcess.ordinal_encoding(np.array(datasets))

        print('features_idx_category_dic: ',features_idx_category_dic)

        trainDataArr = datasetsArr[:, 0:-1]
        trainLabelArr = datasetsArr[:, -1]

        Lib = CartTree_Lib()

        feature_value_set = set()  # 可供选择的特征集合 , 包括 (特征, 切分值)

        for i in range(np.shape(trainDataArr)[1]):  # 遍历所有的特征

            trainDataArr_DevFeature = trainDataArr[:, i]  # 特征 i 单独抽出来

            A_set = {A_i for A_i in trainDataArr_DevFeature}  # trainDataArr_DevFeature 中的 所有取值


            for A_i in A_set:
                feature_value_set.add((i, A_i))  #

        # print(feature_value_set)

        best_feature_value, mini_gini_A = Lib.select_min_gini(trainDataArr, trainLabelArr, feature_value_set)

        print('best feature: {}, split value:{}, gini:{}  '.format(best_feature_value[0], best_feature_value[1],
                                                                   mini_gini_A))

    def test_small_category_dataset(self):
        """
        
        利用《统计学习方法》 表 5.1 中的数据集 测试 决策树 ID3

        数据集 中的特征均为 类别型特征

        :return:
        """

        # 获取训练集
        datasets, labels = self.__create_tarin_data()

        datasetsArr= np.array(datasets)


        trainDataArr= datasetsArr[:, 0:-1]
        trainLabelArr= datasetsArr[:, -1]

        # 开始时间
        start = time.time()

        # 创建决策树
        print('start create tree')

        CT = CartTree_Category(threshold=0.1 , max_depth=10)
        CT.fit(trainDataArr, trainLabelArr)

        print(' tree complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        # 测试数据集
        datasets_test, _ = self.__create_test_data()

        datasetsArr_test = np.array(datasets_test)

        testDataArr= datasetsArr_test[:, 0:-1]
        testLabelArr= datasetsArr_test[:, -1]

        print('res:', CT.score(testDataArr, testLabelArr))


    def test_small_value_dataset(self):
        """

        利用《统计学习方法》 表 5.1 中的数据集 测试 决策树 ID3

        数据集 中的特征均为 类别型特征, 需要将其转换为 数值类型的特征

        :return:
        """

        # 获取训练集
        datasets, labels = self.__create_tarin_data()

        # 将 类别型训练数据 转换为数值型训练数据
        datasetsArr,dic_features_category_value = PreProcess.ordinal_encoding(np.array(datasets))

        print('dic_features_category_value:',dic_features_category_value)

        trainDataArr = datasetsArr[:, 0:-1]
        trainLabelArr = datasetsArr[:, -1]

        # 开始时间
        start = time.time()

        # 创建决策树
        print('start create tree')

        CT = CartTree(threshold=0.1, max_depth=10)
        CT.fit(trainDataArr, trainLabelArr)

        print(' tree complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        # 测试数据集

        datasetsArr_test= datasetsArr # 用 训练数据集 作为 测试数据集

        testDataArr = datasetsArr_test[:, 0:-1]
        testLabelArr = datasetsArr_test[:, -1]

        print('test dataset accuracy :', CT.score(testDataArr, testLabelArr))

        # sklearn 的决策树
        DT = DecisionTreeClassifier( max_depth=10, criterion="gini", splitter='best' )
        DT.fit(trainDataArr, trainLabelArr)

        for name, val in zip(labels, DT.feature_importances_):  # 打印 所有特征的重要程度
            print("{} -> {}".format(name, val))

        for name, val in zip(DT.tree_.feature, DT.tree_.threshold):  # 依次打印 切分的特征 和 切分点
            print("{} -> {}".format(name, val) )

        plt.figure(figsize=(18, 10))
        plot_tree(DT)
        plt.show()

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
        利用 Mnist 数据集 测试 决策树 CART

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

        # CT = CartTree_Category(threshold=0.1, max_depth=50)

        CT = CartTree(threshold=0.1 , max_depth=15)
        CT.fit(trainDataArr, trainLabelArr)


        # 结束时间
        end = time.time()
        print('training cost time :', end - start)

        # 获取测试集
        testDataList, testLabelList = self.loadData('../Mnist/mnist_test.csv',n=n_test)

        print('test data, row num:{} , column num:{} '.format(len(testDataList), len(testDataList[0])))

        testDataArr = np.array(testDataList)
        testLabelArr = np.array(testLabelList)

        print('test accuracy :', CT.score(testDataArr,testLabelArr))

        # sklearn
        DT=DecisionTreeClassifier(max_depth=15,criterion="gini", splitter='best')
        DT.fit(trainDataArr, trainLabelArr)

        for name, val in zip(DT.tree_.feature, DT.tree_.threshold):
            print("{} -> {}".format(name, val))

        plt.figure(figsize=(20, 12))
        plot_tree(DT)
        plt.show()


    def test_iris_dataset(self ):

        # 使用iris数据集，其中有三个分类， y的取值为0,1，2
        X, y = datasets.load_iris(True)  # 包括150行记录
        # 将数据集一分为二，训练数据占80%，测试数据占20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)

        CT = CartTree(max_depth=3)  # 第 m 个弱分类器
        CT.fit(X, y)

        print('by xrh , test accuracy :', CT.score(X_test, y_test))

        DT=DecisionTreeClassifier(max_depth=3)
        DT.fit(X_train,y_train)

        print('by sklearn ,test accuracy :', DT.score(X_test,y_test))


        for name, val in zip(DT.tree_.feature, DT.tree_.threshold):
            print("{} -> {}".format(name, val))

        plt.figure(figsize=(20, 12))
        plot_tree(DT)
        plt.show()



if __name__ == '__main__':

    test=Test()

    # test.test_CartTree_Lib()

    # test.test_small_category_dataset()

    # test.test_small_value_dataset()

    test.test_Mnist_dataset(60000,10000)

    # test.test_iris_dataset()


