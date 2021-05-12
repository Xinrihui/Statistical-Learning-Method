
import numpy as np
import time

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error



# 树节点
class Node:
    def __init__(self, label=None, feature=None, feature_split=None, prev_feature=None, prev_feature_split=None,
                 loss=None,
                 sample_N=None,
                 childs=None):

        self.label = label  # 叶子节点才有标签

        self.feature = feature  # 非叶子节点, 划分 子节点的特征
        self.feature_split = feature_split  # 划分 子节点的特征 的切分点

        self.prev_feature = prev_feature
        self.prev_feature_split = prev_feature_split

        self.loss=loss
        self.sample_N=sample_N

        self.childs = childs


class RegresionTree_deprecated():
    """

    CART 树的最小二乘回归算法

    Author: xrh
    Date: 2021-03-14

    """

    def __init__(self, root=None, threshold=0.2 , max_depth=1):

        self.root = root
        self.threshold = threshold  # 损失的 阈值

        self.max_depth=max_depth #


    def select_min_square_loss_feature(self, trainDataArr, trainLabelArr, feature_set):

        """

        选择最优的特征 和 最优的 切分点

        """

        Ag = None # 最佳特征
        Ag_split = None #最佳特征的　切分点

        Ag_c1=None # 最佳特征 的最佳切分点 切分后 左区域的样本均值
        Ag_c2 = None # 最佳特征 的最佳切分点 切分后 右区域的样本均值

        min_square_loss = float('inf')

        for A_j in feature_set:  # 遍历特征

            trainDataArr_DevFeature = trainDataArr[:, A_j]

            DevFeature_value_set = {v for v in trainDataArr_DevFeature}

            A_j_min_loss = float( 'inf' )  # A_j 的 最小损失
            split_opt = None  # A_j 的最佳分割点

            c1_opt=None
            c2_opt = None

            for s_i in DevFeature_value_set:  # 遍历特征的 所有取值

                # s_i 将数据集 划分为两半 R1 和 R2
                R1 = trainLabelArr[trainDataArr_DevFeature <= s_i]  # 特征值 取值为 <= s_i
                R2 = trainLabelArr[trainDataArr_DevFeature > s_i]

                c1 = np.mean(R1)
                c2 = np.mean(R2)

                loss = np.sum(np.square(R1 - c1)) + np.sum(np.square(R2 - c2))  # 计算平方和

                if loss < A_j_min_loss:
                    A_j_min_loss = loss
                    split_opt = s_i

                    c1_opt=c1
                    c2_opt=c2

            if A_j_min_loss < min_square_loss:
                min_square_loss = A_j_min_loss
                Ag = A_j
                Ag_split = split_opt

                Ag_c1=c1_opt
                Ag_c2=c2_opt

        return Ag, Ag_split, min_square_loss,Ag_c1,Ag_c2


    def __build_tree(self, trainDataArr, trainLabelArr, feature_set,tree_depth, prev_feature=None, prev_feature_split=None,father_label=None):

        T = Node()

        T.prev_feature = prev_feature
        T.prev_feature_split = prev_feature_split

        if len( trainLabelArr ) ==0:  #  形成叶子节点

            T.label=father_label # 说明不能再往下划分了, 使用 上一个 节点给它的标签值

        elif  len( trainLabelArr ) ==1:  #  trainLabelArr 中 仅仅有 1个元素
                                         #  形成叶子节点
            T.label = trainLabelArr[0]

        else:

            Ag, Ag_split, min_square_loss,Ag_c1,Ag_c2 = self.select_min_square_loss_feature(trainDataArr, trainLabelArr,
                                                                                feature_set)

            # print('Ag:{} , Ag_split:{}, min_square_loss:{},c1:{},c2:{}'.format(Ag,Ag_split,min_square_loss,Ag_c1,Ag_c2))

            T.feature = Ag
            T.feature_split = Ag_split

            if min_square_loss < self.threshold or tree_depth >= self.max_depth :  # 损失 小于 阈值 , 进行最后一次划分

                T.childs = dict()
                T.childs[0]=  self.__build_tree([],
                                                [], feature_set ,tree_depth+1,
                                                prev_feature=T.feature,
                                                prev_feature_split='<='+str(Ag_split),father_label=Ag_c1) # 左边的均值 作为下一个节点的标签值

                T.childs[1] = self.__build_tree([],
                                                [], feature_set ,tree_depth+1,
                                                prev_feature=T.feature,
                                                prev_feature_split='>'+str(Ag_split) ,father_label=Ag_c2) # 右边的均值 作为下一个节点的标签值
            else:

                T.childs = dict()

                trainDataArr_DevFeature = trainDataArr[:, Ag]

                # 二叉树 只有左右两个节点
                #　小于等于　切分点
                T.childs[0] = self.__build_tree(trainDataArr[trainDataArr_DevFeature <= Ag_split],
                                                trainLabelArr[trainDataArr_DevFeature <= Ag_split], feature_set ,tree_depth+1,
                                                prev_feature=T.feature,
                                                prev_feature_split='<='+str(Ag_split),father_label=Ag_c1)

                # 大于 切分点
                T.childs[1] = self.__build_tree(trainDataArr[trainDataArr_DevFeature > Ag_split],
                                                trainLabelArr[trainDataArr_DevFeature > Ag_split], feature_set ,tree_depth+1,
                                                prev_feature=T.feature,
                                                prev_feature_split='>'+str(Ag_split) ,father_label=Ag_c2)

        print('T.feature:{}'.format(T.feature))
        print('T.prev_feature:{},T.prev_feature_split:{} '.format(T.prev_feature, T.prev_feature_split))

        print( 'depth:{} '.format(tree_depth) )
        print( 'T.childs:{}'.format(T.childs) )
        print( 'T.label:{}'.format(T.label) )

        print('-----------')

        return T

    def fit(self, trainDataArr, trainLabelArr):

        feature_set = set(range(len(trainDataArr[0])))  # 特征的总数

        self.root = self.__build_tree(trainDataArr, trainLabelArr, feature_set,0)

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

            if row[judge_feature] <= p.feature_split:
                p = p.childs[0]
            else:
                p = p.childs[1]

        return p.label

    def predict(self, testDataArr):
        """
        推理 测试 数据集，返回预测结果

        :param test_data:
        :return:
        """

        res_list = []

        for row in testDataArr:
            res_list.append( self.__predict(row) )

        return np.array(res_list)


    def score(self, testDataArr, testLabelArr):
        """
        推理 测试 数据集，返回预测 的 平方误差

        :param test_data:
        :return:
        """
        res_list= self.predict(testDataArr)

        square_loss = np.average( np.square( res_list - testLabelArr ) )  # 平方误差

        return  square_loss


class RegresionTree():
    """

    CART 树的最小二乘回归算法

    优化:  因为 cart 树为二叉树, 在训练模型时, 提前先得到所有的 特征 和 特征切分点,
          降低时间复杂度

    Author: xrh
    Date: 2021-03-14

    """

    def __init__(self, root=None, threshold=0.2, max_depth=1,print_log=True):

        self.root = root
        self.threshold = threshold  # 损失的 阈值

        self.max_depth = max_depth  #

        self.print_log=print_log

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


    def select_min_square_loss_feature(self, trainDataArr, trainLabelArr, feature_value_set):

        """

        选择最优的特征 和 最优的 切分点

        """
        Ag = None  # 最佳特征
        Ag_split = None  # 最佳特征的　切分点

        Ag_c1 = None  # 最佳特征 的最佳切分点 切分后 左区域的样本均值
        Ag_c2 = None  # 最佳特征 的最佳切分点 切分后 右区域的样本均值

        min_square_loss = float('inf')

        for A_j,s_i  in feature_value_set:  # 遍历 (特征, 切分值)

            trainDataArr_DevFeature = trainDataArr[:, A_j]

            # s_i 将数据集 划分为两半 R1 和 R2
            R1 = trainLabelArr[trainDataArr_DevFeature <= s_i]  # 特征值 取值为 <= s_i
            R2 = trainLabelArr[trainDataArr_DevFeature > s_i]

            c1 = np.mean(R1)
            c2 = np.mean(R2)

            loss = np.sum(np.square(R1 - c1)) + np.sum(np.square(R2 - c2))  # 计算平方和

            if loss < min_square_loss:
                min_square_loss = loss
                Ag = A_j
                Ag_split = s_i

                Ag_c1 = c1
                Ag_c2 = c2

        return Ag, Ag_split, min_square_loss, Ag_c1, Ag_c2

    def __build_tree(self, trainDataArr, trainLabelArr, feature_value_set, tree_depth, prev_feature=None,
                     prev_feature_split=None, father_label=None):

        T = Node()

        T.prev_feature = prev_feature
        T.prev_feature_split = prev_feature_split

        if len(trainLabelArr) == 0 or len(feature_value_set)==0 :   # 划分后 数据集已经为空 OR (特征,切分值) 已经用完,  形成叶子节点

            T.label = father_label  # 说明不能再往下划分了, 使用 上一个 节点给它的标签值

        elif len(trainLabelArr) == 1:  # trainLabelArr 中 仅仅有 1个元素
            #  形成叶子节点
            T.label = trainLabelArr[0]

        else:

            Ag, Ag_split, min_square_loss, Ag_c1, Ag_c2 = self.select_min_square_loss_feature(trainDataArr,
                                                                                              trainLabelArr,
                                                                                              feature_value_set)
            best_feature_value = (Ag, Ag_split)

            # print('Ag:{} , Ag_split:{}, min_square_loss:{},c1:{},c2:{}'.format(Ag,Ag_split,min_square_loss,Ag_c1,Ag_c2))


            T.feature = Ag
            T.feature_split = Ag_split

            if min_square_loss < self.threshold or tree_depth >= self.max_depth:  # 损失小于阈值 , 进行最后一次划分

                T.childs = dict()
                T.childs[0] = self.__build_tree([],
                                                [], feature_value_set-{(best_feature_value)}, tree_depth + 1,
                                                prev_feature=T.feature,
                                                prev_feature_split='<=' + str(Ag_split),
                                                father_label=Ag_c1)  # 左边的均值 作为下一个节点的标签值

                T.childs[1] = self.__build_tree([],
                                                [], feature_value_set-{(best_feature_value)}, tree_depth + 1,
                                                prev_feature=T.feature,
                                                prev_feature_split='>' + str(Ag_split),
                                                father_label=Ag_c2)  # 右边的均值 作为下一个节点的标签值
            else:

                T.childs = dict()

                trainDataArr_DevFeature = trainDataArr[:, Ag]

                # 二叉树 只有左右两个节点
                # 　小于等于　切分点
                T.childs[0] = self.__build_tree(trainDataArr[trainDataArr_DevFeature <= Ag_split],
                                                trainLabelArr[trainDataArr_DevFeature <= Ag_split],
                                                feature_value_set - {(best_feature_value)},
                                                tree_depth + 1,
                                                prev_feature=T.feature,
                                                prev_feature_split='<=' + str(Ag_split), father_label=Ag_c1)

                # 大于 切分点
                T.childs[1] = self.__build_tree(trainDataArr[trainDataArr_DevFeature > Ag_split],
                                                trainLabelArr[trainDataArr_DevFeature > Ag_split],
                                                feature_value_set - {(best_feature_value)},
                                                tree_depth + 1,
                                                prev_feature=T.feature,
                                                prev_feature_split='>' + str(Ag_split), father_label=Ag_c2)

        if self.print_log:  #

            print('T.feature:{}'.format(T.feature))
            print('T.prev_feature:{},T.prev_feature_split:{} '.format(T.prev_feature, T.prev_feature_split))

            print('depth:{} '.format(tree_depth))
            # print('T.childs:{}'.format(T.childs))
            print('T.label:{}'.format(T.label))

            print('-----------')

        return T

    def fit(self, trainDataArr, trainLabelArr , feature_value_set=None):

        if feature_value_set is None:
            feature_value_set = self.get_feature_value_set(trainDataArr)  # 可供选择的特征集合 , 包括 (特征, 切分值)

        self.root = self.__build_tree(trainDataArr, trainLabelArr, feature_value_set, tree_depth=0)

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

            if row[judge_feature] <= p.feature_split:
                p = p.childs[0]
            else:
                p = p.childs[1]

        return p.label

    def predict(self, testDataArr):
        """
        推理 测试 数据集，返回预测结果

        :param test_data:
        :return:
        """

        res_list = []

        for row in testDataArr:
            res_list.append(self.__predict(row))

        return np.array(res_list)

    def score(self, testDataArr, testLabelArr):
        """
        推理 测试 数据集，返回预测 的 平方误差

        :param test_data:
        :return:
        """
        res_list = self.predict(testDataArr)

        square_loss = np.average(np.square(res_list - testLabelArr))  # 平方误差

        return square_loss


class RegresionTree_GBDT():
    """

    CART 回归树

    1.适用于 二分类 GBDT

    2.对于多分类的 GBDT, 一次 迭代训练 K 颗二分类 CART回归树

    Author: xrh
    Date: 2021-04-10

    """

    def __init__(self, root=None, min_square_loss=0.1, max_depth=2, min_sample_split=2 ,print_log=True):
        """

        :param root: 树的根节点
        :param min_square_loss: 最小平方损失的阈值, 若小于此阈值, 不往下分裂, 形成叶子节点
        :param max_depth: 树的最大深度
        :param min_sample_split: 划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
        :param print_log: 是否打印日志
        """

        self.root = root
        self.min_square_loss = min_square_loss  # 损失的 阈值

        self.max_depth = max_depth  # 树的最大深度

        self.min_sample_split=min_sample_split # TODO: 未使用

        self.print_log=print_log #是否打印日志


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


    def select_min_square_loss_feature(self, trainDataArr, trainLabelArr, feature_value_set):

        """

        选择最优的特征 和 最优的 切分点

        """

        Ag = None  # 最佳特征
        Ag_split = None  # 最佳特征的　切分点

        Ag_c1 = None  # 最佳特征 的最佳切分点 切分后 左区域的样本均值
        Ag_c2 = None  # 最佳特征 的最佳切分点 切分后 右区域的样本均值

        min_square_loss = float('inf')

        for A_j,s_i  in feature_value_set:  # 遍历 (特征, 切分值)

            trainDataArr_DevFeature = trainDataArr[:, A_j]

            # s_i 将数据集 划分为两半 R1 和 R2
            R1 = trainLabelArr[trainDataArr_DevFeature <= s_i]  # 特征值 取值为 <= s_i
            R2 = trainLabelArr[trainDataArr_DevFeature > s_i]

            c1 = np.mean(R1)
            c2 = np.mean(R2)

            loss = np.sum(np.square(R1 - c1)) + np.sum(np.square(R2 - c2))  # 计算平方和

            # TODO: 考虑 trainLabelArr 中存在 Nan , 导致 loss=Nan , 导致 下面不等式 永远不满足条件

            if loss <= min_square_loss:

                min_square_loss = loss
                Ag = A_j
                Ag_split = s_i

                Ag_c1 = c1
                Ag_c2 = c2

        return Ag, Ag_split, min_square_loss, Ag_c1, Ag_c2



    def update_leaf_region_lable(self, trainLabelArr,origin_trainLabelArr):
        """
        更新 叶子节点的 预测值

        (1) 需要考虑 会触发 X/0= Nan 的bug

        :param trainLabelArr: 残差 r
        :param origin_trainLabelArr: 样本标签 y
        :return:
        """

        numerator = np.sum( trainLabelArr )
        if numerator == 0:
            return 0.0
        denominator = np.sum( (origin_trainLabelArr -  trainLabelArr) * ( 1 - origin_trainLabelArr + trainLabelArr ) )
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator


    def cal_node_loss(self,trainLabelArr):
        """
        计算节点的 平方误差损失

        :param trainLabelArr:
        :return:
        """
        R=trainLabelArr
        c = np.mean(R)
        loss = np.sum(np.square(R - c))   # 计算平方和

        return loss

    # TODO: 参考 sklearn , 实现非递归建立树
    def __build_tree(self, trainDataArr, trainLabelArr,origin_trainLabelArr, feature_value_set, tree_depth, prev_feature=None,
                     prev_feature_split=None, father_label=None):
        """
        递归 构建树

        递归结束条件：

        (1) 当前属性集为空，或所有样本在所有属性上的取值相同，无法划分：把当前结点标记为叶节点，并将其类别设定为该结点所含样本最多的类别。

        属性集为空的情况：假设有六个特征，六个特征全部用完发现，数据集中还是存在不同类别数据的情况。

        当前特征值全都相同，在类别中有不同取值。


        (2)当前结点包含的样本集合为空，不能划分：将类别设定为父节点所含样本最多的类别。

        出现这个情况的原因是：在生成决策树的过程中，数据按照特征不断的划分，很有可能在使用这个特征某一个值之前，已经可以判断包含该特征值的类别了。所以会出现空的情况。


        :param trainDataArr:
        :param trainLabelArr:
        :param origin_trainLabelArr:
        :param feature_value_set:
        :param tree_depth:
        :param prev_feature:
        :param prev_feature_split:
        :param father_label:
        :return:
        """

        T = Node()

        T.prev_feature = prev_feature
        T.prev_feature_split = prev_feature_split

        if len(trainLabelArr) == 0 :  #or len(trainLabelArr) <= self.min_sample_split :     # 划分后 数据集已经为空,
                                                                     #  trainLabelArr 中 仅仅有 1个元素

            T.label = father_label  # 说明不能再往下划分了, 使用 上一个 节点给它的标签值

        else:

            loss= self.cal_node_loss(trainLabelArr)

            T.loss = loss
            T.sample_N = np.shape(trainLabelArr)[0]

            leaf_label = self.update_leaf_region_lable( trainLabelArr,origin_trainLabelArr ) # 对于叶子节点区域 ，计算出最佳拟合值

            if len(feature_value_set) == 0 or tree_depth >= self.max_depth or loss <= self.min_square_loss:  # 所有 切分(特征, 特征值) 的组合 已经用完,
                                                                                                    # 或者 树的深度达到最大深度 ,
                                                                                                    # 选取 数据集 中最多的样本标签值作为  叶子节点的标签
                T.label = leaf_label

            else:

                Ag, Ag_split, min_square_loss, Ag_c1, Ag_c2 = self.select_min_square_loss_feature(trainDataArr,
                                                                                                  trainLabelArr,
                                                                                                  feature_value_set)
                best_feature_value = (Ag, Ag_split)

                # print('Ag:{} , Ag_split:{}, min_square_loss:{}, Ag_c1:{},Ag_c1:{}'.format(Ag,Ag_split,min_square_loss,Ag_c1,Ag_c1))

                T.feature = Ag
                T.feature_split = Ag_split

                T.childs = dict()

                trainDataArr_DevFeature = trainDataArr[:, Ag]

                # CART 树为二叉树
                # 左节点为  <=  特征值的 分支
                T.childs[0] = self.__build_tree(trainDataArr[trainDataArr_DevFeature <= Ag_split],
                                                trainLabelArr[trainDataArr_DevFeature <= Ag_split],
                                                origin_trainLabelArr[trainDataArr_DevFeature <= Ag_split],
                                                feature_value_set - {(best_feature_value)},
                                                tree_depth + 1,
                                                prev_feature=T.feature,
                                                prev_feature_split='<=' + str(Ag_split), father_label=leaf_label)

                # 右节点为 > 切分特征值的 分支
                T.childs[1] = self.__build_tree(trainDataArr[trainDataArr_DevFeature > Ag_split],
                                                trainLabelArr[trainDataArr_DevFeature > Ag_split],
                                                origin_trainLabelArr[trainDataArr_DevFeature > Ag_split],
                                                feature_value_set - {(best_feature_value)},
                                                tree_depth + 1,
                                                prev_feature=T.feature,
                                                prev_feature_split='>' + str(Ag_split), father_label=leaf_label)

        if self.print_log: #

            print('T.feature:{},T.feature_split:{}, T.loss:{} , T.sample_N:{}  '.format(T.feature,T.feature_split, T.loss,T.sample_N))
            print('T.prev_feature:{},T.prev_feature_split:{} '.format(T.prev_feature, T.prev_feature_split))
            print('depth:{} '.format(tree_depth))
            # print('T.childs:{}'.format(T.childs))
            print('T.label:{}'.format(T.label))
            print('-----------')

        return T

    def fit(self, trainDataArr, trainLabelArr , origin_trainLabelArr, feature_value_set=None):
        """

        :param trainDataArr:  特征 X
        :param trainLabelArr:  残差 r
        :param origin_trainLabelArr:  原始的标签 y
        :param feature_value_set:
        :return:
        """

        if feature_value_set is None:
            feature_value_set = self.get_feature_value_set(trainDataArr)  # 可供选择的特征集合 , 包括 (特征, 切分值)

        # print('Nums of feature_value_set: {}'.format(len(feature_value_set)))

        self.root = self.__build_tree(trainDataArr, trainLabelArr,origin_trainLabelArr, feature_value_set, tree_depth=0) # 根节点树的高度为0

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

            if row[judge_feature] <= p.feature_split:
                p = p.childs[0]
            else:
                p = p.childs[1]

        return p.label

    def predict(self, testDataArr):
        """
        推理 测试 数据集，返回预测结果

        :param test_data:
        :return:
        """

        res_list = []

        for row in testDataArr:
            res_list.append(self.__predict(row))

        return np.array(res_list)

    def score(self, testDataArr, testLabelArr):
        """
        推理 测试 数据集，返回预测 的 平方误差

        :param test_data:
        :return:
        """
        res_list = self.predict(testDataArr)

        square_loss = np.average(np.square(res_list - testLabelArr))  # 平方误差

        return square_loss


class RegresionTree_XGBoost():
    """

    CART 回归树

    1.适用于 XGBoost


    Author: xrh
    Date: 2021-05-08

    """

    def __init__(self, root=None, gama=0, reg_lambda=1,max_depth=2, min_sample_split=2 ,print_log=True):
        """

        :param root: 树的根节点
        :param gama: 在树的叶节点上进行进一步 划分节点 所需的最小 增益(Gain); 若小于此阈值, 不往下分裂, 形成叶子节点 ; 越大gamma，算法将越保守。
        :param max_depth: 树的最大深度
        :param min_sample_split: 划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
        :param print_log: 是否打印日志
        """

        self.root = root

        self.gama = gama  # 损失的 阈值

        self.reg_lambda=reg_lambda

        self.max_depth = max_depth  # 树的最大深度

        self.min_sample_split=min_sample_split #

        self.print_log=print_log #是否打印日志


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
                feature_value_set.add( (i, A_i) )  #

        return feature_value_set


    def select_max_gain_feature(self,trainDataArr, gArr, hArr, feature_value_set):
        """

        选择 最优的 切分特征 与 切分点

        :param trainDataArr: 样本集合 X
        :param gArr: 每一个样本的 损失函数 对 F 的一阶梯度
        :param hArr: 每一个样本的  损失函数 对 F 的 二阶梯度
        :param feature_value_set: 可用的 (特征, 切分值) 的集合
        :return:
        """

        Ag = None  # 最佳特征
        Ag_split = None  # 最佳特征的　切分点

        max_Gain = float('-inf')

        G=np.sum(gArr)
        H=np.sum(hArr)

        loss_old = ( G**2) /  ( H + self.reg_lambda )  # 切分前的损失

        for A_j,s_i  in feature_value_set:  # 遍历 (特征, 切分值)

            trainDataArr_DevFeature = trainDataArr[:, A_j]

            # s_i 将数据集 划分为两半 R1 和 R2
            R1_g = gArr[trainDataArr_DevFeature <= s_i]  # 特征值 取值为 <= s_i
            R1_h = hArr[trainDataArr_DevFeature <= s_i]

            # R2_g = gArr[trainDataArr_DevFeature > s_i]
            # R2_h = hArr[trainDataArr_DevFeature > s_i]

            GL = np.sum(R1_g)
            HL = np.sum(R1_h)

            GR = G - GL # 只算左边, 右边用总和 减去左边得到, 提升效率
            HR = H - HL

            loss_new =  (GL**2) / ( HL + self.reg_lambda)   +   (GR**2) / ( HR + self.reg_lambda )

            gain = loss_new - loss_old

            if gain >= max_Gain:

                max_Gain = gain
                Ag = A_j
                Ag_split = s_i


        return Ag, Ag_split, max_Gain



    def update_leaf_region_lable(self, gArr,hArr):
        """
        更新 叶子节点的 预测值

        (1) 需要考虑 会触发 X/0= Nan 的bug

        :param gArr: 损失函数 对 F 的一阶梯度
        :param hArr: 损失函数 对 F 的 二阶梯度
        :return:
        """

        numerator = np.sum( gArr )

        if numerator == 0:
            return 0.0

        denominator = np.sum( hArr ) + self.reg_lambda

        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return - numerator / denominator



    def __build_tree(self, trainDataArr, gArr,hArr, feature_value_set, tree_depth, prev_feature=None, prev_max_gain=None ,
                     prev_feature_split=None, father_label=None):
        """
        递归 构建树

        递归结束条件：

        (1) 当前属性集为空

        (2)当前结点包含的样本集合为空，不能划分：将类别设定为父节点的类别。



        :param trainDataArr:
        :param gArr: 每一个样本的 损失函数 对 F 的一阶梯度
        :param hArr: 每一个样本的  损失函数 对 F 的 二阶梯度
        :param feature_value_set:
        :param tree_depth:
        :param prev_feature:
        :param prev_feature_split:
        :param father_label:
        :return:
        """

        T = Node()

        T.prev_feature = prev_feature
        T.prev_feature_split = prev_feature_split

        if np.shape(trainDataArr)[0] == 0 or prev_max_gain <= self.gama:  # 划分后 数据集已经为空

            T.label = father_label  # 说明不能再往下划分了, 使用 上一个 节点给它的标签值

        else:

            T.loss = prev_max_gain

            T.sample_N = np.shape(trainDataArr)[0]

            leaf_label = self.update_leaf_region_lable( gArr,hArr ) # 对于叶子节点区域 ，计算出最佳拟合值

            if len(feature_value_set) == 0 \
                    or np.shape(trainDataArr)[0] <= self.min_sample_split \
                    or tree_depth >= self.max_depth \
                    :

            # len(feature_value_set) == 0 : 所有 切分(特征, 特征值) 的组合 已经用完,
            #  np.shape(trainDataArr)[0] <= self.min_sample_split 划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂
            #  tree_depth >= self.max_depth 树的深度达到最大深度 ,
            #  prev_max_gain <= self.gama 考虑 切分的最大的增益是否比 γ(gama) 大，如果小于γ则不进行分裂（预剪枝）

                T.label = leaf_label #  叶子节点的标签

            else:

                Ag, Ag_split, max_gain = self.select_max_gain_feature(trainDataArr, gArr,hArr,feature_value_set)

                best_feature_value = (Ag, Ag_split)

                # print('Ag:{} , Ag_split:{}, max_gain:{}'.format(Ag,Ag_split,max_gain))

                T.feature = Ag
                T.feature_split = Ag_split

                T.childs = dict()

                trainDataArr_DevFeature = trainDataArr[:, Ag]

                # CART 树为二叉树
                # 左节点为  <=  特征值的 分支
                T.childs[0] = self.__build_tree(trainDataArr[trainDataArr_DevFeature <= Ag_split],
                                                gArr[trainDataArr_DevFeature <= Ag_split],
                                                hArr[trainDataArr_DevFeature <= Ag_split],
                                                feature_value_set - {(best_feature_value)},
                                                tree_depth + 1,
                                                prev_feature=T.feature, prev_max_gain=max_gain,
                                                prev_feature_split='<=' + str(Ag_split), father_label=leaf_label)

                # 右节点为 > 切分特征值的 分支
                T.childs[1] = self.__build_tree(trainDataArr[trainDataArr_DevFeature > Ag_split],
                                                gArr[trainDataArr_DevFeature > Ag_split],
                                                hArr[trainDataArr_DevFeature > Ag_split],
                                                feature_value_set - {(best_feature_value)},
                                                tree_depth + 1,
                                                prev_feature=T.feature, prev_max_gain=max_gain,
                                                prev_feature_split='>' + str(Ag_split), father_label=leaf_label)

        if self.print_log: #

            print('T.feature:{},T.feature_split:{}, T.loss:{} , T.sample_N:{}  '.format(T.feature,T.feature_split, T.loss,T.sample_N))
            print('T.prev_feature:{},T.prev_feature_split:{} '.format(T.prev_feature, T.prev_feature_split))
            print('depth:{} '.format(tree_depth))
            # print('T.childs:{}'.format(T.childs))
            print('T.label:{}'.format(T.label))
            print('-----------')

        return T

    def fit(self, trainDataArr, gArr, hArr, feature_value_set=None):
        """

        :param trainDataArr:  特征 X
        :param gArr: 每一个样本的 损失函数 对 F 的一阶梯度
        :param hArr: 每一个样本的  损失函数 对 F 的 二阶梯度
        :param feature_value_set:
        :return:
        """

        if feature_value_set is None:
            feature_value_set = self.get_feature_value_set(trainDataArr)  # 可供选择的特征集合 , 包括 (特征, 切分值)

        # print('Nums of feature_value_set: {}'.format(len(feature_value_set)))

        self.root = self.__build_tree(trainDataArr, gArr, hArr, feature_value_set,prev_max_gain=float('inf') , tree_depth=0) # 根节点树的高度为0

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

            if row[judge_feature] <= p.feature_split:
                p = p.childs[0]
            else:
                p = p.childs[1]

        return p.label

    def predict(self, testDataArr):
        """
        推理 测试 数据集，返回预测结果

        :param test_data:
        :return:
        """

        res_list = []

        for row in testDataArr:
            res_list.append(self.__predict(row))

        return np.array(res_list)

    def score(self, testDataArr, testLabelArr):
        """
        推理 测试 数据集，返回预测 的 平方误差

        :param test_data:
        :return:
        """
        res_list = self.predict(testDataArr)

        square_loss = np.average(np.square(res_list - testLabelArr))  # 平方误差

        return square_loss

class Test:



    def test_small_dataset(self):
        """
        
        利用《统计学习方法》 表 5.2 数据集 测试 CART 回归树

        :return:
        """

        # 获取训练集

        train_X = np.array( [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] ).T
        y = np.array( [4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00] )


        # 开始时间
        start = time.time()

        # 创建决策树
        print('start create tree')

        RT = RegresionTree(threshold=0.2,max_depth=5)

        RT.fit(train_X,y)

        # print(Regression_tree.select_min_square_loss_feature(train_X, y, {0}))

        print(' tree complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        # 测试数据集
        test_X =  np.array([[4.5, 8.9]]).T
        test_Y=  np.array([5.57, 8.85])


        print('test dataset square loss :', RT.score(test_X, test_Y))


    def test_regress_dataset(self):
        """

        利用 boston房价 数据集
        测试  GBDT  回归

        time span: 32.697904109954834

        by sklearn , the squared_error: 0.03845769641189834
        by xrh , the squared_error: 16.929166666666664

        :return:
        """

        # 加载sklearn自带的波士顿房价数据集
        dataset = load_boston()

        # 提取特征数据和目标数据
        X = dataset.data # shape: (506, 13)
        y = dataset.target

        # 将数据集以9:1的比例随机分为训练集和测试集，为了重现随机分配设置随机种子，即random_state参数
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)

        regr = DecisionTreeRegressor( min_impurity_split=0.2 , max_depth=50 ) # sklearn
        regr.fit( X, y )

        y_pred_1 = regr.predict( X_test )

        start = time.time()
        print('start create model')

        clf = RegresionTree( threshold=0.01 , max_depth=50 )
        clf.fit( X_train , y_train )

        print(' model complete ')
        # 结束时间
        end = time.time()
        print('time span:', end - start)

        y_pred_2= clf.predict( X_test )

        print( 'by sklearn , the squared_error:', mean_squared_error(y_test, y_pred_1)) # 8

        print( 'by xrh , the squared_error:', mean_squared_error(y_test, y_pred_2) ) #



if __name__ == '__main__':

    test=Test()

    # test.test_small_dataset()

    test.test_regress_dataset()


