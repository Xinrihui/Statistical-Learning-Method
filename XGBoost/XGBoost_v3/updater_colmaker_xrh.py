#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from tree_model_xrh import *
from split_evaluator_xrh import *
from param_xrh import *

from collections import deque

from deprecated import deprecated


class Builder:
    """
    建立 CART回归树

    1.采用 宽度优先搜索(BFS) 建树 (非递归)
    2.寻找切分点时采用 精确贪心算法, 并实现了 稀疏感知

    Author: xrh
    Date: 2021-05-29

    """

    def __init__(self,
                 gama=0,
                 reg_alpha=0,
                 reg_lambda=1,
                 max_depth=2,
                 min_sample_split=2,
                 max_delta_step=0,
                 min_child_weight=0,
                 tree_method='exact',
                 sketch_eps=0.3,
                 missing={np.nan, 0},
                 print_log=True):

        """

        :param gama: 划分子节点 所需的最小 增益(Gain); 若小于此阈值, 不往下分裂形成子节点 ; 设置越大的 gamma，算法将越保守。

        :param reg_alpha： 一个浮点数，是L1 正则化系数。它是xgb 的alpha 参数
        :param reg_lambda： 一个浮点数，是L2 正则化系数。它是xgb 的lambda 参数

        :param max_depth: 树的最大深度

        :param min_sample_split: 划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2

        :param max_delta_step： 每棵树的叶子节点的权重估计时的最大 delta step。取值范围为 [0, ) ，0 表示没有限制，默认值为 0 。

        :param min_child_weight: 搜索最佳切分点时, 若 min_child_weight < Min(HL, HR) 则放弃此切分点

        :param tree_method： 指定了构建树的算法，可以为下列的值：
                            (1)'exact'： 使用 exact greedy 算法分裂节点
                            (2)'approx'： 使用近似算法分裂节点

        :param sketch_eps： 指定了分桶的步长。其取值范围为 (0,1)， 默认值为 0.3 。
                            它仅仅用于 tree_medhod='approx'。


        :param print_log: 是否打印日志
        """

        self.params = {}

        self.params['gama'] = gama  # 损失的 阈值

        self.params['reg_alpha'] = reg_alpha
        self.params['reg_lambda'] = reg_lambda

        self.params['max_depth'] = max_depth  # 树的最大深度

        self.params['min_sample_split'] = min_sample_split

        self.params['max_delta_step'] = max_delta_step

        self.params['min_child_weight'] = min_child_weight

        self.params['tree_method'] = tree_method

        self.params['sketch_eps'] = sketch_eps

        self.params['missing'] = missing

        self.params['print_log'] = print_log  # 是否打印日志

        # 记录样本属于哪个节点
        self.position = None

        # 待分裂节点的列表
        self.qexpand = None

        # 所有节点的列表
        self.tree = None

        # 树的根节点
        self.root = None

        # 初始化自增id生成器 #TODO: 生成器无法被 序列化 (pickle)
        # self.incre_object = IncreIDGenerator(0)

        self.start_node_id = -1

    def init_new_node(self, fmat, gArr, hArr):
        """
        初始化新节点

        :param fmat: 样本特征, 使用 DMatrix 包装
        :param gArr: 一阶梯度
        :param hArr: 二阶梯度
        :return:
        """

        ndata = fmat.N  # 训练集中的总样本数目

        # 统计 节点的 G 和 L
        for rid in range(ndata):  # rid - 训练数据的行标号

            if self.position[rid] < 0:  # 该样本对应的节点已经是叶子节点了
                continue

            self.tree[self.position[rid]].entry.stats.sum_grad += gArr[rid]  # G
            self.tree[self.position[rid]].entry.stats.sum_hess += hArr[rid]  # H

        self.update_loss_and_weight()

    def update_loss_and_weight(self):
        """
        计算 qexpand 队列中所有候选节点的损失函数和权重

        :return:
        """

        # 按照论文公式计算 待分裂队列中的所有节点 的 权重 与 损失(目标)函数的值
        for nid in self.qexpand:  # nid - 节点的标号

            self.tree[nid].entry.weight = TreeEvaluator.CalcWeight(self.params,
                                                                   self.tree[nid].entry.stats)  # 拆分(split)前 节点的权重

            self.tree[nid].entry.root_loss = TreeEvaluator.CalcLoss(self.params, self.tree[nid].entry.stats)
            # 拆分(split)前节点的损失(目标)函数的值

    def find_split(self, gArr, hArr, p_fmat):
        """
        逐层分裂，找到 expand节点 的分裂方案

        :param gArr: 一阶梯度
        :param hArr: 二阶梯度
        :param p_fmat: 样本特征, 使用 DMatrix 包装
        :return:
        """

        feature_set = list(range(p_fmat.m))  # TODO: 随机采样特征

        # TODO: 在一个 block 中分批取训练数据

        for i in feature_set:  # 遍历所有的特征

            block_i = p_fmat.sorted_pages[i]  # 特征 i 对应的block, block 中出现缺失值的样本以及被排除了
            L = len(block_i)

            # 实现稀疏感知

            # 1.累加 GL,HL 含缺失值的样本被计入了 GR,HR
            for j in range(L):  # 顺序遍历该特征下的所有特征值(已排序), 每一行为 (f_value, rid),
                # f_value - 特征值, rid - 训练数据的行标号

                f_value = block_i[j][0]
                rid = block_i[j][1]

                nid = self.position[rid]
                if nid < 0:  # 说明该样本已经属于一个不能分裂的叶子节点了, 不再考虑
                    continue

                c_node = self.tree[nid]

                # 滚动数组累加
                c_node.entry.left_stats.sum_grad += gArr[rid]  # GL
                c_node.entry.left_stats.sum_hess += hArr[rid]  # HL

                # GR = G - GL
                c_node.entry.right_stats.sum_grad = c_node.entry.stats.sum_grad - c_node.entry.left_stats.sum_grad
                # HR = H - HL
                c_node.entry.right_stats.sum_hess = c_node.entry.stats.sum_hess - c_node.entry.left_stats.sum_hess

                gain = TreeEvaluator.CalcGain(self.params, c_node.entry.root_loss, c_node.entry.left_stats,
                                              c_node.entry.right_stats)

                if gain > c_node.entry.best.max_gain:
                    c_node.entry.best.max_gain = gain
                    c_node.entry.best.feature = i
                    c_node.entry.best.split_default = direction_right
                    c_node.entry.best.split_value = f_value

                    # 左子节点梯度统计信息(GL, HL)
                    c_node.entry.best.split_left_stats = c_node.entry.left_stats
                    # 右子节点梯度统计信息(GR, HR)
                    c_node.entry.best.split_right_stats = c_node.entry.right_stats

            # 节点的 GL, HL, GR, HR 清零
            self.init_nodes_LRstats()

            # 2.累加 GR,HR 含缺失值的样本被计入了 GL,HL
            for j in range(L - 1, -1, -1):  # 逆序遍历该特征下的所有特征值(已排序), 每一行为 (f_value, rid),
                # f_value - 特征值, rid - 训练数据的行标号

                f_value = block_i[j][0]
                rid = block_i[j][1]

                nid = self.position[rid]
                if nid < 0:  # 说明该样本已经属于一个不能分裂的叶子节点了, 不再考虑
                    continue
                c_node = self.tree[nid]

                # 滚动数组累加
                c_node.entry.right_stats.sum_grad += gArr[rid]  # GR
                c_node.entry.right_stats.sum_hess += hArr[rid]  # HR

                # GL = G - GR
                c_node.entry.left_stats.sum_grad = c_node.entry.stats.sum_grad - c_node.entry.right_stats.sum_grad
                # HL = H - HR
                c_node.entry.left_stats.sum_hess = c_node.entry.stats.sum_hess - c_node.entry.right_stats.sum_hess

                gain = TreeEvaluator.CalcGain(self.params, c_node.entry.root_loss, c_node.entry.left_stats,
                                              c_node.entry.right_stats)

                if gain > c_node.entry.best.max_gain:

                    c_node.entry.best.max_gain = gain
                    c_node.entry.best.feature = i
                    c_node.entry.best.split_default = direction_left
                    c_node.entry.best.split_value = f_value - 1e-6  #  逆序遍历, 左半部分没有取到 f_value边界, 而右半部分取到了f_value边界,
                                                                    #  并且之后划分节点时取 '<=', 所以减掉一个很小的数(1e-6)
                                                                    # TODO: 一个更优雅的方法: 最后扫描到的特征值 last_fvalue

                    # 左子节点梯度统计信息(GL, HL)
                    c_node.entry.best.split_left_stats = c_node.entry.left_stats
                    # 右子节点梯度统计信息(GR, HR)
                    c_node.entry.best.split_right_stats = c_node.entry.right_stats

            # 节点的 GL, HL, GR, HR 清零
            self.init_nodes_LRstats()

    def init_nodes_LRstats(self):
        """
        对所有待分裂节点的 GL, HL, GR, HR 清零

        :return:
        """

        for nid in self.qexpand:  # 所有待分裂节点

            node = self.tree[nid]
            node.entry.left_stats = GradeStats()
            node.entry.right_stats = GradeStats()

    @deprecated()
    def split_nodes_deprecated(self, p_fmat):
        """
        根据最优切分点划分所有待分裂节点

        :param p_fmat: 样本特征, 使用 DMatrix 包装

        :return:
        """
        l = len(self.qexpand)

        for _ in range(l):  # 宽度受限的BFS

            nid = self.qexpand.pop()  # nid-父节点标号
            node = self.tree[nid]

            best_split = node.entry.best  # 最佳切分方案

            if best_split.max_gain <= self.params['gama']:  # 划分子节点 所需的最小 增益(Gain); 若小于此阈值, 不往下分裂

                continue

            # 生成左右子节点
            # left_child_id = next(self.incre_object.id)  # 1
            self.start_node_id += 1
            left_child_id = self.start_node_id
            left_child_node = Node(id=left_child_id)

            # right_child_id = next(self.incre_object.id)  # 2
            self.start_node_id += 1
            right_child_id = self.start_node_id
            right_child_node = Node(id=right_child_id)

            left_child_node.entry.stats = best_split.split_left_stats  # 左子节点的 G,H
            right_child_node.entry.stats = best_split.split_right_stats  # 右子节点的 G,H

            if self.params['print_log']:  # 打印当前子树

                print('sub_tree:')
                print('root:{}, root_loss:{}, root_weight:{}'.format(nid, node.entry.root_loss, node.entry.weight))
                print('best_feature:{}, split_value:{}, default_direction:{}'.format(best_split.feature,
                                                                                     best_split.split_value,
                                                                                     best_split.split_default))

            # 左右子节点挂在 node 下
            node.left = left_child_node
            node.right = right_child_node

            block_f = p_fmat.sorted_pages[best_split.feature]  # 切分特征对应的块

            block_f_missing = p_fmat.missing_value_pages[best_split.feature]

            offset = best_split.split_offset
            if best_split.split_default == direction_left:  # 逆序导致需要修正偏移量
                offset -= 1

            # 更新属于左子节点的样本的归属(position)
            for _, rid in block_f[:offset + 1]:

                if self.position[rid] == nid:  # 样本原来的归属是父节点
                    self.position[rid] = left_child_node.id

            # 更新属于右子节点的样本的归属(position)
            for _, rid in block_f[offset + 1:]:

                if self.position[rid] == nid:  # 样本原来的归属是父节点
                    self.position[rid] = right_child_node.id

            # 更新存在缺失值的样本 的归属 # TODO: 此步时间复杂度过高, 是否可以不管 存在缺失值的样本
            if best_split.split_default == direction_right:

                for rid in block_f_missing:

                    if self.position[rid] == nid:  # 样本原来的归属是父节点
                        self.position[rid] = right_child_node.id
            else:
                for rid in block_f_missing:

                    if self.position[rid] == nid:  # 样本原来的归属是父节点
                        self.position[rid] = left_child_node.id

            # 左右子节点入 待分裂队列 qexpand
            self.qexpand.appendleft(left_child_node.id)
            self.qexpand.appendleft(right_child_node.id)

            # 左右子节点加入 tree
            self.tree.append(left_child_node)
            self.tree.append(right_child_node)

    def split_nodes(self, p_fmat):
        """
        根据最优切分点划分所有待分裂节点

        :param p_fmat: 样本特征, 使用 DMatrix 包装

        :return:
        """

        # 1.在树上分裂父节点
        for nid in self.qexpand:
            # nid-父节点标号

            node = self.tree[nid]
            best_split = node.entry.best  # 最佳切分方案

            if best_split.max_gain <= self.params['gama']:  # 划分子节点 所需的最小 增益(Gain); 若小于此阈值, 不往下分裂

                # 父节点不能分裂, 设置该父节点状态为不能更新, 但是它还是叶子节点
                # 稍后要对属于父节点的样本进行标记(设置 position<0), 这些样本以后就不考虑了
                node.can_update = False
                # node.is_leaf = True

            else:

                # 生成左右子节点
                # left_child_id = next(self.incre_object.id)  # 1
                self.start_node_id += 1
                left_child_id = self.start_node_id
                left_child_node = Node(id=left_child_id)  # 新生成的节点初始化为: 叶子节点(isleaf = True)
                # 节点状态为可以分裂 ( can_update=True )

                # right_child_id = next(self.incre_object.id)  # 2
                self.start_node_id += 1
                right_child_id = self.start_node_id
                right_child_node = Node(id=right_child_id)

                left_child_node.entry.stats = best_split.split_left_stats  # 左子节点的 G,H
                right_child_node.entry.stats = best_split.split_right_stats  # 右子节点的 G,H

                # 左右子节点挂在父节点下
                node.left = left_child_node
                node.right = right_child_node
                # 更新父节点为 非叶子节点, 显然它有子节点了
                node.is_leaf = False

                self.tree.append(left_child_node)
                self.tree.append(right_child_node)

            if self.params['print_log']:  # 打印当前子树

                print('')
                print('root_id:{}, root_loss:{}, root_weight:{}'.format(nid, node.entry.root_loss, node.entry.weight))
                print('is_leaf:{} , can_update:{}'.format(node.is_leaf, node.can_update))
                print('best_feature:{}, split_value:{}, default_direction:{}, max_gain:{}'.format(best_split.feature,
                                                                                     best_split.split_value,
                                                                                     best_split.split_default, best_split.max_gain))
        # 2.更新样本到节点的映射关系(position)
        self.update_position(p_fmat)

    def update_position(self, p_fmat):
        """
        根据分裂结果，将数据重新映射到子节点 , 即更新数据样本到树节点的映射关系

        :param p_fmat: 样本特征, 使用 DMatrix 包装

        :return:
        """
        # 更新 特征值非空的样本的映射
        self.update_none_missing_position(p_fmat)

        ndata = p_fmat.N  # 训练集中的总样本数目( 包含特征值为空的样本 )
        for rid in range(ndata):  # rid - 训练数据的行标号

            nid = self.decode_position(rid)  # 节点标号

            if self.tree[nid].is_leaf:  # 节点是叶子节点
                # 特征值非空的样本都被挂到了新生成的子节点(is_leaf=True, can_update=True)下

                if not self.tree[nid].can_update:  # 节点状态为不能分裂
                    self.position[rid] = ~nid  # 对节点标号按位取反, 变为负数

            else:  # 节点不是叶子节点
                # 特征值为空的样本
                if self.tree[nid].entry.best.split_default == direction_left:
                    self.encode_position(rid, self.tree[nid].left.id)

                else:
                    self.encode_position(rid, self.tree[nid].right.id)

    def update_none_missing_position(self, p_fmat):
        """
        更新 特征值非空的样本的映射

        :param p_fmat: 样本特征, 使用 DMatrix 包装
        :return:
        """
        feature_split = set()  # 待分裂的特征集合

        for nid in self.qexpand:

            # nid-父节点标号
            node = self.tree[nid]

            if node.can_update:  # 若状态为 能分裂(更新)

                feature_split.add(node.entry.best.feature)  # 最佳切分特征

        # feature_split = sorted(feature_split)

        for fid in feature_split:  # 遍历切分特征, fid-特征标号

            block_f = p_fmat.sorted_pages[fid]  # 切分特征对应的块, sorted_pages 中都是特征值非空的样本
            L = len(block_f)

            for j in range(L):  # 顺序遍历该特征下的所有特征值(已排序), 每一行为 (f_value, rid)
                # f_value - 特征值, rid - 样本标号

                f_value = block_f[j][0]
                rid = block_f[j][1]

                nid = self.decode_position(rid)  # 待分裂节点(父节点)

                if not self.tree[nid].is_leaf and self.tree[nid].entry.best.feature == fid:  # 若 不是叶子节点 才进行分裂

                    if f_value <= self.tree[nid].entry.best.split_value:  # 此处取 '<='
                        self.encode_position(rid, self.tree[nid].left.id)  # 样本更新为属于左子节点(左子节点此时为叶子节点)
                    else:
                        self.encode_position(rid, self.tree[nid].right.id)  # 样本更新为属于右子节点

    def decode_position(self, rid):
        """
        返回样本属于哪个节点

        :param rid: 样本标号
        :return: nid 节点标号

        关于按位取反(~)

        eg.
        a = 2
        ~a = -3
        ~(~a) = 2

        """
        nid = self.position[rid]

        if nid < 0:  # 说明该样本对应的节点标号 上一次被按位取反了,
            # 再做一次按位取反, 可以得到节点标号
            nid = ~nid

        return nid

    def encode_position(self, rid, nid):
        """
        设置 样本到节点的映射

        :param rid:
        :param nid:
        :return:
        """

        if self.position[rid] < 0:

            self.position[rid] = ~nid

        else:
            self.position[rid] = nid

    def update_qexpand(self,):
        """
        对于 qexpand 中非叶子节点分裂出的左、右节点加入新的队列

        :return:
        """

        new_qexpand = []

        for nid in self.qexpand:
            # nid-父节点标号

            node = self.tree[nid]

            if not node.is_leaf:

                new_qexpand.append(node.left.id)
                new_qexpand.append(node.right.id)

        return new_qexpand


    def fit(self, p_fmat, gArr, hArr):
        """
        训练一颗 cart 回归树

        :param p_fmat:  样本特征 X , 使用 DMatrix 表示
        :param gArr: 每一个样本的 损失函数 对 F 的一阶梯度
        :param hArr: 每一个样本的  损失函数 对 F 的 二阶梯度
        :return:
        """

        # root_id = next(self.incre_object.id)
        self.start_node_id += 1
        root_id = self.start_node_id

        self.root = Node(id=root_id)  # 根节点 id=0

        # 所有节点的列表
        self.tree = [self.root]

        # 样本到节点的映射关系, 记录样本属于哪个节点
        # eg. position[0]=1 样本0 目前位于 节点1 中
        self.position = [root_id] * (p_fmat.N)  # 初始状况, 所有样本都属于 根节点(节点标号为0)

        # 待分裂节点的列表
        self.qexpand = [root_id]  # 初始化: 只有根节点(节点标号为0)

        # 1. 初始化
        self.init_new_node(p_fmat, gArr, hArr)

        # 树的生长
        # 根据参数 param.max_depth(树的最大深度) 逐层分裂生成节点
        for depth in range(self.params["max_depth"]):
            # 2.搜索最优切分点
            self.find_split(gArr, hArr, p_fmat)

            # 3.根据最优切分点划分所有待分裂节点
            self.split_nodes(p_fmat)

            # 4.将待分割的叶子结点用于替换 qexpand，作为下一轮split的候选节点
            self.qexpand = self.update_qexpand()

            # 4. 计算 qexpand 队列中所有节点的损失函数和权重
            self.update_loss_and_weight()

    def __inference(self, row):
        """
        推理一个样本

        :param row: 一行特征
        :return:
        """

        p = self.root

        while not p.is_leaf:  # 到达 叶子节点 退出循环

            judge_feature = p.entry.best.feature  # 当前节点分裂的特征

            if row[judge_feature] not in self.params['missing']:  # 不是缺失值
                if row[judge_feature] <= p.entry.best.split_value:  # 应该取等号
                    p = p.left
                else:
                    p = p.right

            else:  # 是缺失值, 走默认的方向
                if p.entry.best.split_default == direction_left:
                    p = p.left

                else:
                    p = p.right

        return p.entry.weight

    def inference(self, X):
        """
        利用训练好的 Cart 树进行推理

        :param X: 样本特征 X , shape (N_sample, N_feature)
        :return:
        """

        res_list = []

        for row in X:
            res_list.append(self.__inference(row))

        return np.array(res_list)


class Test:
    pass


if __name__ == '__main__':
    test = Test()
