#!/usr/bin/python
# -*- coding: UTF-8 -*-


class NodeEntry:
    """
    节点的统计信息

    """

    def __init__(self):

        # 梯度统计信息(G, H)
        self.stats = GradeStats()

        # 左子节点梯度统计信息(GL, HL)
        self.left_stats = GradeStats()

        # 右子节点梯度统计信息(GR, HR)
        self.right_stats = GradeStats()

        # TODO: 最后扫描到的特征值
        # self.last_value = 0.0

        # 节点没有分裂时的目标函数值
        self.root_loss = 0.0

        # 当前节点的权重 (score)
        self.weight = 0.0

        # 最优的分裂方案
        self.best = SplitEntry()


class GradeStats:
    """
    梯度 统计信息

    """

    def __init__(self):
        # 一阶梯度的和
        self.sum_grad = 0

        # 二阶梯度的和
        self.sum_hess = 0


# 出现缺失值时的默认方向
direction_left = 0  # 左边
direction_right = 1  # 右边


class SplitEntry:

    def __init__(self):

        # 节点分裂的最大增益
        self.max_gain = float('-inf')

        # 最佳分裂特征的标号
        self.feature = None

        # 最佳分裂特征的最优切分点 在block 中的行偏移量
        # self.split_offset = None

        # 最佳分裂特征出现缺失值时的默认方向
        self.split_default = None

        # 最佳分裂特征的最优切分点的特征值
        self.split_value = None

        # 左子节点梯度统计信息(GL, HL)
        self.split_left_stats = GradeStats()

        # 右子节点梯度统计信息(GR, HR)
        self.split_right_stats = GradeStats()
