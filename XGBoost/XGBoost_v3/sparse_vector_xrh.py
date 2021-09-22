#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

class DMatrix:

    def __init__(self, data_arr, missing={np.nan, 0}):
        """

        :param data_arr: 样本特征 (不含标签)
        :param missing: 缺失值的集合, 若特征值在此集合中, 则认为其为缺失值

        """

        # N 样本总个数( 包含缺出现缺失值的样本 )
        # m 特征的总数
        self.N, self.m = np.shape(data_arr)

        # row_index 样本行的索引
        self.row_index = list(range(self.N))

        # 样本行
        self.row_data = data_arr

        # 所有特征对应的块集合
        self.sorted_pages = []

        # 不同特征中出现过特征缺失值的行的集合
        # self.missing_value_pages = []

        for i in range(self.m):  # 遍历所有的特征

            feature = data_arr[:, i]  # 特征 i 拎出来 shape:(N,)
            feature_index = []

            missing_value_index = []

            for rid in range(self.N):

                if feature[rid] not in missing:  # 特征值 不在 缺失值集合中
                    feature_index.append((feature[rid], rid))  # (特征值, 样本标号)
                # else:
                #     missing_value_index.append(rid)

            # 按照特征值的大小排序
            sorted_feature_index = sorted(feature_index, key=lambda t: t[0])

            self.sorted_pages.append(sorted_feature_index)
            # self.missing_value_pages.append(missing_value_index)