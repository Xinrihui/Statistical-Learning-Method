#!/usr/bin/python
# -*- coding: UTF-8 -*-

from param_xrh import *


class IncreIDGenerator:
    """
    自增 id 生成器

    eg.
    ob = IncreIDGenerator(0)

    next(ob.id) # 0
    next(ob.id) # 1

    """
    def __init__(self, start=0):
        """

        :param start: id开始值
        """

        self.id = self.incre(start)

    def incre(self, n):
        """
        使用生成器实现

        :return:
        """
        while True:
            yield n
            n += 1


# 树节点
class Node:

    def __init__(self, id=0):

        self.id = id  # 树节点的标号
        self.is_leaf = True  # 是否是叶子节点
        self.can_update = True  # 该叶子节点是否可以分裂

        self.entry = NodeEntry()

        self.left = None
        self.right = None
