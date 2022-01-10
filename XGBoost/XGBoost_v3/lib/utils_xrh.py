#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np


class Utils:

    @staticmethod
    def convert_to_one_hot(x, class_num, dtype=np.int32):
        """
        将标签值转为 one-hot 向量

        :param x:
        :param class_num: 标签类别的数量
        :param dtype:  取决于标签类别的数量, 一般用 int32 装得下
        :return: one_hot_vec shape:

        """
        x_shape = np.shape(x)

        idx = x.reshape(-1).astype(dtype)

        one_hot_flat = np.eye(class_num)[idx]  # shape (N,class_num) , class_num - 类别个数  N - 样本个数

        one_hot_vec = one_hot_flat.reshape(x_shape + (-1,))

        return one_hot_vec


class Test:

    def test_convert_to_one_hot(self):
        x1 = np.array([1, 2, 3, 4])
        class_num = 5
        print('x1 one-hot :', Utils.convert_to_one_hot(x1, class_num))  # shape: (N,class_num)

        N, T, class_num = 5, 3, 4
        x2 = np.zeros((N, T)).astype(np.int8)
        x2[0][1] = 1
        x2[1][0] = 2
        x2[2][2] = 3
        x2[3][0] = 1
        x2[3][1] = 2
        x2[4][2] = 3

        print('x2:', x2)

        x2_one_hot = Utils.convert_to_one_hot(x2, class_num)

        print('x2 one-hot :', x2_one_hot)

        print('x2 shape:', np.shape(x2_one_hot))


if __name__ == '__main__':
    test = Test()

    test.test_convert_to_one_hot()
