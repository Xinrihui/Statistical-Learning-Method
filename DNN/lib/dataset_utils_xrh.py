#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np


def sample_dataset_minibatch(batch_size=64, **kwargs):
    """
    从数据集中 采样出一个 batch 的训练数据

    :param batch_size:
    :param kwargs:
    :return:
    """

    feature_batch = None
    label_batch = None

    if len(kwargs) == 1:  # 1个 k-v 的参数

        dataset = kwargs.pop('dataset', None)

        N = np.shape(dataset['feature'])[0]  # 数据集中的样本个数

        mask = np.random.choice(N, batch_size)

        feature_batch = dataset['feature'][mask]
        label_batch = dataset['label'][mask]

    elif len(kwargs) == 2:  # 2个 k-v 的参数

        feature = kwargs.pop('feature', None)
        label = kwargs.pop('label', None)

        N = np.shape(feature)[0]  # 数据集中的样本个数

        mask = np.random.choice(N, batch_size)

        feature_batch = feature[mask]
        label_batch = label[mask]

    return  feature_batch, label_batch