#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from build_dataset_xrh import *


class FlickerDataset:
    """
    包装了 Flicker 数据集,  我们通过此类来访问该数据集

    1.使用之前先对数据集进行预处理, 详见 build_dataset_xrh.py, 预处理后的数据集在 /cache_data 目录下

    Author: xrh
    Date: 2021-9-30

    """

    def __init__(self, base_dir='../../dataset/ImageCaption/'):
        caption_file_dir = os.path.join(base_dir, 'Flicker8k/Flickr8k.token.txt')
        image_folder_dir = os.path.join(base_dir, 'Flicker8k/Flicker8k_Dataset/')

        dataset_dir = os.path.join(base_dir, 'cache_data/train_dataset.json')
        image_caption_dict_dir = os.path.join(base_dir, 'cache_data/image_caption_dict.bin')

        self.data_process = DataPreprocess(
            caption_file_dir=caption_file_dir,
            image_folder_dir=image_folder_dir,
            dataset_dir=dataset_dir,
            image_caption_dict_dir=image_caption_dict_dir
        )

        vocab_path = os.path.join(base_dir, 'cache_data/vocab.bin')

        self.vocab_obj = BuildVocab(load_vocab_dict=True, vocab_path=vocab_path)

        dataset_dir = os.path.join(base_dir, 'cache_data/train_dataset.json')
        self.dataset = pd.read_json(dataset_dir)

        self.image_feature = np.array(self.dataset['image_feature'].tolist())
        self.caption_encoding = np.array(self.dataset['caption_encoding'].tolist())

        self.N = len(self.caption_encoding)  #  N - 样本总数
        self.feature_dim = self.image_feature.shape[1]  # feature_dim - 图片向量的维度
        self.caption_length = self.caption_encoding.shape[1]  # caption_length - 图片描述的长度


    def sample_minibatch(self, batch_size=128, max_length=30):
        """
        从数据集中采样 1个 batch 的样本用于训练

        :param batch_size:  1个 batch的样本个数

        :return:
        """

        mask = np.random.choice(self.N, batch_size)  # 从 range(m) 中随机采样batch_size 组成list

        batch_image_feature = self.image_feature[mask]
        batch_caption_encoding = self.caption_encoding[mask]

        batch_caption_encoding = batch_caption_encoding[:, :max_length]

        return batch_caption_encoding, batch_image_feature
