#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import os
import h5py
import numpy as np


class MicrosoftCocoDataset:
    """
    包装了 MicrosoftCoco 数据集,  我们通过此类来访问该数据集

    1.初始化完毕后, 结果放入 self.dataset (字典), 数据集的各个部分如下:

    训练集(train)
    key               shape        value
    train_captions    (400135, 17) 图片评论
    train_image_idxs  (400135,)    评论到图片的映射, 通过映射找到评论对应的图片向量
    train_features    (82783, 512) 向量化后的图片, 维度为 512
    train_urls        (82783,)     图片的 url 地址

    验证集(val)
    val_captions      (195954, 17)
    val_image_idxs    (195954,)
    val_features      (40504, 512)
    val_urls          (40504,)

    idx_to_word        1004        单词标号到单词的词典
    word_to_idx        1004        单词到单词标号的词典

    Author: xrh
    Date: 2021-9-30

    """

    def __init__(self, base_dir='../../dataset/ImageCaption/microsoft_coco', sample_N=None, use_pca_features=True):
        """

        :param base_dir: 数据集集的根路径
        :param sample_N: 训练数据集(采样后)的样本数
        :param use_pca_features: 是否使用 PCA 降维后的图像特征

        """

        self.base_dir = base_dir
        self.sample_N = sample_N
        self.use_pca_features = use_pca_features

        self.dataset = {}

        self.__load_data()  # 读取预处理后的数据集

        self.vocab_obj = Vocab(id_to_word=self.dataset['idx_to_word'], word_to_id=self.dataset['word_to_idx'])

        if sample_N is not None:

            # 对训练数据集进行采样
            np.random.seed(231)  # 控制随机数, 在程序的当前上下文有效

            N_train = np.shape(self.dataset['train_captions'])[0]  # 训练集的样本总数
            mask = np.random.randint(N_train, size=self.sample_N)
            self.dataset['train_captions'] = self.dataset['train_captions'][mask]
            self.dataset['train_image_idxs'] = self.dataset['train_image_idxs'][mask]

            # 对验证(测试)数据集进行采样, 采样个数为 训练集的 1/2

            N_val = np.shape(self.dataset['train_captions'])[0]  # 训练集的样本总数
            mask = np.random.randint(N_val, size=self.sample_N//2)
            self.dataset['val_captions'] = self.dataset['val_captions'][mask]
            self.dataset['val_image_idxs'] = self.dataset['val_image_idxs'][mask]

            self.N = sample_N

        else:
            self.N = np.shape(self.dataset['train_captions'])[0]

        self.feature_dim = self.dataset['train_features'].shape[1]  # feature_dim - 图片向量的维度
        self.caption_length = self.dataset['train_captions'].shape[1]  # caption_length - 图片描述的长度


    def __load_data(self):
        """
        载入数据集

        :return:
        """
        caption_file = os.path.join(self.base_dir, 'coco2014_captions.h5')
        with h5py.File(caption_file, 'r') as f:
            for k, v in f.items():
                self.dataset[k] = np.asarray(v)

        if self.use_pca_features:
            train_feat_file = os.path.join(self.base_dir, 'train2014_vgg16_fc7_pca.h5')
        else:
            train_feat_file = os.path.join(self.base_dir, 'train2014_vgg16_fc7.h5')
        with h5py.File(train_feat_file, 'r') as f:
            self.dataset['train_features'] = np.asarray(f['features'])

        if self.use_pca_features:
            val_feat_file = os.path.join(self.base_dir, 'val2014_vgg16_fc7_pca.h5')
        else:
            val_feat_file = os.path.join(self.base_dir, 'val2014_vgg16_fc7.h5')
        with h5py.File(val_feat_file, 'r') as f:
            self.dataset['val_features'] = np.asarray(f['features'])

        dict_file = os.path.join(self.base_dir, 'coco2014_vocab.json')
        with open(dict_file, 'r') as f:
            dict_data = json.load(f)
            for k, v in dict_data.items():
                self.dataset[k] = v

        train_url_file = os.path.join(self.base_dir, 'train2014_urls.txt')
        with open(train_url_file, 'r') as f:
            train_urls = np.asarray([line.strip() for line in f])
        self.dataset['train_urls'] = train_urls

        val_url_file = os.path.join(self.base_dir, 'val2014_urls.txt')
        with open(val_url_file, 'r') as f:
            val_urls = np.asarray([line.strip() for line in f])
        self.dataset['val_urls'] = val_urls


    def decode_captions(self, captions):
        """
        将由单词标号组成的句子 解码为 原始句子

        :param captions: 一个句子, 或者一个句子列表
        :return:
        """

        singleton = False

        if captions.ndim == 1:  # 说明 captions 只有一个句子
            singleton = True
            captions = captions[None]

        decoded = []
        N, T = captions.shape

        for i in range(N):
            words = []
            for t in range(T):
                word = self.dataset['idx_to_word'][captions[i, t]]
                if word != '<NULL>':
                    words.append(word)
                if word == '<END>':
                    break
            decoded.append(' '.join(words))

        if singleton:
            decoded = decoded[0]

        return decoded



    def sample_minibatch(self, batch_size=128, Type='train', return_url=False):
        """
        从数据集中采样 1个 batch 的样本用于训练

        :param batch_size:  1个 batch的样本个数
        :param Type: 数据集的类型, 'train' 为训练数据集
        :param return_url: 是否返回图片的 url
        :return:

        captions, image_features

        """

        split_size = self.dataset['%s_captions' % Type].shape[0]
        mask = np.random.choice(split_size, batch_size)  # 第二次调用此函数时, 类初始化(def __init__) 中的控制随机数就失效了

        captions = self.dataset['%s_captions' % Type][mask]
        image_idxs = self.dataset['%s_image_idxs' % Type][mask]

        image_features = self.dataset['%s_features' % Type][image_idxs]

        urls = self.dataset['%s_urls' % Type][image_idxs]

        if return_url:

            return captions, image_features, urls

        return captions, image_features


class Vocab:

    def __init__(self, word_to_id, id_to_word, _unk_str='<UNK>'):

        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

        self._unk_str = _unk_str


    def map_id_to_word(self, id):
        """
        输入单词标号, 返回单词

        :param id:
        :return:
        """

        return self.id_to_word[id]

    def map_word_to_id(self, word):
        """
        输入单词, 返回单词标号

        考虑未登录词:
        1.若输入的单词不在词典中, 返回 '<UNK>' 的标号

        :param word: 单词
        :return:
        """

        if word not in self.word_to_id:
            return self.word_to_id[self._unk_str]
        else:
            return self.word_to_id[word]


