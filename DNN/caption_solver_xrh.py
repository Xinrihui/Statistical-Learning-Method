#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from lib import Optimizer_xrh as optim

from lib.lstm_caption_xrh import *
from lib.rnn_caption_xrh import *

from lib.coco_utils import *
from lib.image_utils import *

class ImageCaptionSolver:
    """

    对图片做文字描述 (image caption) 的模型的包装器

    示例代码:

    data = load_coco_data()
    model = RNNModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    其中模型对象必须实现以下的方法(API):

    - model.params 是一个字典, 包含模型所有需要更新的参数

    - model.fit_batch(images_feature, captions) 计算模型的损失函数和所有参数的梯度, 它的输入输出为:

      Inputs:
      - images_feature: 一次抽取特征后的向量化的图片 shape (N, n_p)  N-样本个数 n_p-图片向量维度
      - batch_sentence: 一批句子 shape(N,T) ,句子由单词的标号构成 N-样本个数 T-句子长度

      Returns:
      - loss: 损失函数的值
      - grads: 是一个字典, 包含了模型所有需要更新的参数的梯度, 它里面的元素和 model.params 中的一一对应

    - model.inference_sample(images_feature, caption_length) 利用训练好的模型进行推理

    Author: xrh
    Date: 2021-08-01

    """

    def __init__(self, model, dataset,
                 model_path='model/lstm_caption.model',
                 **kwargs):
        """

        必要参数:
        - model: RNN 模型对象
        - dataset: 从 load_coco_data 中获得的训练集和验证集数据
        - model_path: 预训练模型的路径

        可选参数:
        - optimize_mode: 可选的优化算法
                      'BGD' : 批量梯度下降 BGD
                      'MinBatch': min-Batch梯度下降 (默认)
                      'Momentum': 带动量的 Mini-batch 梯度下降
                      'Adam': Adam Mini-batch 梯度下降

        - optim_config: 字典类型, 优化算法的超参数 (详见 Optimizer_xrh.py )
                      {
                     'learning_rate': 5e-3,
                      }
        - batch_size: 选择 min-Batch梯度下降时, 每一次输入模型的样本个数 (默认 = 64)
        - num_epochs: 模型训练的 epoch 个数,  一般训练集所有的样本模型都见过一遍才算一个 epoch
        - print_log: 是否打印日志
        - print_every: 打印训练误差的步长, 训练多少个 iteration 就打印一次训练误差

        """
        self.model = model
        self.dataset = dataset

        self.model_path = model_path

        # 解析可选参数列表
        # kwargs = dict()
        self.optimize_mode = kwargs.pop('optimize_mode', 'MinBatch') # 将元素从字典中弹出, 若字典中没有此 key, 则赋值为默认值
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_log = kwargs.pop('print_log', True)
        self.print_every = kwargs.pop('print_every', 10)

        # 如果kwargs中还有其他参数未被弹出, 则报错
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # 配置优化算法
        class_name = self.optimize_mode + 'Optimizer'  # 'BGD' + 'Optimizer' = 'BGDOptimizer'
        if not hasattr( optim , class_name):
            raise ValueError('Invalid optimize_mode "%s"' % self.optimize_mode)

        Optimizer = getattr(optim, class_name)

        self.optimizer = Optimizer(self.optim_config)
        self.optim_param = {}  # 记录优化算法中需要更新的参数

        # 记录每一次迭代(iteration)的损失函数的值
        self.loss_history = []


    def fit_one_iteration(self):
        """
        模型训练的一次迭代 (iteration) , 多次迭代组成一个 epcho

        :return:
        """

        captions, images_feature, urls = sample_coco_minibatch(self.dataset, batch_size=self.batch_size, split='train')

        # images_feature = images_feature.T

        loss, grads = self.model.fit_batch( batch_sentence=captions, images_feature=images_feature)

        self.loss_history.append(loss) # 记录这次迭代的损失

        # 更新模型的参数
        for param_name, param_value in self.model.params.items():
            # self.params = {"U":U,"W":W,"V":V,"b_z":b_z,"b_o":b_o,"W_embed":W_embed,"W_pict":W_pict,"b_pict":b_pict}

            grad_param = grads['grad_'+param_name]

            next_param_value = self.optimizer.update_parameter((param_name,param_value), grad_param, self.optim_param)

            self.model.params[param_name] = next_param_value


    def fit(self):
        """
        训练模型

        :return:
        """

        num_train = self.dataset['train_captions'].shape[0] # 训练数据集中含有的样本个数 50

        iterations_per_epoch = max(num_train // self.batch_size, 1) # 每一个 epoch 要迭代的次数 50/25=2

        num_iterations = self.num_epochs * iterations_per_epoch # 总共迭代的次数

        for t in range(num_iterations):

            self.fit_one_iteration()

            # Maybe print training loss
            if self.print_log and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))

        # 存储训练好的模型
        self.model.save(self.model_path)

class Test:


    def test_rnn_image_caption(self):


        # 0. 准备数据集
        data = load_coco_data(base_dir='../dataset/coco_captioning', pca_features=True)

        # Print out all the keys and values from the data dictionary
        # for k, v in data.items():
        #     if type(v) == np.ndarray:
        #         print(k, type(v), v.shape, v.dtype)
        #     else:
        #         print(k, type(v), len(v))

        # train_captions  (400135, 17) train 为训练集
        # train_image_idxs  (400135,)  评论到图片的映射, 通过映射找到评论对应的图片向量
        # val_captions   (195954, 17)  val 为验证集
        # val_image_idxs  (195954,)
        # train_features  (82783, 512) 向量化后的图片, 维度为 512
        # val_features  (40504, 512)
        # idx_to_word <class 'list'> 1004 词表中单词的个数
        # word_to_idx <class 'dict'> 1004
        # train_urls   (82783,)
        # val_urls     (40504,)

        # Sample a minibatch and show the images and captions

        batch_size = 2

        # captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
        # for i, (caption, url) in enumerate(zip(captions, urls)):
        #     plt.imshow(image_from_url(url))
        #     plt.axis('off')
        #     caption_str = decode_captions(caption, data['idx_to_word'])
        #     plt.title(caption_str)
        #     plt.show()


        # np.random.seed(231)

        # 1.读取训练数据
        small_data = load_coco_data(base_dir='../dataset/coco_captioning',max_train=80000)

        # 2. 训练模型
        rnn_model = CaptionRNN(

            feature_dim=data['train_features'].shape[1],
            word_to_idx=data['word_to_idx'],
            hidden_dim=512,
            wordvec_dim=256,
            use_pre_train=False
        )

        solver = ImageCaptionSolver(rnn_model, small_data,
                                            optimize_mode='Adam',
                                            num_epochs=20,
                                            batch_size=512,
                                            optim_config={
                                                'learning_rate': 1e-3,
                                                'bias_correct':False
                                            },
                                            print_log=True,
                                            print_every=10,
                                            )


        # solver.fit()

        # # Plot the training losses
        # plt.plot(solver.loss_history)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Training loss history')
        # plt.show()

        # 3.测试模型

        rnn_model = CaptionRNN(
            use_pre_train=True
        )


        split = 'train'

        minibatch = sample_coco_minibatch(small_data, split=split, batch_size=10)

        origin_captions, features, urls = minibatch
        origin_captions = decode_captions(origin_captions, data['idx_to_word'])

        sample_captions = rnn_model.inference_sample(features)

        res_captions = decode_captions(sample_captions, data['idx_to_word'])

        for gt_caption, sample_caption, url in zip(origin_captions, res_captions, urls):
            plt.imshow(image_from_url(url))
            plt.title('%s\n %s\n origin:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.show()


    def test_lstm_image_caption(self):


        # 0. 了解数据集
        # data = load_coco_data(base_dir='../dataset/coco_captioning', pca_features=True)


        # train_captions  (400135, 17) train 为训练集
        # train_image_idxs  (400135,)  评论到图片的映射, 通过映射找到评论对应的图片向量
        # val_captions   (195954, 17)  val 为验证集
        # val_image_idxs  (195954,)
        # train_features  (82783, 512) 向量化后的图片, 维度为 512
        # val_features  (40504, 512)
        # idx_to_word <class 'list'> 1004 词表中单词的个数
        # word_to_idx <class 'dict'> 1004
        # train_urls   (82783,)
        # val_urls     (40504,)

        # Sample a minibatch and show the images and captions

        # batch_size = 2
        # captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
        # for i, (caption, url) in enumerate(zip(captions, urls)):
        #     plt.imshow(image_from_url(url))
        #     plt.axis('off')
        #     caption_str = decode_captions(caption, data['idx_to_word'])
        #     plt.title(caption_str)
        #     plt.show()


        np.random.seed(231)

        # 1.读取训练数据
        small_data = load_coco_data(base_dir='../dataset/coco_captioning', max_train=50000)

        # 2. 训练模型
        lstm_model = CaptionLSTM(

            feature_dim=small_data['train_features'].shape[1],
            word_to_idx=small_data['word_to_idx'],
            hidden_dim=512,
            wordvec_dim=256,
            use_pre_train=False,
            model_path='model/lstm_caption.model'
        )

        solver = ImageCaptionSolver(lstm_model, small_data,
                                            model_path='model/lstm_caption.model',
                                            optimize_mode='Adam',
                                            num_epochs=20,
                                            batch_size=512,
                                            optim_config={
                                                'learning_rate': 5e-3,
                                                'bias_correct':False
                                            },
                                            print_log=True,
                                            print_every=10,
                                            )


        solver.fit()

        # Plot the training losses
        plt.plot(solver.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training loss history')
        plt.show()

        # 3.测试模型

        lstm_model = CaptionLSTM(
            use_pre_train=True,
            model_path = 'model/lstm_caption.model'
        )


        split = 'train'

        minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2) # 在 small_data 中随机采样

        origin_captions, features, urls = minibatch
        origin_captions = decode_captions(origin_captions, small_data['idx_to_word'])

        sample_captions = lstm_model.inference_sample(features)

        res_captions = decode_captions(sample_captions, small_data['idx_to_word'])

        for gt_caption, sample_caption, url in zip(origin_captions, res_captions, urls):
            plt.imshow(image_from_url(url))
            plt.title('%s\n %s\n origin:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.show()

if __name__ == '__main__':

    test = Test()

    test.test_lstm_image_caption()


