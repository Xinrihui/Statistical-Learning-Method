#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib import optimizer_xrh as optim

from lib.lstm_caption_xrh import *
from lib.rnn_caption_xrh import *
from lib.bleu_xrh import *

from lib.microsoft_coco_dataset_xrh import *
from flicker_dataset_xrh import *
from lib.image_utils import *
from lib.evaluate_xrh import *


class ImageCaptionSolver:
    """

    对图片做文字描述 (image caption) 的模型的包装器

    示例代码:

    dataset = DataSet()
    model = RNNModel(hidden_dim=100)
    solver = CaptioningSolver(model, dataset,
                    update_rule='BGD',
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

    - model.inference_batch(images_feature, caption_length) 利用训练好的模型进行推理

    Author: xrh
    Date: 2021-08-01

    """

    def __init__(self, model, dataset,
                 model_path='models/lstm_caption.model',
                 **kwargs):
        """

        必要参数:
        - model: RNN 模型对象
        - dataset: 数据集对象, 通过它可以获得的训练集和验证集数据
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
        self.optimize_mode = kwargs.pop('optimize_mode', 'MinBatch')  # 将元素从字典中弹出, 若字典中没有此 key, 则赋值为默认值
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size = kwargs.pop('batch_size', 128)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_log = kwargs.pop('print_log', True)
        self.print_every = kwargs.pop('print_every', 10)

        # 如果kwargs中还有其他参数未被弹出, 则报错
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # 配置优化算法
        class_name = self.optimize_mode + 'Optimizer'  # 'BGD' + 'Optimizer' = 'BGDOptimizer'
        if not hasattr(optim, class_name):
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

        captions, images_feature = self.dataset.sample_minibatch(
            batch_size=self.batch_size)  # 从 dataset 随机采样 batch_size 个样本

        # images_feature = images_feature.T

        loss, grads = self.model.fit_batch(batch_sentence=captions, images_feature=images_feature)

        self.loss_history.append(loss)  # 记录这次迭代的损失

        # 更新模型的参数
        for param_name, param_value in self.model.params.items():
            # self.params = {"U":U,"W":W,"V":V,"b_z":b_z,"b_o":b_o,"W_embed":W_embed,"W_pict":W_pict,"b_pict":b_pict}

            grad_param = grads['grad_' + param_name]

            next_param_value = self.optimizer.update_parameter((param_name, param_value), grad_param, self.optim_param)

            self.model.params[param_name] = next_param_value

    def fit(self):
        """
        训练模型

        :return:
        """

        print('train dataset num:{}, picture_feature_dim:{}, caption_length:{}'.format(self.dataset.N, self.dataset.feature_dim, self.dataset.caption_length))  # 训练数据集中含有的样本个数

        iterations_per_epoch = max(self.dataset.N // self.batch_size, 1)  # 每一个 epoch 要迭代的次数 50/25=2

        # num_iterations = self.num_epochs * iterations_per_epoch  # 总共迭代的次数

        for epoch in range(1, self.num_epochs+1):

            print('Epoch: {}/{}'.format(epoch, self.num_epochs))

            for t in tqdm(range(iterations_per_epoch)):  # 可显示进度条
                self.fit_one_iteration()

            print('loss:{}'.format(self.loss_history[-1]))  # TODO：tqdm 会在这里多打印一行

        # 存储训练好的模型
        self.model.save()

    def inference(self, features, max_length):
        """
        推理, 并对推理结果进行解码

        :param features: 图片特征向量
        :param max_length: 推理序列的长度
        :return:
        """

        decode_result = self.model.inference_batch(features, caption_length=max_length)

        candidates = []

        # print(decode_result)

        for prediction in decode_result:
            output = ' '.join([self.dataset.vocab_obj.map_id_to_word(i) for i in prediction])
            candidates.append(output)

        return candidates




class Test:

    def test_rnn_image_caption_microsoft_coco(self):

        # 0. 了解数据集

        # coco_dataset = MicrosoftCocoDataset(base_dir='../dataset/ImageCaption/microsoft_coco')

        # Print out all the keys and values from the data dictionary

        # for k, v in coco_dataset.dataset.items():
        #     if type(v) == np.ndarray:
        #         print(k, type(v), v.shape, v.dtype)
        #     else:
        #         print(k, type(v), len(v))

        # Sample a minibatch and show the images and captions

        # batch_size = 2
        #
        # captions, features, urls = coco_dataset.sample_minibatch(batch_size=batch_size, return_url=True)
        # for i, (caption, url) in enumerate(zip(captions, urls)):
        #     plt.imshow(image_from_url(url))
        #     plt.axis('off')
        #     caption_str = coco_dataset.decode_captions(caption)
        #     plt.title(caption_str)
        #     plt.show()

        # np.random.seed(231)

        # 1.读取训练数据
        coco_dataset = MicrosoftCocoDataset(base_dir='../dataset/ImageCaption/microsoft_coco', sample_N=500)

        model_path = 'models/microsoft coco/rnn_caption.model'

        # 2. 训练模型
        rnn_model = CaptionRNN(

            feature_dim=coco_dataset.dataset['train_features'].shape[1],
            word_to_idx=coco_dataset.dataset['word_to_idx'],
            hidden_dim=512,
            wordvec_dim=256,
            model_path=model_path,
            use_pre_train=False
        )

        solver = ImageCaptionSolver(rnn_model, coco_dataset,
                                    optimize_mode='Adam',
                                    num_epochs=50,
                                    batch_size=25,
                                    optim_config={
                                        'learning_rate': 5e-3,
                                        'bias_correct': True
                                    },
                                    print_log=True,
                                    print_every=10
                                    )

        solver.fit()

        # # Plot the training losses
        # plt.plot(solver.loss_history)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Training loss history')
        # plt.show()

        # 3.模型推理

        rnn_model_infer = CaptionRNN(

            use_pre_train=True,
            model_path=model_path  # 预训练的模型的路径
        )

        infer = ImageCaptionSolver(rnn_model_infer, coco_dataset)

        Type = 'train'  # 训练数据集
        # Type = 'val'

        minibatch = coco_dataset.sample_minibatch(Type=Type, batch_size=4, return_url=True)  # 在 small_data 中随机采样

        origin_captions, features, urls = minibatch
        references = coco_dataset.decode_captions(origin_captions)

        candidates = infer.inference(features, max_length=20)

        # candidates = coco_dataset.decode_captions(candidates)

        print('candidates: ', candidates)
        print('references: ', references)

        # for gt_caption, sample_caption, url in zip(origin_captions, res_captions, urls):
        #     plt.imshow(image_from_url(url))
        #     plt.title('%s\n %s\n origin:%s' % (Type, sample_caption, gt_caption))
        #     plt.axis('off')
        #     plt.show()

    def test_lstm_image_caption_microsoft_coco(self):


        # 1.读取训练数据

        # coco_dataset = MicrosoftCocoDataset(sample_N=50000)

        coco_dataset = MicrosoftCocoDataset(base_dir='../dataset/ImageCaption/microsoft_coco', sample_N=500, use_pca_features=False)

        print('train dataset num:{}, picture_feature_dim:{}, caption_length:{}'.format(coco_dataset.N, coco_dataset.feature_dim, coco_dataset.caption_length))  # 训练数据集中含有的样本个数


        # 2. 训练模型

        model_path = 'models/microsoft coco/lstm_caption.model'

        lstm_model = CaptionLSTM(

            feature_dim=coco_dataset.dataset['train_features'].shape[1],
            word_to_idx=coco_dataset.dataset['word_to_idx'],
            hidden_dim=512,
            wordvec_dim=512,
            use_pre_train=False,
            model_path=model_path  # 预训练的模型的路径
        )

        solver = ImageCaptionSolver(lstm_model, coco_dataset,
                                    optimize_mode='Adam',
                                    num_epochs=10,
                                    batch_size=64,
                                    optim_config={
                                        'learning_rate': 5e-3,
                                        'bias_correct': True
                                    },
                                    print_log=True,
                                    print_every=10,
                                    )

        solver.fit()

        # Plot the training losses
        # plt.plot(solver.loss_history)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Training loss history')
        # plt.show()

        # 3.模型推理

        lstm_model_infer = CaptionLSTM(

            use_pre_train=True,
            model_path=model_path  # 预训练的模型的路径
        )

        infer = ImageCaptionSolver(lstm_model_infer, coco_dataset)

        Type = 'train'  # 训练数据集
        # Type = 'val'
        max_length = 20

        test_data = coco_dataset.sample_minibatch(Type=Type, batch_size=4, return_url=True)  # 在 small_data 中随机采样

        origin_captions, features, urls = test_data

        references = coco_dataset.decode_captions(origin_captions)

        candidates = infer.inference(features, max_length=max_length)

        print('candidates: ', candidates)
        print('references: ', references)

        # for gt_caption, sample_caption, url in zip(origin_captions, res_captions, urls):
        #     plt.imshow(image_from_url(url))
        #     plt.title('%s\n %s\n origin:%s' % (Type, sample_caption, gt_caption))
        #     plt.axis('off')
        #     plt.show()

    def test_rnn_image_caption_flicker(self):

        # 1.读取训练数据

        flicker_dataset = FlickerDataset(base_dir='../dataset/ImageCaption/')

        # 2. 训练模型

        model_path = 'models/rnn_caption_300_256.model'

        lstm_model = CaptionRNN(

            feature_dim=flicker_dataset.image_feature.shape[1],
            word_to_idx=flicker_dataset.vocab_obj.word_to_id,
            hidden_dim=300,
            wordvec_dim=256,
            use_pre_train=False,
            model_path=model_path  # 预训练的模型的路径
        )

        trainer = ImageCaptionSolver(lstm_model, flicker_dataset,
                                    optimize_mode='Adam',
                                    num_epochs=20,
                                    batch_size=64,
                                    optim_config={
                                        'learning_rate': 5e-3,
                                        'bias_correct': True
                                    },
                                    print_log=True,
                                    print_every=10,
                                    )

        trainer.fit()

        # Plot the training losses
        # plt.plot(solver.loss_history)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Training loss history')
        # plt.show()

        # 3.模型推理

        lstm_model_infer = CaptionRNN(

            feature_dim=flicker_dataset.image_feature.shape[1],
            word_to_idx=flicker_dataset.vocab_obj.word_to_id,
            hidden_dim=300,
            wordvec_dim=256,
            use_pre_train=True,
            model_path=model_path  # 预训练的模型的路径
        )

        infer = ImageCaptionSolver(lstm_model_infer, flicker_dataset)

        image_caption_dict = flicker_dataset.data_process.load_image_caption_dict()

        image_dir_list = list(image_caption_dict.keys())

        m = 1619  # 测试数据集的图片个数

        image_dir_batch = image_dir_list[:m]

        print('test image num:{}'.format(len(image_dir_batch)))

        batch_image_feature = np.array(
            [list(image_caption_dict[image_dir]['feature']) for image_dir in image_dir_batch])

        references = [image_caption_dict[image_dir]['caption'] for image_dir in image_dir_batch]

        candidates = infer.inference(batch_image_feature, max_length=20)

        # print('candidates: ', candidates)
        # print('reference: ', references)

        evaluate_obj = Evaluate()

        bleu_score, _ = evaluate_obj.evaluate_bleu(references, candidates)

        print('bleu_score:{}'.format(bleu_score))


    def test_lstm_image_caption_flicker(self):

        # 1.读取训练数据

        flicker_dataset = FlickerDataset(base_dir='../dataset/ImageCaption/')

        # 2. 训练模型

        model_path = 'models/lstm_caption_512_512.model'

        lstm_model = CaptionLSTM(

            feature_dim=flicker_dataset.image_feature.shape[1],
            word_to_idx=flicker_dataset.vocab_obj.word_to_id,
            # dtype=np.float64,
            hidden_dim=512,
            wordvec_dim=512,
            use_pre_train=False,
            model_path=model_path  # 预训练的模型的路径
        )

        # trainer = ImageCaptionSolver(lstm_model, flicker_dataset,
        #                             optimize_mode='Adam',
        #                             num_epochs=50,
        #                             batch_size=128,
        #                             optim_config={
        #                                 'learning_rate': 5e-3,
        #                                 'bias_correct': False
        #                             },
        #                             print_log=True,
        #                             print_every=10,
        #                             )

        # trainer.fit()

        # Plot the training losses
        # plt.plot(trainer.loss_history)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Training loss history')
        # plt.show()

        # 3.模型推理

        lstm_model_infer = CaptionLSTM(

            use_pre_train=True,
            model_path=model_path  # 预训练的模型的路径
        )

        infer = ImageCaptionSolver(lstm_model_infer, flicker_dataset)

        image_caption_dict = flicker_dataset.data_process.load_image_caption_dict()

        image_dir_list = list(image_caption_dict.keys())

        m = 1619  # 测试数据集的图片个数
        max_length = 30

        image_dir_batch = image_dir_list[:m]

        print('test image num:{}'.format(len(image_dir_batch)))

        batch_image_feature = np.array(
            [list(image_caption_dict[image_dir]['feature']) for image_dir in image_dir_batch])

        references = [image_caption_dict[image_dir]['caption'] for image_dir in image_dir_batch]

        candidates = infer.inference(batch_image_feature, max_length=max_length)

        print('candidates: ', candidates[0:10])

        print('reference: ', references[0:10])

        evaluate_obj = Evaluate()

        bleu_score, _ = evaluate_obj.evaluate_bleu(references, candidates)

        print('bleu_score:{}'.format(bleu_score))

if __name__ == '__main__':
    test = Test()

    # test.test_rnn_image_caption_microsoft_coco()

    test.test_lstm_image_caption_microsoft_coco()

    # test.test_rnn_image_caption_flicker()

    # test.test_lstm_image_caption_flicker()
