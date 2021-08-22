#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from lstm_layers_xrh import *

from Initializer_xrh import *

from utils_xrh import *

import pickle

class CaptionLSTM:
    """
    LSTM 循环神经网络,

    适用于构建
    (1) 语言模型 (language model)
    (2) 对图片做文字描述 (image caption)

    词典大小为 vocab_size
    词向量的维度为 m
    隐藏层的维度为 n_h
    输出层的维度为 class_num
    图片一次提取特征后的维度为 n_p
    图片二次提取特征后的维度为 n_h

    Author: xrh
    Date: 2021-08-21

    """

    def __init__(self, word_to_idx=None, feature_dim=512,wordvec_dim=256, hidden_dim=128, dtype=np.float32,
                      model_path = 'model/lstm_caption.model',
                      use_pre_train = True

    ):
        """
        RNN 参数的初始化


        :param feature_dim: 图片提取特征后的维度
        :param word_to_idx: 单词到单词标号的映射
        :param wordvec_dim: 词向量的维度 m
        :param hidden_dim: 隐藏层的维度 n_h
        :param dtype: 模型参数的数据类型, 可以配置模型的精度
                      np.float32
                      np.float64

        :param model_path: 预训练模型的路径
        :param use_pre_train: 是否使用预训练的模型
                     True: 读取预训练模型的参数后直接可以进行推理, 训练时在预训练的基础上进行训练
                     False: 从头开始训练模型
        """

        if not use_pre_train:  # 从头开始训练模型

            self.params = {}  # 模型参数(需要更新)
            self.dtype = dtype # 模型参数的数据类型

            self.word_to_idx = word_to_idx # 单词到单词标号的映射
            self.idx_to_word = {i: w for w, i in word_to_idx.items()} # 单词标号到单词的映射

            vocab_size = len(word_to_idx) # 词典的大小

            self._null = word_to_idx['<NULL>'] # 空
            self._start = word_to_idx.get('<START>', None) # 句子的开始
            self._end = word_to_idx.get('<END>', None) # 句子的结束

            self.n_p = feature_dim
            self.n_h = hidden_dim
            self.m = wordvec_dim
            self.class_num = vocab_size # 语言模型的输出为词表中单词出现的概率

            # 需要 Xavier 初始化的参数:
            # V shape(class_num,n_h) , W_pict shape(n_h,n_p),
            # U_f shape(n_h,m), W_f shape(n_h, n_h)
            # U_i shape(n_h,m), W_i shape(n_h, n_h)
            # U_a shape(n_h,m), W_a shape(n_h, n_h)
            # U_o shape(n_h,m), W_o shape(n_h, n_h)

            xavier = XavierInitializer()
            V,W_pict,U_f,W_f,U_i,W_i,U_a,W_a,U_o,W_o = xavier.initialize_parameters([(self.class_num,self.n_h),(self.n_h, self.n_p),
                                                                                     (self.n_h,self.m),(self.n_h,self.n_h),
                                                                                     (self.n_h, self.m), (self.n_h, self.n_h),
                                                                                     (self.n_h, self.m), (self.n_h, self.n_h),
                                                                                     (self.n_h, self.m), (self.n_h, self.n_h),
                                                                                     ])

            # 需要 0初始化的参数:
            # b_y shape(class_num,1), b_pict shape(n_h,1),
            # b_f shape(n_h,1), b_i shape(n_h,1) , b_a shape(n_h,1) , b_o shape(n_h,1)
            zero = ZeroInitializer()
            b_y, b_pict, b_f, b_i, b_a, b_o = zero.initialize_parameters([(self.class_num,1),(self.n_h, 1),
                                                                         (self.n_h,1),(self.n_h,1),(self.n_h,1),(self.n_h,1)
                                                                         ])

            # 需要 随机初始化的参数:
            # W_embed shape (vocab_size, m)
            rand = RandomInitializer()
            W_embed = rand.initialize_parameters([(vocab_size, self.m)])[0]

            self.params = {"V":V,"W_pict":W_pict,
                           "U_f":U_f,"W_f":W_f,
                           "U_i":U_i,"W_i":W_i,
                           "U_a":U_a,"W_a":W_a,
                           "U_o": U_o, "W_o": W_o,
                           "b_y": b_y, "b_pict": b_pict,
                           "b_f": b_f, "b_i": b_i,"b_a": b_a, "b_o": b_o,
                           "W_embed": W_embed
                           }

            self.rnn_layer = LSTMLayer()

            # 将所有的参数转换为 dtype 类型
            for k, v in self.params.items():
                self.params[k] = v.astype(self.dtype)


        else:  # 使用预训练模型

            self.load(model_path)


    def save(self, model_dir):
        """
        保存训练好的模型

        :param train_data_dir:
        :return:
        """
        save_dict = {}

        save_dict['params'] = self.params
        save_dict['dtype'] = self.dtype

        save_dict['word_to_idx'] = self.word_to_idx
        save_dict['idx_to_word'] = self.idx_to_word

        save_dict['_null'] = self._null
        save_dict['_start'] = self._start
        save_dict['_end'] = self._end

        save_dict['n_p'] = self.n_p
        save_dict['n_h'] = self.n_h
        save_dict['m'] = self.m
        save_dict['class_num'] = self.class_num

        save_dict['rnn_layer'] = self.rnn_layer

        with open(model_dir, 'wb') as f:
            pickle.dump(save_dict, f)

        print("Save model successful!")

    def load(self, file_path):
        """
        读取预训练的模型

        :param file_path:
        :return:
        """

        with open(file_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.params = save_dict['params']
        self.dtype = save_dict['dtype']

        self.word_to_idx = save_dict['word_to_idx']
        self.idx_to_word = save_dict['idx_to_word']

        self._null = save_dict['_null']
        self._start = save_dict['_start']
        self._end = save_dict['_end']

        self.n_p = save_dict['n_p']
        self.n_h = save_dict['n_h']
        self.m = save_dict['m']
        self.class_num = save_dict['class_num']

        self.rnn_layer = save_dict['rnn_layer']

        print("Load model successful!")

    def fit_batch(self, batch_sentence,images_feature):
        """
        用一个批次的训练数据拟合 RNN,
        依次运行各个层的前向传播算法 和 后向传播算法, 计算损失函数, 计算模型参数的梯度

        :param images_feature: 一次特征抽取后的向量化的图片 shape (N,n_p)  N-样本个数 n_p-图片向量维度
        :param batch_sentence: 一批句子 shape(N,T) ,句子由单词的标号构成 N-样本个数 T-句子长度

        eg.
           N = 2,
           origin_sentences[0] = '<start>/今天/是/个/好日子/<end>/<NULL>/<NULL>/<NULL>/<NULL>'
           origin_sentences[1] = '<start>/天空/是/蔚蓝色/<end>/<NULL>/<NULL>/<NULL>/<NULL>'

           一个 batch 中 sentence为固定长度(T = 9), 若不足则在末尾补充 <NULL> (padding)

           # 1-<start> 2-<end> 0-<NULL>
           batch_sentence[0] = [1,10,11,12,13,2,0,0,0]
           batch_sentence[1] = [1,20,11,21,2,0,0,0,0]

        :return: loss 模型的损失 ;
                 grads 模型的梯度
        """
        # 语言模型的输入 和 输出要错开一个时刻,
        # eg.
        #  output: 今天   /是   /个/好日子/<end>
        #   input: <start>/今天/是/个    /好日子/

        batch_out = batch_sentence[:, 1:] #  shape(N,T-1)
        batch_in = batch_sentence[:, :-1] #  shape(N,T-1)

        mask = (batch_out != self._null) # shape(N,T-1)
        # 因为训练时采用 mini-batch, 一个 batch 中的所有的 sentence 都是定长, 若有句子不够长度 则用 <null> 进行填充
        # 用 <null> 填充的时刻不能被计入损失中, 也不用求梯度

        # 词嵌入层
        x_list, cache_embed = self.rnn_layer.word_embedding_forward(parameters=self.params, batch_sentence=batch_in)
        # x_list shape (N, T, m)

        # images_feature = images_feature.T # (N,n_p)

        # 图片嵌入层
        h0, cache_pict = self.rnn_layer.picture_embedding_forward(parameters=self.params, origin_feature=images_feature)
        # h0 shape(n_h,N)

        # 中间层
        o_list, C_list, h_list, cache_list_mid = self.rnn_layer.middle_forwoard_propagation(parameters=self.params, x_list=x_list, h_init=h0)
        # h_list shape (N,T,n_h)

        # 输出层
        z_y_list, y_ba_list, cache_out = self.rnn_layer.temporal_affine_forward(parameters=self.params, o_list=o_list, C_list=C_list, h_list=h_list)
        # z_y_list shape (N, T, class_num)

        # 样本标签的 one-hot 化 shape (N,T-1) ->  (N,T-1,class_num)
        batch_out_onehot = Utils.convert_to_one_hot(x=batch_out,class_num=self.class_num)  # shape (N,T-1,class_num)

        # 计算损失函数
        loss, grad_z_y_list = self.rnn_layer.multi_classify_loss_func(z_y_list=z_y_list, y_ba_list=y_ba_list, y_onehot_list=batch_out_onehot, mask=mask)

        #计算梯度
        # 输出层
        outLayer_grad_C_list, outLayer_grad_h_list, grad_dict_out = self.rnn_layer.temporal_affine_bakward(grad_z_y_list=grad_z_y_list,cache=cache_out)

        # 中间层
        grad_h_pre, grad_x_list, grad_dict_middle = self.rnn_layer.middle_bakwoard_propagation(outLayer_grad_C_list=outLayer_grad_C_list,
                                                                                             outLayer_grad_h_list=outLayer_grad_h_list,
                                                                                             cache_list=cache_list_mid)

        # 词嵌入层
        grad_dict_embed = self.rnn_layer.word_embedding_backward(grad_x_list=grad_x_list, cache=cache_embed)

        # 图片嵌入层
        grad_dict_pict = self.rnn_layer.picture_embedding_backward(grad_a0=grad_h_pre, cache=cache_pict)

        # 各个层梯度合并
        grads = {**grad_dict_out, **grad_dict_middle, **grad_dict_embed, **grad_dict_pict}

        assert len(grads) == (len(grad_dict_out)+len(grad_dict_middle)+len(grad_dict_embed)+len(grad_dict_pict)) # 各个 dict 中不能有重复的元素

        assert len(grads) == len(self.params) # 模型参数的梯度必须和模型参数 匹配

        return loss, grads


    def inference_sample(self, images_feature, caption_length=30):
        """
        利用训练好的模型进行推理, 输出对输入图片的描述(caption);
        第一个时间步输入的词为 <start>, 然后取模型输出的概率最大的词作为下一个时间步的输入,
        以此类推, 知道达到最大的 caption长度

        :param images_feature: 一次抽取特征后的向量化的图片 shape (N, n_p)  N-样本个数 n_p-图片向量维度
        :param caption_length: 输出的 caption 的最大的长度
        :return: caption

        """
        N = np.shape(images_feature)[0]

        # 第一个时间步输入的词为 <start>
        batch_in = np.ones((N,1))*self._start

        caption = np.zeros((N,caption_length),dtype=np.int32)

        # 图片嵌入层
        h0, _ = self.rnn_layer.picture_embedding_forward(parameters=self.params, origin_feature=images_feature)
        # h0 shape(n_h,N)
        C0 = None

        for t in range(caption_length): # 遍历所有时间步

            # 词嵌入层
            x_t, _ = self.rnn_layer.word_embedding_forward(parameters=self.params, batch_sentence=batch_in)
            # x_t shape (N, T=1, m)

            # 中间层
            o_t, C_t, h_t, _ = self.rnn_layer.middle_forwoard_propagation(parameters=self.params, x_list=x_t, C_init=C0, h_init=h0)
            # h_t shape (N,T=1,n_h)
            # C_t shape (N,T=1,n_h)

            # C_t,h_t 要输入到下一个时间步
            h0 = h_t.reshape(N,self.n_h).T # shape(n_h,N)
            C0 = C_t.reshape(N, self.n_h).T  # shape(n_h,N)

            # 输出层
            _, y_ba_t, _ = self.rnn_layer.temporal_affine_forward(parameters=self.params, o_list=o_t, C_list=C_t, h_list=h_t)
            #  y_ba_t  shape (N, T=1, class_num)

            y_ba_t = np.squeeze(y_ba_t) # shape (N, class_num)

            # 选择出现概率最大的单词
            caption[:,t] = np.argmax(y_ba_t, axis=1)  # axis=1 干掉第1个维度, shape: (N,)
            #  caption[:,t] shape (N,1)

            # 将概率最大的单词 输入下一个时间步
            batch_in = caption[:,t].reshape(-1,1)

        return caption



