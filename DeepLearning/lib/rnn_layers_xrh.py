#!/usr/bin/python
# -*- coding: UTF-8 -*-

from deprecated import deprecated

import numpy as np

from Activation_xrh import Activation

from gradient_check_xrh import *

from collections import *

from utils_xrh import *

class RNNLayer:
    """
    RNN 的相关层

    (1) 中间层的前向传播
        中间层的反向传播
    (2) 词嵌入层的正向传播
        词嵌入层的反向传播
    (3) 时序仿射层(输出层)的正向传播
        时序仿射层的反向传播
    (4) 图片嵌入层的正向传播
        图片嵌入层的反向传播
    (5) 损失函数层

    Author: xrh
    Date: 2021-07-26

    """

    def picture_embedding_forward(self,parameters,origin_feature):
        """
        图片嵌入层的正向传播算法

        :param parameters: 模型参数
        :param origin_feature: 一次抽取特征后的图片 shape (N,n_p)

        :return: pict_feature: 二次抽取特征后的图片  shape(n_h,N) ;
                 cache : (origin_feature, parameters) 用于反向传播

        """
        W_pict = parameters["W_pict"]
        b_pict = parameters["b_pict"]

        pict_feature = np.dot(W_pict,origin_feature.T) + b_pict # shape(n_h,N)
        # W_pict shape(n_h,n_p), images_feature.T shape (n_p,N)

        cache = (origin_feature, parameters)

        return pict_feature,cache

    def picture_embedding_backward(self, grad_a0, cache):
        """
        图片嵌入层的反向传播算法

        :param grad_a0:  shape: (n_h,N)
                       初始时刻的 a 的梯度: 由 1时刻传播而来

        :param cache: 正向传播的缓存
                
        :return: 
        """
        N = np.shape(grad_a0)[1]

        origin_feature, parameters = cache

        grad_W_pict = np.dot(grad_a0, origin_feature)
        # grad_W_pict /= N

        grad_b_pict = np.sum(grad_a0,axis=1,keepdims=True)
        # grad_b_pict /= N

        grad_dict = {"grad_W_pict":grad_W_pict, "grad_b_pict":grad_b_pict }

        return grad_dict



    def word_embedding_forward(self,parameters,batch_sentence):
        """
        词嵌入层的前向传播算法

        :param batch_sentence: 一批句子 shape(N,T) 句子由单词的标号构成
        eg.
           N = 2,
           origin_sentences[0] = '<start>/今天/是/个/好日子/<end><NULL><NULL><NULL><NULL>'
           origin_sentences[1] = '<start>/天空/是/蔚蓝色/<end><NULL><NULL><NULL><NULL>'

           sentence为固定长度(T = 9), 若不足则在末尾补充0

           batch_sentence[0] = [1,10,11,12,13,2,0,0,0]
           batch_sentence[1] = [1,20,11,21,2,0,0,0,0] # 1-<start> 2-<end> 0-<NULL>

        :param parameters: 模型参数

        parameters["W_emb"]: 词向量的矩阵 shape (vocab_size, m) vocab_size-词典大小 m-词向量的维度

        :return: x_list:  词向量 shape (N, T, m) 作为下游 RNN中间层的输入 ;
                 cache: (word_list, W_embed) 用于反向传播

        """
        W_embed = parameters["W_embed"]

        x_list = W_embed[batch_sentence.astype(np.int32)] # shape (N, T, m)
        # W_embed shape (vocab_size, m) , batch_sentence shape (N, T)

        cache = (batch_sentence, W_embed)

        return x_list, cache

    def word_embedding_backward(self, grad_x_list, cache):
        """
        词嵌入的反向传播算法

        :param grad_x_list: 上游传递给词向量的梯度 shape (N, T, m) N- 样本个数 ,T-时刻长度 ,m-特征维度
        :param cache: 词嵌入层的前向传播算法的缓存
               cache = (word_list, word2vec)
        :return:
            grad_word2vec: shape (vocab_size, m)

        """
        N = np.shape(grad_x_list)[0]

        batch_sentence, W_embed = cache
        grad_W_embed = np.zeros(np.shape(W_embed))  # shape (vocab_size, m)

        np.add.at(grad_W_embed, batch_sentence, grad_x_list) # grad_X_list 中可能会含有同一个词向量的梯度
        # add.at 和 += 类似
        # grad_word2vec shape (vocab_size, m), word_list shape (N, T) ,  grad_X_list shape (N, T, m)

        # grad_W_embed /= N # TODO: 参考 cs231n 去掉除以 N

        grad_dict = {"grad_W_embed":grad_W_embed}

        return grad_dict

    def middle_layer_forwoard(self,parameters,x,a_prev,activation_func):
        """
        t时刻的前向传播算法, 只有中间层, 不包括输出层

        样本特征的维度为 n_m, 隐藏层的维度为 n_h

        :param parameters:模型参数
        :param x: t时刻的样本特征 shape (m,N)  N-样本个数 , m-特征维度
        :param a_prev: t-1时刻的隐藏层的激活值  shape (n_h,N)
        :param activation_func: 选择的激活函数
        :return: a,cache
        cache = (x, a_prev,z,a, parameters)

        """

        U = parameters["U"]
        W = parameters["W"]
        b_z = parameters["b_z"]

        # 中间层
        z = np.dot(U,x)+np.dot(W,a_prev)+b_z # shape: (n_h,N)
        # U shape: (n_h,m) , x shape: (m,N) -> (n_h,N)
        # W shape (n_h, n_h) , a_prev shape  (n_h,N)  ->  (n_h,N)

        a = activation_func(z) # shape: (n_h,N)

        cache = (x,a_prev,z,a,parameters)

        return a,cache


    def middle_layer_bakwoard(self,grad_a,cache,grad_activation_func):
        """
        t时刻中间层的后向传播算法

        :param grad_a:   shape: (n_h,N)
                当前时刻的 a 的梯度由两部分组成：
                (1) 由输出层传播而来
                (2) 由下一时刻传播而来
        :param cache: 前向传播 t时刻的缓存
                    cache = (x,a_prev,z,a,parameters)
        :param grad_activation_func: 激活函数的一阶导数
        :return:
            grad_a_pre : 传递给前一个时刻的梯度
            grad_x: 传递给 词向量的梯度
            grad_dic: 之后要更新的模型参数的梯度
             {"grad_U":grad_U,"grad_W":grad_W,"grad_b_z":grad_b_z}


        """
        N = np.shape(grad_a)[1]

        (x, a_prev, z, a, parameters) = cache

        U = parameters['U']
        W = parameters['W']


        grad_z = grad_a*grad_activation_func(z)
        # grad_a shape:(n_h,N) ,  z shape:(n_h,N)

        grad_U = np.dot(grad_z,x.T) # shape: (n_h,m)
        # grad_z shape:(n_h,N) , x.T shape:(N,m)
        # grad_U /= N # TODO: 参考 cs231n 不除 N

        grad_W = np.dot(grad_z,a_prev.T) # shape: (n_h,n_h)
        # grad_z shape:(n_h,N) , a_prev.T shape:(N,n_h)
        # grad_W /= N

        grad_b_z = np.sum(grad_z, axis=1, keepdims=True) # shape:(n_h,1)
        # grad_b_z /= N

        grad_a_pre = np.dot(W.T,grad_z) # shape:(n_h,N)
        # W.T shape: (n_h,n_h) , grad_z shape:(n_h,N)

        grad_x = np.dot(U.T, grad_z)  # shape:(m,N)
        # U.T shape: (m,n_h) , grad_z shape:(n_h,N)

        grad_dic = {"grad_U":grad_U,"grad_W":grad_W,"grad_b_z":grad_b_z}

        return grad_a_pre,grad_x,grad_dic

    def middle_forwoard_propagation(self,
                             parameters,
                             x_list,
                             a_init=None,
                             activation='tanh'
                             ):
        """
        RNN 中间层的前向传播算法

        :param parameters: 模型参数
        :param x_list: 样本特征的时刻列表 shape (N,T,m) N- 样本个数 ,T-时刻长度 ,m-特征维度
        :param y_onehot_list: 样本标签的时刻列表 shape (N,T,class_num) N- 样本个数 ,T-时刻长度 ,
                                                                     class_num-分类的个数
        :param a_init 隐藏层的初始化向量, shape: (n_h,N)
        :param activation: 激活函数

        :return: a_list : 中间层的激活向量 shape: (N,T,n_h) ;
            cache_list : 所有时刻的 cache 列表, 用于反向传播

        """
        N, T, m = np.shape(x_list)

        n_h = np.shape(parameters['W'])[0]  # 隐藏层的维度

        if a_init is None:
            a_init = np.zeros((n_h,N))

        if not hasattr(Activation , activation):
            raise ValueError('Invalid activation func "%s"' % activation)

        activation_func = getattr(Activation, activation) # 激活函数
        grad_activation_func = getattr(Activation, 'grad_'+activation) # 激活函数的一阶导数

        cache_list=[]
        a_list = np.zeros((N,T,n_h))

        a_prev = a_init

        # t=0,1,...,T-1
        for t in range(T):

            a,cache = self.middle_layer_forwoard(parameters=parameters, x=x_list[:,t,:].T, a_prev=a_prev, activation_func=activation_func)
            # a shape: (n_h,N)

            a_list[:,t,:] = a.T # a.T shape: (N,n_h)
            cache_list.append(cache)

            a_prev = a

        return a_list,cache_list

    def middle_bakwoard_propagation(self, outLayer_grad_a_list,
                                          cache_list,
                                          activation='tanh'
                                    ):
        """
         RNN 中间层的后向传播算法

        :param outLayer_grad_a_list:  输出层对隐藏层的向量 a的一阶梯度的列表 shape (N,T,n_h)
        :param cache_list: 中间层的前向传播算法的缓存
                      cache = (x,a_prev,z,a,parameters)
        :param activation: 激活函数
        :return:

            grad_a_pre : 第 1 时间步的激活值的梯度
            grad_x_list: 词向量的梯度列表 shape (N,T,m)
            sum_grad : 所有时刻的梯度加和
                    sum_grad = {"grad_U":grad_U,"grad_W":grad_W,"grad_b_z":grad_b_z}

        """

        N, T, n_h = np.shape(outLayer_grad_a_list)

        x = cache_list[0][0]
        m = np.shape(x)[0]

        if not hasattr(Activation , activation):
            raise ValueError('Invalid activation func "%s"' % activation)

        activation_func = getattr(Activation, activation) # 激活函数
        grad_activation_func = getattr(Activation, 'grad_'+activation) # 激活函数的一阶导数

        grad_list=[]
        grad_x_list = np.zeros((N,T,m))

        # 1.最后一个时刻 t=T-1
        t = T-1
        grad_a_pre,grad_x,grad = self.middle_layer_bakwoard( grad_a=outLayer_grad_a_list[:,t,:].T, cache=cache_list[t],
                                      grad_activation_func=grad_activation_func)
        # 最后一个时刻的a 的梯度 grad_a 只由输出层传递而来

        # grad = {"grad_U":grad_U,"grad_W":grad_W,"grad_b_z":grad_b_z}

        grad_x_list[:,t,:] = grad_x.T

        grad_list.append(grad)

        # 2.中间和初始时刻
        for t in range(T-2,-1,-1): # t = T-2,...,0

            grad_a_pre,grad_x,grad = self.middle_layer_bakwoard(grad_a=outLayer_grad_a_list[:,t,:].T + grad_a_pre, cache=cache_list[t],
                                                     grad_activation_func=grad_activation_func)
            # 当前时刻的 a 的梯度 grad_a 由两部分组成:
            #  一部分由输出层传播而来 outLayer_grad_a_list[t] , 另一部分由下一时刻传播而来 grad["grad_a_pre"]

            grad_x_list[:, t, :] = grad_x.T

            grad_list.append(grad)

        # 对所有时刻的梯度进行加和
        sum_grad_dict = defaultdict(float)

        for t in range(len(grad_list)):

            for grad_name,grad in grad_list[t].items():

                    sum_grad_dict[grad_name] += grad

        # for grad_name, grad in sum_grad_dict.items():
        #     sum_grad_dict[grad_name] /= N

        return grad_a_pre,grad_x_list,sum_grad_dict


    def temporal_affine_forward(self,parameters,a_list):
        """
        时序仿射层的前向算法

        :param parameters: 模型参数
        :param a_list: RNN 中间层的激活值 shape (N, T, n_h)  N- 样本个数 ,T-时刻长度 ,n_h-隐藏层的向量维度

        :return:  o_list :  shape (N, T, class_num) N- 样本个数, T-时刻长度, class_num-分类的个数 ;
                  y_ba_list : shape (N, T, class_num) ;
                  cache:  (parameters,o_list,a_list) 用于反向传播

        """
        V = parameters["V"] #  shape (class_num,n_h)
        b_o = parameters["b_o"] # shape (class_num,1)

        N, T, n_h = np.shape(a_list)
        class_num,_ = np.shape(b_o)

        a_list_flat = a_list.reshape(N * T, n_h)

        o = np.dot(V,a_list_flat.T) + b_o # shape (class_num, N * T)

        y_ba = Activation.softmax(o) # shape (class_num, N * T)

        o_list = o.T.reshape(N,T,class_num)
        y_ba_list = y_ba.T.reshape(N,T,class_num)

        cache = (parameters,o_list,a_list)

        return o_list,y_ba_list,cache


    @deprecated(version='1.0', reason="You should use another function")
    def temporal_affine_forward_deprecated(self,parameters,a_list):
        """
        时序仿射层的前向算法

        :param parameters: 模型参数
        :param a_list: RNN 中间层的激活值 shape (N, T, n_h)  N- 样本个数 ,T-时刻长度 ,n_h-隐藏层的向量维度

        :return:  o_list :  shape (N, T, class_num) N- 样本个数, T-时刻长度, class_num-分类的个数 ;
                  y_ba_list : shape (N, T, class_num) ;
                  cache:  (parameters,o_list,a_list) 用于反向传播

        """
        V = parameters["V"]
        b_o = parameters["b_o"]

        N, T, n_h = np.shape(a_list)
        class_num,_ = np.shape(b_o)

        o_list = np.zeros((N, T, class_num))

        y_ba_list = np.zeros((N,T,class_num)) # 样本标签估计值的时刻列表 shape (N,T,class_num)

        for t in range(T): # 遍历所有的时刻

            o = np.dot(V,a_list[:,t,:].T)+b_o # shape: (class_num,N)
            # V shape:(class_num,n_h) , a_list[:,t,:].T shape: (n_h,N) -> (class_num,N)

            o_list[:,t,:] = o.T

            y_ba = Activation.softmax(o) # shape: (class_num,N)
            # o shape: (class_num,N)

            y_ba_list[:,t,:] = y_ba.T

        cache = (parameters,o_list,a_list)

        return o_list,y_ba_list,cache


    def temporal_affine_bakward(self,grad_o_list,cache):
        """
        时序仿射层的后向算法

        :param grad_o_list:  shape (N,T,class_num) N- 样本个数 ,T-时刻长度 , class_num-分类的个数

        :param cache: 前向算法的缓存

        :return: outLayer_grad_a_list: 输出层传递给隐藏层的梯度, shape (N,T,n_h) ;
                 sum_grad: 所有时刻的梯度的加和
                            { "grad_V":grad_V, "grad_b_o":grad_b_o, "grad_o":grad_o }

        """
        N, T, class_num = np.shape(grad_o_list)

        parameters, o_list, a_list = cache

        n_h = np.shape(a_list)[2]

        # o_list  shape (N, T, class_num)
        # a_list  shape (N, T, n_h)

        V = parameters["V"] # shape (class_num,n_h)

        grad_o = grad_o_list.reshape( N*T, class_num )

        a_list_flat = a_list.reshape( N*T, n_h )

        outLayer_grad_a = np.dot(V.T, grad_o.T)  # shape(n_h,N*T)
        # V.T shape (n_h,class_num) , grad_o.T shape (class_num,N*T) -> shape shape(n_h,N*T)

        outLayer_grad_a_list = outLayer_grad_a.T.reshape(N,T,n_h)

        grad_V = np.dot(grad_o.T, a_list_flat)  # shape(class_num,n_h)
        # grad_o.T shape (class_num,N*T) , a_list_flat shape(N*T,n_h)

        grad_b_o = np.sum(grad_o.T, axis=1, keepdims=True)
        # grad_o.T shape (class_num,N*T)

        grad = {"grad_V": grad_V, "grad_b_o": grad_b_o}  # 需要更新的参数

        return outLayer_grad_a_list,grad


    @deprecated(version='1.0', reason="You should use another function")
    def temporal_affine_bakward_deprecated(self,grad_o_list,cache):
        """
        时序仿射层的后向算法

        :param grad_o_list:  shape (N,T,class_num) N- 样本个数 ,T-时刻长度 , class_num-分类的个数

        :param cache: 前向算法的缓存

        :return: outLayer_grad_a_list: 输出层传递给隐藏层的梯度, shape (N,T,n_h) ;
                 sum_grad: 所有时刻的梯度的加和
                            { "grad_V":grad_V, "grad_b_o":grad_b_o, "grad_o":grad_o }

        """
        N, T, class_num = np.shape(grad_o_list)

        parameters, o_list, a_list = cache
        V = parameters["V"]

        outLayer_grad_a_list = np.zeros(np.shape(a_list))
        grad_list = []

        for t in range(T):

            grad_o = grad_o_list[:,t,:].T  # shape (class_num,N)

            outLayer_grad_a = np.dot(V.T,grad_o) # shape(n_h,N)
            # V.T shape(n_h,class_num) , grad_o shape (class_num,N) -> shape shape(n_h,N)

            outLayer_grad_a_list[:,t,:] = outLayer_grad_a.T

            grad_V = np.dot(grad_o,a_list[:,t,:]) # shape(class_num,n_h)
            # grad_o shape (class_num,N) , a_list[t] shape(N,n_h)
            # grad_V /= N

            grad_b_o = np.sum(grad_o, axis=1, keepdims=True)
            # grad_b_o /= N

            grad = { "grad_V":grad_V, "grad_b_o":grad_b_o } # 需要更新的参数

            grad_list.append(grad)

        # 对所有时刻的梯度进行加和
        sum_grad_dict = defaultdict(float)

        for t in range(len(grad_list)):

            for grad_name,grad in grad_list[t].items():

                    sum_grad_dict[grad_name] += grad

        # for grad_name, grad in sum_grad_dict.items():
        #     sum_grad_dict[grad_name] /= N

        return outLayer_grad_a_list,sum_grad_dict


    @deprecated(version='1.0', reason="You should use another function")
    def multi_classify_loss_func_deprecated(self,o_list,y_onehot_list=None):
        """
        多分类下, 所有时序的损失函数的加和

        :param o_list: shape (N,T,class_num)
        :param y_onehot_list: 样本标签的时刻列表 shape (N,T,class_num) N- 样本个数 ,T-时刻长度 ,
                                                                     class_num-分类的个数

        :return:
            loss : 损失函数的加和
            y_ba_list: 样本标签估计值的时刻列表 shape (N,T,class_num)

        """

        N, T, class_num = np.shape(o_list)

        loss = 0

        for t in range(T): # 遍历所有的时刻

            loss += np.sum(-y_onehot_list[:,t,:].T * Activation.log_softmax(o_list[:,t,:].T))  # shape:(1,)
            # y_onehot_list[:,t,:].T shape (class_num,N) , o_list[:,t,:].T shape: (class_num,N)

        loss = loss / N

        return loss

    def multi_classify_loss_func(self,o_list,y_ba_list,y_onehot_list,mask=None):
        """
        多分类下, 所有时序的损失函数的加和,
        考虑被屏蔽的时间步

        :param o_list: shape (N,T,class_num)

        :param y_ba_list: shape (N,T,class_num)

        :param y_onehot_list: 样本标签的时刻列表 shape (N,T,class_num) N- 样本个数 ,T-时刻长度 ,
                                                                     class_num-分类的个数
        :param mask: 用于屏蔽被选择的时间步 shape(N,T) , 默认为 None

        :return:
            loss : 损失函数的加和
            grad_o_list:  shape (N,T,class_num)

        """

        N, T, class_num = np.shape(o_list)

        y_onehot_list_flat = y_onehot_list.reshape(N*T,class_num)
        o_list_flat = o_list.reshape(N*T,class_num)
        y_ba_list_flat = y_ba_list.reshape(N*T,class_num)

        if mask is None:
            mask = np.ones((N,T))

        mask_flat = mask.reshape(1,N*T)

        grad_o = mask_flat*(y_ba_list_flat.T - y_onehot_list_flat.T)  # shape (class_num,N*T)
        # o_list_flat.T shape (class_num,N*T) , y_onehot_list_flat.T shape (class_num,N*T)

        grad_o /= N

        grad_o_list = grad_o.T.reshape(N,T,class_num)

        loss = np.sum( mask_flat * (-y_onehot_list_flat.T) * Activation.log_softmax(o_list_flat.T))  # shape:(1,)
        # mask_flat shape (1,N*T), y_onehot_list_flat.T shape (class_num,N*T) , o_list_flat.T shape: (class_num,N*T)

        loss = loss / N

        return loss, grad_o_list


class UnitTest:
    """
    单元测试

    """

    def test_middle_layer_forward(self):

        N, m, n_h = 3, 10, 4

        x = np.linspace(-0.4, 0.7, num=N * m).reshape(N,m).T
        a_prev = np.linspace(-0.2, 0.5, num=N * n_h).reshape(N,n_h).T

        U = np.linspace(-0.1, 0.9, num=m * n_h).reshape(m,n_h).T
        W = np.linspace(-0.3, 0.7, num=n_h * n_h).reshape(n_h, n_h).T
        b_z = np.linspace(-0.2, 0.4, num=n_h).reshape(n_h, 1)

        parameters = {'U':U,'W':W,'b_z':b_z}

        rnn_layer = RNNLayer()

        activation = Activation.tanh

        next_h, _ = rnn_layer.middle_layer_forwoard(parameters=parameters,x=x,a_prev=a_prev,activation_func=activation)

        expected_next_h = np.asarray([
            [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
            [0.66854692, 0.79562378, 0.87755553, 0.92795967],
            [0.97934501, 0.99144213, 0.99646691, 0.99854353]]).T

        print('next_h error: ', rel_error(expected_next_h, next_h))


    def test_middle_layer_backward(self):

        np.random.seed(231)

        N, m, n_h = 4, 5, 6

        x = np.random.randn(N, m).T
        a_prev = np.random.randn(N, n_h).T

        U = np.random.randn(m, n_h).T
        W = np.random.randn(n_h, n_h).T
        b_z = np.random.randn(n_h,1)

        parameters = {'U': U, 'W': W, 'b_z': b_z}

        rnn_layer = RNNLayer()
        activation = Activation.tanh

        out, cache = rnn_layer.middle_layer_forwoard(parameters=parameters, x=x, a_prev=a_prev,
                                                    activation_func=activation)

        dnext_h = np.random.randn(*np.shape(out))

        fx = lambda x: rnn_layer.middle_layer_forwoard(parameters=parameters, x=x, a_prev=a_prev,
                                                    activation_func=activation)[0]
        fh = lambda a_prev: rnn_layer.middle_layer_forwoard(parameters=parameters, x=x, a_prev=a_prev,
                                                    activation_func=activation)[0]

        def fWx(U):

            tmp = parameters['U']
            parameters['U'] = U
            res = rnn_layer.middle_layer_forwoard(parameters=parameters, x=x, a_prev=a_prev,
                                                    activation_func=activation)[0]
            parameters['U'] = tmp
            return res

        def fWh(W):

            tmp = parameters['W']
            parameters['W'] = W
            res = rnn_layer.middle_layer_forwoard(parameters=parameters, x=x, a_prev=a_prev,
                                                    activation_func=activation)[0]
            parameters['W'] = tmp
            return res

        def fb(b_z):

            tmp = parameters['b_z']
            parameters['b_z'] = b_z
            res = rnn_layer.middle_layer_forwoard(parameters=parameters, x=x, a_prev=a_prev,
                                                    activation_func=activation)[0]
            parameters['b_z'] = tmp
            return res


        dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
        dprev_h_num = eval_numerical_gradient_array(fh, a_prev, dnext_h)

        dWx_num = eval_numerical_gradient_array(fWx, U, dnext_h)

        dWh_num = eval_numerical_gradient_array(fWh, W, dnext_h)
        db_num = eval_numerical_gradient_array(fb, b_z, dnext_h)


        dprev_h, dx, grad_dic = rnn_layer.middle_layer_bakwoard(grad_a=dnext_h, cache=cache, grad_activation_func=Activation.grad_tanh)
        # grad_dic = {"grad_U": grad_U, "grad_W": grad_W, "grad_b_z": grad_b_z}

        print('dx error: ', rel_error(dx_num, dx))
        print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
        print('dWx error: ', rel_error(dWx_num, grad_dic['grad_U']))
        print('dWh error: ', rel_error(dWh_num, grad_dic['grad_W']))
        print('db error: ', rel_error(db_num, grad_dic['grad_b_z']))


    def test_middle_forwoard_propagation(self):

        N, T, m, n_h = 2, 3, 4, 5

        x = np.linspace(-0.1, 0.3, num=N * T * m).reshape(N, T, m)
        h0 = np.linspace(-0.3, 0.1, num=N * n_h).reshape(N, n_h).T

        U = np.linspace(-0.2, 0.4, num=m * n_h).reshape(m, n_h).T
        W = np.linspace(-0.4, 0.1, num=n_h * n_h).reshape(n_h, n_h).T
        b_z = np.linspace(-0.7, 0.1, num=n_h).reshape(n_h, 1)

        parameters = {'U': U, 'W': W, 'b_z': b_z}

        rnn_layer = RNNLayer()

        a_list,_ =rnn_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, a_init=h0)


        expected_h = np.asarray([
            [
                [-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
                [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
                [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525],
            ],
            [
                [-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
                [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
                [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043]]])

        print('h error: ', rel_error(expected_h, a_list))


    def test_middle_bakwoard_propagation(self):

        np.random.seed(231)

        N, m, T, n_h = 2, 3, 10, 5

        x = np.random.randn(N, T, m)
        h0 = np.random.randn(N, n_h).T

        U = np.random.randn(m, n_h).T
        W = np.random.randn(n_h, n_h).T
        b_z = np.random.randn(n_h,1)

        parameters = {'U': U, 'W': W, 'b_z': b_z}

        rnn_layer = RNNLayer()

        out,cache =rnn_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, a_init=h0)

        dout = np.random.randn(*out.shape)


        fx = lambda x: rnn_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, a_init=h0)[0]
        fh0 = lambda h0: rnn_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, a_init=h0)[0]

        def fWx(U):

            tmp = parameters['U']
            parameters['U'] = U
            res = rnn_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, a_init=h0)[0]
            parameters['U'] = tmp
            return res

        def fWh(W):

            tmp = parameters['W']
            parameters['W'] = W
            res = rnn_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, a_init=h0)[0]
            parameters['W'] = tmp
            return res

        def fb(b_z):

            tmp = parameters['b_z']
            parameters['b_z'] = b_z
            res = rnn_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, a_init=h0)[0]
            parameters['b_z'] = tmp
            return res


        dh0, dx, grad_dict_middle = rnn_layer.middle_bakwoard_propagation(outLayer_grad_a_list=dout, cache_list=cache)


        dx_num = eval_numerical_gradient_array(fx, x, dout)
        dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
        dWx_num = eval_numerical_gradient_array(fWx, U, dout)
        dWh_num = eval_numerical_gradient_array(fWh, W, dout)
        db_num = eval_numerical_gradient_array(fb, b_z, dout)

        print('dx error: ', rel_error(dx_num, dx))
        print('dh0 error: ', rel_error(dh0_num, dh0))
        print('dWx error: ', rel_error(dWx_num, grad_dict_middle['grad_U']))
        print('dWh error: ', rel_error(dWh_num,  grad_dict_middle['grad_W']))
        print('db error: ', rel_error(db_num,  grad_dict_middle['grad_b_z']))


    def test_word_embedding_forward(self):

        N, T, vocab_size, m = 2, 4, 5, 3

        x = np.asarray([[0, 3, 1, 2],
                        [2, 1, 0, 3]])

        W_embed = np.linspace(0, 1, num=vocab_size * m).reshape(vocab_size, m)

        parameters = {"W_embed":W_embed}

        rnn_layer = RNNLayer()

        out, _ = rnn_layer.word_embedding_forward(parameters=parameters, batch_sentence=x)

        expected_out = np.asarray([
            [[0., 0.07142857, 0.14285714],
             [0.64285714, 0.71428571, 0.78571429],
             [0.21428571, 0.28571429, 0.35714286],
             [0.42857143, 0.5, 0.57142857]],
            [[0.42857143, 0.5, 0.57142857],
             [0.21428571, 0.28571429, 0.35714286],
             [0., 0.07142857, 0.14285714],
             [0.64285714, 0.71428571, 0.78571429]]])

        print('out error: ', rel_error(expected_out, out))

    def test_word_embedding_backward(self):

        np.random.seed(231)

        N, T, vocab_size, m = 50, 3, 5, 6

        x = np.random.randint(vocab_size, size=(N, T))
        W_embed = np.random.randn(vocab_size, m)

        parameters = {"W_embed":W_embed}

        rnn_layer = RNNLayer()

        out, cache = rnn_layer.word_embedding_forward(parameters=parameters, batch_sentence=x)

        dout = np.random.randn(*out.shape)

        def f(W_embed):

            tmp = parameters['W_embed']
            parameters['W_embed'] = W_embed
            res = rnn_layer.word_embedding_forward(parameters=parameters, batch_sentence=x)[0]
            parameters['W_embed'] = tmp
            return res

        grad_dict_embed = rnn_layer.word_embedding_backward(grad_x_list=dout, cache=cache)

        dW_num = eval_numerical_gradient_array(f, W_embed, dout)

        print('dW error: ', rel_error(grad_dict_embed['grad_W_embed'], dW_num))


    def test_temporal_affine_bakward(self):
        np.random.seed(231)

        # Gradient check for temporal affine layer
        N, T, n_h, class_num = 2, 3, 4, 5

        x = np.random.randn(N, T, n_h)

        V = np.random.randn(n_h, class_num).T
        b_o = np.random.randn(class_num,1)

        parameters = {"V": V,"b_o":b_o}

        # out, cache = temporal_affine_forward(x, w, b)

        rnn_layer = RNNLayer()

        out,_,cache = rnn_layer.temporal_affine_forward_deprecated(parameters=parameters, a_list=x)

        dout = np.random.randn(*out.shape)

        fx = lambda x: rnn_layer.temporal_affine_forward_deprecated(parameters=parameters, a_list=x)[0]

        def fw(V):

            tmp = parameters['V']
            parameters['V'] = V
            res = rnn_layer.temporal_affine_forward_deprecated(parameters=parameters, a_list=x)[0]
            parameters['V'] = tmp
            return res

        def fb(b_o):

            tmp = parameters['b_o']
            parameters['b_o'] = b_o
            res = rnn_layer.temporal_affine_forward_deprecated(parameters=parameters, a_list=x)[0]
            parameters['b_o'] = tmp
            return res

        dx_num = eval_numerical_gradient_array(fx, x, dout)

        dw_num = eval_numerical_gradient_array(fw, V, dout)
        db_num = eval_numerical_gradient_array(fb, b_o, dout)

        dx, grad_dict_out = rnn_layer.temporal_affine_bakward_deprecated(dout, cache)

        print('dx error: ', rel_error(dx_num, dx))
        print('dw error: ', rel_error(dw_num, grad_dict_out["grad_V"]))
        print('db error: ', rel_error(db_num, grad_dict_out["grad_b_o"]))

    def test_multi_classify_loss_func(self):

        np.random.seed(231)

        rnn_layer = RNNLayer()

        def check_loss(N, T, class_num, p):

            o_list = 0.001 * np.random.randn(N, T, class_num)

            y_ba_list = 0.001 * np.random.randn(N, T, class_num)

            y = np.random.randint(class_num, size=(N, T))
            y_one_hot = Utils.convert_to_one_hot(x=y,class_num=class_num)  # shape (N,T,class_num)

            mask = np.random.rand(N, T) <= p
            print(rnn_layer.multi_classify_loss_func(o_list=o_list,y_ba_list=y_ba_list,y_onehot_list=y_one_hot,mask=mask)[0])

        check_loss(100, 1, 10, 1.0)  # Should be about 2.3
        check_loss(100, 10, 10, 1.0)  # Should be about 23
        check_loss(5000, 10, 10, 0.1)  # Should be about 2.3

        # Gradient check for temporal softmax loss
        N, T, class_num = 7, 8, 9

        o_list = np.random.randn(N, T, class_num)

        # 计算 y_ba_list
        o_list_flat = o_list.reshape(N*T,class_num)
        y_ba_flat = Activation.softmax(o_list_flat.T) # shape (class_num,N*T)
        y_ba_list = y_ba_flat.T.reshape(N,T,class_num)

        y = np.random.randint(class_num, size=(N, T))
        y_one_hot = Utils.convert_to_one_hot(x=y, class_num=class_num)  # shape (N,T,class_num)

        mask = (np.random.rand(N, T) > 0.5)

        loss, dx = rnn_layer.multi_classify_loss_func(o_list=o_list,y_ba_list=y_ba_list,y_onehot_list=y_one_hot,mask=mask)

        dx_num = eval_numerical_gradient(lambda x: rnn_layer.multi_classify_loss_func(o_list=o_list,y_ba_list=y_ba_list,y_onehot_list=y_one_hot,mask=mask)[0], o_list, verbose=False)

        print('dx error: ', rel_error(dx, dx_num))


if __name__ == '__main__':

    test = UnitTest()

    # test.test_middle_layer_forward()

    # test.test_middle_layer_backward()

    # test.test_middle_forwoard_propagation()

    # test.test_middle_bakwoard_propagation()

    # test.test_word_embedding_forward()

    # test.test_word_embedding_backward()

    # test.test_temporal_affine_bakward()

    test.test_multi_classify_loss_func()