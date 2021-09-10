#!/usr/bin/python
# -*- coding: UTF-8 -*-

from deprecated import deprecated

import numpy as np

from activation_xrh import Activation

from gradient_check_xrh import *

from collections import *

from utils_xrh import *

from rnn_layers_xrh import *

class LSTMLayer(RNNLayer):
    """
    LSTM 的相关层

    (1) LSTM 中间层的前向传播
        LSTM 中间层的反向传播

    (2) 时序仿射层(输出层)的正向传播
        时序仿射层的反向传播

    (3) 损失函数层

    Author: xrh
    Date: 2021-08-21

    ref:
    https://www.cnblogs.com/pinard/p/6519110.html

    """


    def middle_layer_forwoard(self,parameters,x,C_prev,h_prev,activation_func='tanh'):
        """
        t时刻的前向传播算法, 只有中间层, 不包括输出层

        样本特征的维度为 n_m, 隐藏层的维度为 n_h

        :param parameters:模型参数
        :param x: t时刻的样本特征 shape (m,N)  N-样本个数 , m-特征维度
        :param C_prev: t-1时刻的细胞状态值 shape (n_h,N)
        :param h_prev: t-1时刻的隐藏层的激活值  shape (n_h,N)
        :param activation_func: 选择的激活函数,

        :return: C,h,cache
        cache =  (x, C_prev, h_prev, C_t, h_t, f_t, i_t, i_t, o_t, parameters)

        """
        # 遗忘门参数
        W_f = parameters["W_f"]
        U_f = parameters["U_f"]
        b_f = parameters["b_f"]

        # 输入门参数
        W_i = parameters["W_i"]
        U_i = parameters["U_i"]
        b_i = parameters["b_i"]

        # 状态更新参数
        W_a = parameters["W_a"]
        U_a = parameters["U_a"]
        b_a = parameters["b_a"]

        # 输出门参数
        W_o = parameters["W_o"]
        U_o = parameters["U_o"]
        b_o = parameters["b_o"]

        # 遗忘门 forget gate
        f_t = Activation.sigmoid(np.dot(W_f, h_prev) + np.dot(U_f, x) + b_f) # shape (n_h,N)

        # 输入门 input gate
        i_t = Activation.sigmoid(np.dot(W_i, h_prev) + np.dot(U_i, x) + b_i) # shape (n_h,N)

        # 细胞状态更新 cell state update
        a_t = Activation.tanh(np.dot(W_a, h_prev) + np.dot(U_a, x) + b_a) # shape (n_h,N)
        C_t = f_t * C_prev + i_t * a_t # shape (n_h,N)

        # 输出门 output gate
        o_t = Activation.sigmoid(np.dot(W_o, h_prev) + np.dot(U_o, x) + b_o) # shape (n_h,N)

        # 隐藏状态更新 hidden state update
        h_t = o_t * np.tanh(C_t) # shape (n_h,N)

        cache = (x, C_prev, h_prev, C_t, h_t, f_t, i_t, a_t, o_t, parameters)

        return o_t, C_t, h_t, cache


    def middle_layer_bakwoard(self,grad_h,grad_C,cache,grad_activation_func='grad_tanh'):
        """
        t时刻中间层的后向传播算法

        :param grad_h:   shape: (n_h,N)
                当前时刻的 h 的梯度由两部分组成：
                (1) 由输出层传播而来
                (2) 由下一时刻传播而来

        :param grad_C:   shape: (n_h,N)
                当前时刻的 C 的梯度由两部分组成：
                (1) 由输出层传播而来
                (2) 由下一时刻传播而来

        :param cache: 前向传播 t时刻的缓存

        :param grad_activation_func: 激活函数的一阶导数
        :return:
            grad_h_pre : 传递给前一个时刻的梯度

            grad_C_pre : 传递给前一个时刻的梯度

            grad_x: 传递给 词向量的梯度

            grad_dic: 之后要更新的模型参数的梯度


        """

        (x, C_prev, h_prev, C_t, h_t, f_t, i_t, a_t, o_t, parameters) = cache

        # 遗忘门参数
        W_f = parameters["W_f"]
        U_f = parameters["U_f"]

        # 输入门参数
        W_i = parameters["W_i"]
        U_i = parameters["U_i"]

        # 状态更新参数
        W_a = parameters["W_a"]
        U_a = parameters["U_a"]

        # 输出门参数
        W_o = parameters["W_o"]
        U_o = parameters["U_o"]

        # 遗忘门
        grad_z_f = grad_C * C_prev * f_t * (1-f_t) # shape (n_h,N)

        grad_W_f = np.dot(grad_z_f,h_t.T) # shape (n_h,n_h)
        # grad_z_f shape (n_h,N) , h_t.T shape (N,n_h) -> shape (n_h,n_h)

        grad_U_f = np.dot(grad_z_f,x.T) # shape (n_h,m)
        # grad_z_f shape (n_h,N) , x.T shape (N,m) -> shape (n_h,m)

        grad_b_f = np.sum(grad_z_f, axis=1, keepdims=True)  # shape:(n_h,1)

        # 输入门
        grad_z_i = grad_C * a_t * i_t * (1-i_t)

        grad_W_i = np.dot(grad_z_i,h_t.T) # shape (n_h,n_h)
        # grad_z_i shape (n_h,N) , h_t.T shape (N,n_h) -> shape (n_h,n_h)

        grad_U_i = np.dot(grad_z_i,x.T) # shape (n_h,m)
        # grad_z_f shape (n_h,N) , x.T shape (N,m) -> shape (n_h,m)

        grad_b_i = np.sum(grad_z_i, axis=1, keepdims=True)  # shape:(n_h,1)

        # 细胞状态更新
        grad_z_a = grad_C * i_t * (1-a_t**2)

        grad_W_a = np.dot(grad_z_a,h_t.T) # shape (n_h,n_h)
        # grad_z_a shape (n_h,N) , h_t.T shape (N,n_h) -> shape (n_h,n_h)

        grad_U_a = np.dot(grad_z_a,x.T) # shape (n_h,m)
        # grad_z_a shape (n_h,N) , x.T shape (N,m) -> shape (n_h,m)

        grad_b_a = np.sum(grad_z_a, axis=1, keepdims=True)  # shape:(n_h,1)

        # 输出门
        grad_z_o = grad_h * np.tanh(C_t) * o_t * (1-o_t)

        grad_W_o = np.dot(grad_z_o,h_t.T) # shape (n_h,n_h)
        # grad_z_o shape (n_h,N) , h_t.T shape (N,n_h) -> shape (n_h,n_h)

        grad_U_o = np.dot(grad_z_o,x.T) # shape (n_h,m)
        # grad_z_o shape (n_h,N) , x.T shape (N,m) -> shape (n_h,m)

        grad_b_o = np.sum(grad_z_o, axis=1, keepdims=True)  # shape:(n_h,1)

        # 计算传给前一个时刻的梯度
        grad_h_pre = np.dot(W_o.T, grad_z_o) + np.dot(W_f.T, grad_z_f) + np.dot(W_i.T, grad_z_i) + np.dot(W_a.T, grad_z_a)

        grad_C_pre = grad_C * f_t

        # 计算传给词嵌入层的梯度
        grad_x = np.dot(U_o.T, grad_z_o) + np.dot(U_f.T, grad_z_f) + np.dot(U_i.T, grad_z_i) + np.dot(U_a.T,grad_z_a)

        # 将需要更新的模型参数的梯度包装为 dict
        grad_dic = {"grad_W_f":grad_W_f, "grad_U_f":grad_U_f, "grad_b_f":grad_b_f,
                    "grad_W_i": grad_W_i, "grad_U_i": grad_U_i, "grad_b_i": grad_b_i,
                    "grad_W_a": grad_W_a, "grad_U_a": grad_U_a, "grad_b_a": grad_b_a,
                    "grad_W_o": grad_W_o, "grad_U_o": grad_U_o, "grad_b_o": grad_b_o,
                    }

        return grad_h_pre,grad_C_pre,grad_x,grad_dic

    def middle_forwoard_propagation(self,
                             parameters,
                             x_list,
                             C_init=None,
                             h_init=None,
                             activation='tanh'
                             ):
        """
        LSTM 中间层的前向传播算法

        :param parameters: 模型参数
        :param x_list: 样本特征的时刻列表 shape (N,T,m) N- 样本个数 ,T-时刻长度 ,m-特征维度

        :param C_init 细胞状态的初始化向量 shape (n_h,N)

        :param h_init 隐藏层的初始化向量, shape (n_h,N)

        :param activation: 激活函数

        :return: C_list : 细胞状态的向量列表 shape: (N,T,n_h) ;
                 h_list : 隐藏层的向量列表 shape: (N,T,n_h) ;
                 cache_list : 所有时刻的 cache 列表, 用于反向传播

        """
        N, T, m = np.shape(x_list)

        n_h = np.shape(parameters['W_f'])[0]  # 隐藏层的维度

        if C_init is None:
            C_init = np.zeros((n_h,N))

        if h_init is None:
            h_init = np.zeros((n_h,N))


        if not hasattr(Activation , activation):
            raise ValueError('Invalid activation func "%s"' % activation)

        activation_func = getattr(Activation, activation) # 激活函数

        cache_list=[]

        o_list = np.zeros((N, T, n_h))
        C_list = np.zeros((N, T, n_h))
        h_list = np.zeros((N, T, n_h))

        C_prev = C_init
        h_prev = h_init

        # t=0,1,...,T-1
        for t in range(T):

            o_t, C_t, h_t, cache = self.middle_layer_forwoard(parameters=parameters, x=x_list[:,t,:].T, C_prev=C_prev, h_prev=h_prev, activation_func=activation_func)
            # h_t shape: (n_h,N)

            o_list[:, t, :] = o_t.T  # o_t.T shape: (N,n_h)
            C_list[:, t, :] = C_t.T  # C_t.T shape: (N,n_h)
            h_list[:,t,:] = h_t.T # h_t.T shape: (N,n_h)
            cache_list.append(cache)

            C_prev = C_t
            h_prev = h_t

        return o_list, C_list, h_list, cache_list

    def middle_bakwoard_propagation(self, outLayer_grad_C_list, outLayer_grad_h_list, cache_list,activation='tanh'):

        """
         LSTM 中间层的后向传播算法

        :param outLayer_grad_C_list:  输出层对细胞状态向量 C的一阶梯度的列表 shape (N,T,n_h)
        :param outLayer_grad_h_list:  输出层对隐藏层的向量 h的一阶梯度的列表 shape (N,T,n_h)

        :param cache_list: 中间层的前向传播算法的缓存
                      cache = (x,a_prev,z,a,parameters)
        :param activation: 激活函数
        :return:

            grad_h_pre : 第 1 时间步的隐藏层向量的梯度
            grad_x_list: 词向量的梯度列表 shape (N,T,m)
            sum_grad : 所有时刻的梯度加和

        """

        N, T, n_h = np.shape(outLayer_grad_C_list)

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

        grad_h_pre, grad_C_pre,grad_x,grad = self.middle_layer_bakwoard( grad_h=outLayer_grad_h_list[:,t,:].T,grad_C=outLayer_grad_C_list[:,t,:].T, cache=cache_list[t],
                                      grad_activation_func=grad_activation_func)
        # 最后一个时刻的 h 的梯度 grad_h 只由输出层传递而来


        grad_x_list[:,t,:] = grad_x.T

        grad_list.append(grad)

        # 2.中间和初始时刻
        for t in range(T-2,-1,-1): # t = T-2,...,0

            grad_h_pre, grad_C_pre,grad_x,grad = self.middle_layer_bakwoard(grad_h=outLayer_grad_h_list[:,t,:].T + grad_h_pre,grad_C=outLayer_grad_C_list[:,t,:].T + grad_C_pre, cache=cache_list[t],
                                                     grad_activation_func=grad_activation_func)
            # 当前时刻的 h 的梯度 grad_h 由两部分组成:
            #  一部分由输出层传播而来 outLayer_grad_h_list[t] , 另一部分由下一时刻传播而来 grad["grad_h_pre"]

            grad_x_list[:, t, :] = grad_x.T

            grad_list.append(grad)

        # 对所有时刻的梯度进行加和
        sum_grad_dict = defaultdict(float)

        for t in range(len(grad_list)):

            for grad_name,grad in grad_list[t].items():

                    sum_grad_dict[grad_name] += grad


        return grad_h_pre,grad_x_list,sum_grad_dict


    def temporal_affine_forward(self,parameters,o_list,C_list,h_list):
        """
        时序仿射层的前向算法

        :param parameters: 模型参数

        :param C_list: RNN 细胞状态的向量列表 shape (N, T, n_h) N- 样本个数 ,T-时刻长度 ,n_h-隐藏层的向量维度
        :param h_list: RNN 隐藏层的向量列表 shape (N, T, n_h)  N- 样本个数 ,T-时刻长度 ,n_h-隐藏层的向量维度

        :return:  o_list :  shape (N, T, class_num) N- 样本个数, T-时刻长度, class_num-分类的个数 ;
                  y_ba_list : shape (N, T, class_num) ;
                  cache:  (parameters,z_y_list,o_list,C_list,h_list) 用于反向传播

        """
        V = parameters["V"] #  shape (class_num,n_h)
        b_y = parameters["b_y"] # shape (class_num,1)

        N, T, n_h = np.shape(h_list)
        class_num,_ = np.shape(b_y)

        h_list_flat = h_list.reshape(N * T, n_h)

        z_y = np.dot(V,h_list_flat.T) + b_y # shape (class_num, N * T)

        y_ba = Activation.softmax(z_y) # shape (class_num, N * T)

        z_y_list = z_y.T.reshape(N,T,class_num)
        y_ba_list = y_ba.T.reshape(N,T,class_num)

        cache = (parameters,z_y_list,o_list,C_list,h_list)

        return z_y_list,y_ba_list,cache



    def temporal_affine_bakward(self,grad_z_y_list,cache):
        """
        时序仿射层的后向算法

        :param grad_z_y_list:  shape (N,T,class_num) N- 样本个数 ,T-时刻长度 , class_num-分类的个数

        :param cache: 前向算法的缓存

        :return: outLayer_grad_a_list: 输出层传递给隐藏层的梯度, shape (N,T,n_h) ;
                 sum_grad: 所有时刻的梯度的加和
                            { "grad_V":grad_V, "grad_b_o":grad_b_o, "grad_o":grad_o }

        """
        N, T, class_num = np.shape(grad_z_y_list)

        (parameters,z_y_list,o_list,C_list,h_list) = cache

        n_h = np.shape(h_list)[2]
        # z_y_list  shape (N, T, class_num)
        # h_list  shape (N, T, n_h)

        V = parameters["V"] # shape (class_num,n_h)

        grad_z_y = grad_z_y_list.reshape( N*T, class_num )

        h_list_flat = h_list.reshape( N*T, n_h )
        C_list_flat = C_list.reshape(N * T, n_h)
        o_list_flat = o_list.reshape(N * T, n_h)

        outLayer_grad_h = np.dot(V.T, grad_z_y.T)  # shape(n_h,N*T)
        # V.T shape (n_h,class_num) , grad_z_y.T shape (class_num,N*T) -> shape shape(n_h,N*T)

        outLayer_grad_C = o_list_flat * (1 - (np.tanh(C_list_flat))**2) * outLayer_grad_h.T # shape (N * T, n_h)
        # o_list_flat shape (N * T, n_h) , outLayer_grad_h.T shape (N * T, n_h)

        outLayer_grad_h_list = outLayer_grad_h.T.reshape(N,T,n_h)

        outLayer_grad_C_list = outLayer_grad_C.reshape(N,T,n_h)

        grad_V = np.dot(grad_z_y.T, h_list_flat)  # shape(class_num,n_h)
        # grad_z_y.T shape (class_num,N*T) , h_list_flat shape(N*T,n_h)

        grad_b_y = np.sum(grad_z_y.T, axis=1, keepdims=True)
        # grad_y.T shape (class_num,N*T)

        grad = {"grad_V": grad_V, "grad_b_y": grad_b_y}  # 需要更新的参数

        return outLayer_grad_C_list, outLayer_grad_h_list, grad



    def multi_classify_loss_func(self,z_y_list,y_ba_list,y_onehot_list,mask=None):
        """
        多分类下, 所有时序的损失函数的加和,
        考虑被屏蔽的时间步

        :param z_y_list: shape (N,T,class_num)

        :param y_ba_list: shape (N,T,class_num)

        :param y_onehot_list: 样本标签的时刻列表 shape (N,T,class_num) N- 样本个数 ,T-时刻长度 ,
                                                                     class_num-分类的个数
        :param mask: 用于屏蔽被选择的时间步 shape(N,T) , 默认为 None

        :return:
            loss : 损失函数的加和
            grad_o_list:  shape (N,T,class_num)

        """

        N, T, class_num = np.shape(z_y_list)

        y_onehot_list_flat = y_onehot_list.reshape(N*T,class_num)
        z_y_list_flat = z_y_list.reshape(N*T,class_num)
        y_ba_list_flat = y_ba_list.reshape(N*T,class_num)

        if mask is None:
            mask = np.ones((N,T))

        mask_flat = mask.reshape(1,N*T)

        grad_z_y = mask_flat*(y_ba_list_flat.T - y_onehot_list_flat.T)  # shape (class_num,N*T)
        # y_ba_list_flat.T shape (class_num,N*T) , y_onehot_list_flat.T shape (class_num,N*T)

        grad_z_y /= N

        grad_z_y_list = grad_z_y.T.reshape(N,T,class_num)

        loss = np.sum( mask_flat * (-y_onehot_list_flat.T) * Activation.log_softmax(z_y_list_flat.T))  # shape:(1,)
        # mask_flat shape (1,N*T), y_onehot_list_flat.T shape (class_num,N*T) , z_y_list_flat.T shape: (class_num,N*T)

        loss = loss / N

        return loss, grad_z_y_list



