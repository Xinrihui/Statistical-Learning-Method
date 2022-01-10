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

    def middle_layer_forwoard(self, parameters, x, C_prev, h_prev, activation_func='tanh'):
        """
        t时刻的前向传播算法, 只有中间层, 不包括输出层

        样本特征的维度为 n_m, 隐藏层的维度为 n_h

        :param parameters:模型参数
        :param x: t时刻的样本特征 shape (m,N)  N-样本个数 , m-特征维度
        :param C_prev: t-1时刻的细胞状态值 shape (n_h,N)
        :param h_prev: t-1时刻的隐藏状态值  shape (n_h,N)
        :param activation_func: 选择的激活函数,

        :return: C,h,cache
        cache =  (x, C_prev, h_prev, C_t, h_t, f_t, i_t, i_t, o_t, parameters)

        """
        # 遗忘门参数
        W_f = parameters["W_f"]  # shape(n_h, n_h)
        U_f = parameters["U_f"]  # shape(n_h,m)
        b_f = parameters["b_f"]  # shape(n_h,1)

        # 输入门参数
        W_i = parameters["W_i"]  # shape(n_h, n_h)
        U_i = parameters["U_i"]  # shape(n_h,m)
        b_i = parameters["b_i"]  # shape(n_h,1)

        # 状态更新参数
        W_a = parameters["W_a"]  # shape(n_h, n_h)
        U_a = parameters["U_a"]  # shape(n_h,m)
        b_a = parameters["b_a"]  # shape(n_h,1)

        # 输出门参数
        W_o = parameters["W_o"]  # shape(n_h, n_h)
        U_o = parameters["U_o"]  # shape(n_h,m)
        b_o = parameters["b_o"]  # shape(n_h,1)

        # 遗忘门 forget gate
        f_t = Activation.sigmoid(np.dot(W_f, h_prev) + np.dot(U_f, x) + b_f)  # shape (n_h,N)
        # W_f shape(n_h, n_h), h_prev shape (n_h,N) -> shape (n_h,N)
        # U_f shape(n_h, m), x shape (m,N) -> shape (n_h,N)

        # 输入门 input gate
        i_t = Activation.sigmoid(np.dot(W_i, h_prev) + np.dot(U_i, x) + b_i)  # shape (n_h,N)

        # 细胞状态更新 cell state update
        a_t = Activation.tanh(np.dot(W_a, h_prev) + np.dot(U_a, x) + b_a)  # shape (n_h,N)

        C_t = f_t * C_prev + i_t * a_t  # shape (n_h,N)
        # f_t shape (n_h,N), C_prev shape (n_h,N)

        # 输出门 output gate
        o_t = Activation.sigmoid(np.dot(W_o, h_prev) + np.dot(U_o, x) + b_o)  # shape (n_h,N)

        # 隐藏状态更新 hidden state update
        tanh_C_t = np.tanh(C_t)
        h_t = o_t * tanh_C_t  # shape (n_h,N)

        cache = (x, C_prev, h_prev, C_t, tanh_C_t, h_t, f_t, i_t, a_t, o_t, parameters)

        return C_t, h_t, cache

    def middle_layer_bakwoard(self, grad_h_in, grad_C_in, cache, grad_activation_func='grad_tanh'):
        """
        t时刻中间层的后向传播算法

        :param grad_h_in:  shape: (n_h,N)
                输入当前时刻的 h 的梯度由两部分组成：
                (1) 由输出层传播而来
                (2) 由下一时刻传播而来

        :param grad_C_in:  shape: (n_h,N)
                输入当前时刻的 C 的梯度由下一时刻传播而来

        :param cache: 前向传播 t时刻的缓存

        :param grad_activation_func: 激活函数的一阶导数
        :return:
            grad_h_pre : 传递给前一个时刻的梯度

            grad_C_pre : 传递给前一个时刻的梯度

            grad_x: 传递给 词向量的梯度

            grad_dict: 之后要更新的模型参数的梯度


        """

        (x, C_prev, h_prev, C_t, tanh_C_t, h_t, f_t, i_t, a_t, o_t, parameters) = cache

        N = np.shape(x)[1]

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

        # 损失函数对当前隐状态 ht 的偏导数
        grad_h = grad_h_in

        # 损失函数对当前细胞状态 Ct 的偏导数
        grad_C = o_t * (1 - tanh_C_t**2) * grad_h + grad_C_in

        # 遗忘门
        grad_z_f = grad_C * C_prev * f_t * (1 - f_t)  # shape (n_h,N)

        grad_W_f = np.dot(grad_z_f, h_prev.T)  # shape (n_h,n_h)
        # grad_z_f shape (n_h,N) , h_prev.T shape (N,n_h) -> shape (n_h,n_h)
        # grad_W_f /= N

        grad_U_f = np.dot(grad_z_f, x.T)  # shape (n_h,m)
        # grad_z_f shape (n_h,N) , x.T shape (N,m) -> shape (n_h,m)
        # grad_U_f /= N

        grad_b_f = np.sum(grad_z_f, axis=1, keepdims=True)  # shape:(n_h,1)
        # grad_b_f /= N

        # 输入门
        grad_z_i = grad_C * a_t * i_t * (1 - i_t)

        grad_W_i = np.dot(grad_z_i, h_prev.T)  # shape (n_h,n_h)
        # grad_z_i shape (n_h,N) , h_prev.T shape (N,n_h) -> shape (n_h,n_h)
        # grad_W_i /= N

        grad_U_i = np.dot(grad_z_i, x.T)  # shape (n_h,m)
        # grad_z_f shape (n_h,N) , x.T shape (N,m) -> shape (n_h,m)
        # grad_U_i /= N

        grad_b_i = np.sum(grad_z_i, axis=1, keepdims=True)  # shape:(n_h,1)
        # grad_b_i /= N

        # 细胞状态更新
        grad_z_a = grad_C * i_t * (1 - a_t ** 2)

        grad_W_a = np.dot(grad_z_a, h_prev.T)  # shape (n_h,n_h)
        # grad_z_a shape (n_h,N) , h_prev.T shape (N,n_h) -> shape (n_h,n_h)
        # grad_W_a /= N

        grad_U_a = np.dot(grad_z_a, x.T)  # shape (n_h,m)
        # grad_z_a shape (n_h,N) , x.T shape (N,m) -> shape (n_h,m)
        # grad_U_a /= N

        grad_b_a = np.sum(grad_z_a, axis=1, keepdims=True)  # shape:(n_h,1)
        # grad_b_a /= N

        # 输出门
        grad_z_o = grad_h * tanh_C_t * o_t * (1 - o_t)

        grad_W_o = np.dot(grad_z_o, h_prev.T)  # shape (n_h,n_h)
        # grad_z_o shape (n_h,N) , h_prev.T shape (N,n_h) -> shape (n_h,n_h)
        # grad_W_o /= N

        grad_U_o = np.dot(grad_z_o, x.T)  # shape (n_h,m)
        # grad_z_o shape (n_h,N) , x.T shape (N,m) -> shape (n_h,m)
        # grad_U_o /= N

        grad_b_o = np.sum(grad_z_o, axis=1, keepdims=True)  # shape:(n_h,1)
        # grad_b_o /= N

        # 计算传给前一个时刻的梯度
        grad_h_pre = np.dot(W_o.T, grad_z_o) + np.dot(W_f.T, grad_z_f) + np.dot(W_i.T, grad_z_i) + np.dot(W_a.T,
                                                                                                          grad_z_a)

        grad_C_pre = grad_C * f_t

        # 计算传给词嵌入层的梯度
        grad_x = np.dot(U_o.T, grad_z_o) + np.dot(U_f.T, grad_z_f) + np.dot(U_i.T, grad_z_i) + np.dot(U_a.T, grad_z_a)

        # 将需要更新的模型参数的梯度包装为 dict
        grad_dict = {"grad_W_f": grad_W_f, "grad_U_f": grad_U_f, "grad_b_f": grad_b_f,
                    "grad_W_i": grad_W_i, "grad_U_i": grad_U_i, "grad_b_i": grad_b_i,
                    "grad_W_a": grad_W_a, "grad_U_a": grad_U_a, "grad_b_a": grad_b_a,
                    "grad_W_o": grad_W_o, "grad_U_o": grad_U_o, "grad_b_o": grad_b_o,
                    }

        return grad_h_pre, grad_C_pre, grad_x, grad_dict

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

        :return: h_list : 隐藏状态向量的列表 shape: (N,T,n_h) ;
                 cache_list : 所有时刻的 cache 列表, 用于反向传播

        """
        N, T, m = np.shape(x_list)

        n_h = np.shape(parameters['W_f'])[0]  # 隐藏层的维度

        if C_init is None:
            C_init = np.zeros((n_h, N))

        if h_init is None:
            h_init = np.zeros((n_h, N))

        if not hasattr(Activation, activation):
            raise ValueError('Invalid activation func "%s"' % activation)

        activation_func = getattr(Activation, activation)  # 激活函数

        cache_list = []

        h_list = np.zeros((N, T, n_h))
        C_list = np.zeros((N, T, n_h))

        C_prev = C_init
        h_prev = h_init

        # t=0,1,...,T-1
        for t in range(T):

            C_t, h_t, cache = self.middle_layer_forwoard(parameters=parameters, x=x_list[:, t, :].T, C_prev=C_prev,
                                                              h_prev=h_prev, activation_func=activation_func)
            # h_t shape: (n_h,N)

            h_list[:, t, :] = h_t.T  # h_t.T shape: (N,n_h)
            C_list[:, t, :] = C_t.T  # C_t.T shape: (N,n_h)
            cache_list.append(cache)

            C_prev = C_t
            h_prev = h_t

        return C_list, h_list, cache_list

    def middle_bakwoard_propagation(self, outLayer_grad_h_list, cache_list, activation='tanh'):

        """
         LSTM 中间层的后向传播算法

        :param outLayer_grad_h_list:  输出层对隐藏层的向量 h的一阶梯度的列表 shape (N,T,n_h)

        :param cache_list: 中间层的前向传播算法的缓存
                      cache = (x,a_prev,z,a,parameters)
        :param activation: 激活函数
        :return:

            grad_h_pre : 第 1 时间步的隐藏层向量的梯度
            grad_x_list: 词向量的梯度列表 shape (N,T,m)
            sum_grad : 所有时刻的梯度加和

        """

        N, T, n_h = np.shape(outLayer_grad_h_list)

        x = cache_list[0][0]
        m = np.shape(x)[0]

        if not hasattr(Activation, activation):
            raise ValueError('Invalid activation func "%s"' % activation)

        activation_func = getattr(Activation, activation)  # 激活函数
        grad_activation_func = getattr(Activation, 'grad_' + activation)  # 激活函数的一阶导数

        grad_list = []
        grad_x_list = np.zeros((N, T, m))

        # 1.最后一个时刻 t=T-1
        t = T - 1

        grad_C_pre = np.zeros((n_h, N))

        grad_h_pre, grad_C_pre, grad_x, grad = self.middle_layer_bakwoard(grad_h_in=outLayer_grad_h_list[:, t, :].T,
                                                                          grad_C_in=grad_C_pre,
                                                                          cache=cache_list[t],
                                                                          grad_activation_func=grad_activation_func)
        # 最后一个时刻的 h 的梯度 grad_h 只由输出层传递而来

        grad_x_list[:, t, :] = grad_x.T

        grad_list.append(grad)

        # 2.中间和初始时刻
        for t in range(T - 2, -1, -1):  # t = T-2,...,0

            grad_h_pre, grad_C_pre, grad_x, grad = self.middle_layer_bakwoard(
                grad_h_in=outLayer_grad_h_list[:, t, :].T + grad_h_pre,
                grad_C_in=grad_C_pre,
                cache=cache_list[t],
                grad_activation_func=grad_activation_func)
            # 当前时刻的 h 的梯度 grad_h 由两部分组成:
            #  一部分由输出层传播而来 outLayer_grad_h_list[t] , 另一部分由下一时刻传播而来 grad["grad_h_pre"]

            grad_x_list[:, t, :] = grad_x.T

            grad_list.append(grad)

        # 对所有时刻的梯度进行加和
        sum_grad_dict = defaultdict(float)

        for t in range(len(grad_list)):

            for grad_name, grad in grad_list[t].items():
                sum_grad_dict[grad_name] += grad

        return grad_h_pre, grad_x_list, sum_grad_dict



class UnitTest:
    """
    单元测试

    """

    def test_middle_layer_backward(self):
        np.random.seed(231)

        N, m, n_h = 4, 5, 6

        x = np.random.randn(m, N)

        C_prev = np.random.randn(n_h, N)
        h_prev = np.random.randn(n_h, N)

        U_f = np.random.randn(n_h, m)
        W_f = np.random.randn(n_h, n_h)
        b_f = np.random.randn(n_h, 1)

        U_i = np.random.randn(n_h, m)
        W_i = np.random.randn(n_h, n_h)
        b_i = np.random.randn(n_h, 1)

        U_a = np.random.randn(n_h, m)
        W_a = np.random.randn(n_h, n_h)
        b_a = np.random.randn(n_h, 1)

        U_o = np.random.randn(n_h, m)
        W_o = np.random.randn(n_h, n_h)
        b_o = np.random.randn(n_h, 1)

        parameters = {
            "U_f": U_f, "W_f": W_f,
            "U_i": U_i, "W_i": W_i,
            "U_a": U_a, "W_a": W_a,
            "U_o": U_o, "W_o": W_o,
            "b_f": b_f, "b_i": b_i, "b_a": b_a, "b_o": b_o,
        }

        lstm_layer = LSTMLayer()

        # 正向传播
        C_t, h_t, cache = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev,
                                                                h_prev=h_prev)

        # 下一层传给当前层的梯度
        grad_C = np.random.randn(*np.shape(C_t))
        grad_h = np.random.randn(*np.shape(h_t))

        # 反向传播取得梯度
        grad_h_pre, grad_C_pre, grad_x, grad_dict = lstm_layer.middle_layer_bakwoard(grad_h, grad_C, cache)

        # 通过数值方法取得梯度
        fx_C = lambda x: lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
        fh_C = lambda h_prev: \
        lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
        fC_C = lambda C_prev: \
        lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]

        # ------------#

        fx_h = lambda x: lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
        fh_h = lambda h_prev: \
        lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
        fC_h = lambda C_prev: \
        lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]


        def fUf_C(U_f):
            tmp = parameters['U_f']
            parameters['U_f'] = U_f
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['U_f'] = tmp
            return res

        def fWf_C(W_f):
            tmp = parameters['W_f']
            parameters['W_f'] = W_f
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['W_f'] = tmp
            return res

        def fbf_C(b_f):
            tmp = parameters['b_f']
            parameters['b_f'] = b_f
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['b_f'] = tmp
            return res


        def fUi_C(U_i):
            tmp = parameters['U_i']
            parameters['U_i'] = U_i
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['U_i'] = tmp
            return res

        def fWi_C(W_i):
            tmp = parameters['W_i']
            parameters['W_i'] = W_i
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['W_i'] = tmp
            return res

        def fbi_C(b_i):
            tmp = parameters['b_i']
            parameters['b_i'] = b_i
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['b_i'] = tmp
            return res

        def fUa_C(U_a):
            tmp = parameters['U_a']
            parameters['U_a'] = U_a
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['U_a'] = tmp
            return res

        def fWa_C(W_a):
            tmp = parameters['W_a']
            parameters['W_a'] = W_a
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['W_a'] = tmp
            return res

        def fba_C(b_a):
            tmp = parameters['b_a']
            parameters['b_a'] = b_a
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['b_a'] = tmp
            return res

        def fUo_C(U_o):
            tmp = parameters['U_o']
            parameters['U_o'] = U_o
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['U_o'] = tmp
            return res

        def fWo_C(W_o):
            tmp = parameters['W_o']
            parameters['W_o'] = W_o
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['W_o'] = tmp
            return res

        def fbo_C(b_o):
            tmp = parameters['b_o']
            parameters['b_o'] = b_o
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[0]
            parameters['b_o'] = tmp
            return res

        # ------------#

        def fUf_h(U_f):
            tmp = parameters['U_f']
            parameters['U_f'] = U_f
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['U_f'] = tmp
            return res

        def fWf_h(W_f):
            tmp = parameters['W_f']
            parameters['W_f'] = W_f
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['W_f'] = tmp
            return res

        def fbf_h(b_f):
            tmp = parameters['b_f']
            parameters['b_f'] = b_f
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['b_f'] = tmp
            return res


        def fUi_h(U_i):
            tmp = parameters['U_i']
            parameters['U_i'] = U_i
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['U_i'] = tmp
            return res

        def fWi_h(W_i):
            tmp = parameters['W_i']
            parameters['W_i'] = W_i
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['W_i'] = tmp
            return res

        def fbi_h(b_i):
            tmp = parameters['b_i']
            parameters['b_i'] = b_i
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['b_i'] = tmp
            return res

        def fUa_h(U_a):
            tmp = parameters['U_a']
            parameters['U_a'] = U_a
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['U_a'] = tmp
            return res

        def fWa_h(W_a):
            tmp = parameters['W_a']
            parameters['W_a'] = W_a
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['W_a'] = tmp
            return res

        def fba_h(b_a):
            tmp = parameters['b_a']
            parameters['b_a'] = b_a
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['b_a'] = tmp
            return res

        def fUo_h(U_o):
            tmp = parameters['U_o']
            parameters['U_o'] = U_o
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['U_o'] = tmp
            return res

        def fWo_h(W_o):
            tmp = parameters['W_o']
            parameters['W_o'] = W_o
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['W_o'] = tmp
            return res

        def fbo_h(b_o):
            tmp = parameters['b_o']
            parameters['b_o'] = b_o
            res = lstm_layer.middle_layer_forwoard(parameters=parameters, x=x, C_prev=C_prev, h_prev=h_prev)[1]
            parameters['b_o'] = tmp
            return res

        grad_num_x = eval_numerical_gradient_array(fx_C, x, grad_C)+eval_numerical_gradient_array(fx_h, x, grad_h)
        grad_num_h = eval_numerical_gradient_array(fh_C, h_prev, grad_C)+eval_numerical_gradient_array(fh_h, h_prev, grad_h)
        grad_num_C = eval_numerical_gradient_array(fC_C, C_prev, grad_C)+eval_numerical_gradient_array(fC_h, C_prev, grad_h)

        grad_num_Uf = eval_numerical_gradient_array(fUf_C, U_f, grad_C)+eval_numerical_gradient_array(fUf_h, U_f, grad_h)
        grad_num_Wf = eval_numerical_gradient_array(fWf_C, W_f, grad_C)+eval_numerical_gradient_array(fWf_h, W_f, grad_h)
        grad_num_bf = eval_numerical_gradient_array(fbf_C, b_f, grad_C)+eval_numerical_gradient_array(fbf_h, b_f, grad_h)

        grad_num_Ui = eval_numerical_gradient_array(fUi_C, U_i, grad_C)+eval_numerical_gradient_array(fUi_h, U_i, grad_h)
        grad_num_Wi = eval_numerical_gradient_array(fWi_C, W_i, grad_C)+eval_numerical_gradient_array(fWi_h, W_i, grad_h)
        grad_num_bi = eval_numerical_gradient_array(fbi_C, b_i, grad_C)+eval_numerical_gradient_array(fbi_h, b_i, grad_h)

        grad_num_Ua = eval_numerical_gradient_array(fUa_C, U_a, grad_C)+eval_numerical_gradient_array(fUa_h, U_a, grad_h)
        grad_num_Wa = eval_numerical_gradient_array(fWa_C, W_a, grad_C)+eval_numerical_gradient_array(fWa_h, W_a, grad_h)
        grad_num_ba = eval_numerical_gradient_array(fba_C, b_a, grad_C)+eval_numerical_gradient_array(fba_h, b_a, grad_h)
        
        grad_num_Uo = eval_numerical_gradient_array(fUo_C, U_o, grad_C)+eval_numerical_gradient_array(fUo_h, U_o, grad_h)
        grad_num_Wo = eval_numerical_gradient_array(fWo_C, W_o, grad_C)+eval_numerical_gradient_array(fWo_h, W_o, grad_h)
        grad_num_bo = eval_numerical_gradient_array(fbo_C, b_o, grad_C)+eval_numerical_gradient_array(fbo_h, b_o, grad_h)

        # grad_dic = {"grad_W_f": grad_W_f, "grad_U_f": grad_U_f, "grad_b_f": grad_b_f,
        #             "grad_W_i": grad_W_i, "grad_U_i": grad_U_i, "grad_b_i": grad_b_i,
        #             "grad_W_a": grad_W_a, "grad_U_a": grad_U_a, "grad_b_a": grad_b_a,
        #             "grad_W_o": grad_W_o, "grad_U_o": grad_U_o, "grad_b_o": grad_b_o,
        #             }
        
        print('grad_x error: ', rel_error(grad_num_x, grad_x))
        print('grad_h_pre error: ', rel_error(grad_num_h, grad_h_pre))
        print('grad_C_pre error: ', rel_error(grad_num_C, grad_C_pre))

        print('grad_Uf error: ', rel_error(grad_num_Uf, grad_dict['grad_U_f']))
        print('grad_Wf error: ', rel_error(grad_num_Wf, grad_dict['grad_W_f']))
        print('grad_bf error: ', rel_error(grad_num_bf, grad_dict['grad_b_f']))

        print('grad_Ui error: ', rel_error(grad_num_Ui, grad_dict['grad_U_i']))
        print('grad_Wi error: ', rel_error(grad_num_Wi, grad_dict['grad_W_i']))
        print('grad_bi error: ', rel_error(grad_num_bi, grad_dict['grad_b_i']))

        print('grad_Ua error: ', rel_error(grad_num_Ua, grad_dict['grad_U_a']))
        print('grad_Wa error: ', rel_error(grad_num_Wa, grad_dict['grad_W_a']))
        print('grad_ba error: ', rel_error(grad_num_ba, grad_dict['grad_b_a']))

        print('grad_Uo error: ', rel_error(grad_num_Uo, grad_dict['grad_U_o']))
        print('grad_Wo error: ', rel_error(grad_num_Wo, grad_dict['grad_W_o']))
        print('grad_bo error: ', rel_error(grad_num_bo, grad_dict['grad_b_o']))


    def test_middle_bakwoard_propagation(self):
        np.random.seed(231)

        N, m, T, n_h = 2, 3, 10, 5

        x = np.random.randn(N, T, m)

        C0 = np.random.randn(n_h, N)
        h0 = np.random.randn(n_h, N)

        U_f = np.random.randn(n_h, m)
        W_f = np.random.randn(n_h, n_h)
        b_f = np.random.randn(n_h, 1)

        U_i = np.random.randn(n_h, m)
        W_i = np.random.randn(n_h, n_h)
        b_i = np.random.randn(n_h, 1)

        U_a = np.random.randn(n_h, m)
        W_a = np.random.randn(n_h, n_h)
        b_a = np.random.randn(n_h, 1)

        U_o = np.random.randn(n_h, m)
        W_o = np.random.randn(n_h, n_h)
        b_o = np.random.randn(n_h, 1)

        parameters = {
            "U_f": U_f, "W_f": W_f,
            "U_i": U_i, "W_i": W_i,
            "U_a": U_a, "W_a": W_a,
            "U_o": U_o, "W_o": W_o,
            "b_f": b_f, "b_i": b_i, "b_a": b_a, "b_o": b_o,
        }

        lstm_layer = LSTMLayer()

        # 正向传播
        C_list, h_list, cache_list = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)

        #下一层传给当前层的梯度
        outLayer_grad_h_list = np.random.randn(*h_list.shape)

        # 通过反向传播取得梯度
        grad_h_pre, grad_x_list, grad_dict = lstm_layer.middle_bakwoard_propagation(outLayer_grad_h_list=outLayer_grad_h_list, cache_list=cache_list)

        # 通过数值方法取得梯度
        fx = lambda x: lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]

        fh0 = lambda h0: \
            lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]

        fC0 = lambda C0: \
            lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]

        def fUf(U_f):
            tmp = parameters['U_f']
            parameters['U_f'] = U_f
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['U_f'] = tmp
            return res

        def fWf(W_f):
            tmp = parameters['W_f']
            parameters['W_f'] = W_f
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['W_f'] = tmp
            return res

        def fbf(b_f):
            tmp = parameters['b_f']
            parameters['b_f'] = b_f
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['b_f'] = tmp
            return res


        def fUi(U_i):
            tmp = parameters['U_i']
            parameters['U_i'] = U_i
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['U_i'] = tmp
            return res

        def fWi(W_i):
            tmp = parameters['W_i']
            parameters['W_i'] = W_i
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['W_i'] = tmp
            return res

        def fbi(b_i):
            tmp = parameters['b_i']
            parameters['b_i'] = b_i
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['b_i'] = tmp
            return res

        def fUa(U_a):
            tmp = parameters['U_a']
            parameters['U_a'] = U_a
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['U_a'] = tmp
            return res

        def fWa(W_a):
            tmp = parameters['W_a']
            parameters['W_a'] = W_a
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['W_a'] = tmp
            return res

        def fba(b_a):
            tmp = parameters['b_a']
            parameters['b_a'] = b_a
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['b_a'] = tmp
            return res

        def fUo(U_o):
            tmp = parameters['U_o']
            parameters['U_o'] = U_o
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['U_o'] = tmp
            return res

        def fWo(W_o):
            tmp = parameters['W_o']
            parameters['W_o'] = W_o
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['W_o'] = tmp
            return res

        def fbo(b_o):
            tmp = parameters['b_o']
            parameters['b_o'] = b_o
            res = lstm_layer.middle_forwoard_propagation(parameters=parameters, x_list=x, C_init=C0, h_init=h0)[0]
            parameters['b_o'] = tmp
            return res

        grad_num_x = eval_numerical_gradient_array(fx, x, outLayer_grad_h_list)

        grad_num_h0 = eval_numerical_gradient_array(fh0, h0, outLayer_grad_h_list)

        grad_num_Uf = eval_numerical_gradient_array(fUf, U_f, outLayer_grad_h_list)

        grad_num_Wf = eval_numerical_gradient_array(fWf, W_f, outLayer_grad_h_list)

        grad_num_bf = eval_numerical_gradient_array(fbf, b_f, outLayer_grad_h_list)

        grad_num_Ui = eval_numerical_gradient_array(fUi, U_i, outLayer_grad_h_list)

        grad_num_Wi = eval_numerical_gradient_array(fWi, W_i, outLayer_grad_h_list)

        grad_num_bi = eval_numerical_gradient_array(fbi, b_i, outLayer_grad_h_list)

        grad_num_Ua = eval_numerical_gradient_array(fUa, U_a, outLayer_grad_h_list)

        grad_num_Wa = eval_numerical_gradient_array(fWa, W_a, outLayer_grad_h_list)

        grad_num_ba = eval_numerical_gradient_array(fba, b_a, outLayer_grad_h_list)

        grad_num_Uo = eval_numerical_gradient_array(fUo, U_o, outLayer_grad_h_list)

        grad_num_Wo = eval_numerical_gradient_array(fWo, W_o, outLayer_grad_h_list)

        grad_num_bo = eval_numerical_gradient_array(fbo, b_o, outLayer_grad_h_list)


        # grad_dic = {"grad_W_f": grad_W_f, "grad_U_f": grad_U_f, "grad_b_f": grad_b_f,
        #             "grad_W_i": grad_W_i, "grad_U_i": grad_U_i, "grad_b_i": grad_b_i,
        #             "grad_W_a": grad_W_a, "grad_U_a": grad_U_a, "grad_b_a": grad_b_a,
        #             "grad_W_o": grad_W_o, "grad_U_o": grad_U_o, "grad_b_o": grad_b_o,
        #             }

        print('grad_x_list error: ', rel_error(grad_num_x, grad_x_list))
        print('grad_h_pre error: ', rel_error(grad_num_h0, grad_h_pre))

        print('grad_Uf error: ', rel_error(grad_num_Uf, grad_dict['grad_U_f']))
        print('grad_Wf error: ', rel_error(grad_num_Wf, grad_dict['grad_W_f']))
        print('grad_bf error: ', rel_error(grad_num_bf, grad_dict['grad_b_f']))

        print('grad_Ui error: ', rel_error(grad_num_Ui, grad_dict['grad_U_i']))
        print('grad_Wi error: ', rel_error(grad_num_Wi, grad_dict['grad_W_i']))
        print('grad_bi error: ', rel_error(grad_num_bi, grad_dict['grad_b_i']))

        print('grad_Ua error: ', rel_error(grad_num_Ua, grad_dict['grad_U_a']))
        print('grad_Wa error: ', rel_error(grad_num_Wa, grad_dict['grad_W_a']))
        print('grad_ba error: ', rel_error(grad_num_ba, grad_dict['grad_b_a']))

        print('grad_Uo error: ', rel_error(grad_num_Uo, grad_dict['grad_U_o']))
        print('grad_Wo error: ', rel_error(grad_num_Wo, grad_dict['grad_W_o']))
        print('grad_bo error: ', rel_error(grad_num_bo, grad_dict['grad_b_o']))



    def test_temporal_affine_bakward(self):
        np.random.seed(231)

        # Gradient check for temporal affine layer
        N, T, n_h, class_num = 2, 3, 4, 5

        x = np.random.randn(N, T, n_h)

        V = np.random.randn(n_h, class_num).T
        b_y = np.random.randn(class_num, 1)

        parameters = {"V": V, "b_y": b_y}

        lstm_layer = LSTMLayer()

        # 正向传播
        out, _, cache = lstm_layer.temporal_affine_forward(parameters=parameters, h_list=x)

        # 下一层传给当前层的梯度
        dout = np.random.randn(*out.shape)

        # 通过反向传播取得梯度
        outLayer_grad_h_list, grad_dict = lstm_layer.temporal_affine_bakward(grad_z_y_list=dout, cache=cache)

        # 通过数值方法取得梯度
        fx = lambda x: lstm_layer.temporal_affine_forward(parameters=parameters, h_list=x)[0]

        def fV(V):
            tmp = parameters['V']
            parameters['V'] = V
            res = lstm_layer.temporal_affine_forward(parameters=parameters, h_list=x)[0]
            parameters['V'] = tmp
            return res

        def fby(b_y):
            tmp = parameters['b_y']
            parameters['b_y'] = b_y
            res = lstm_layer.temporal_affine_forward(parameters=parameters, h_list=x)[0]
            parameters['b_y'] = tmp
            return res

        grad_num_x = eval_numerical_gradient_array(fx, x, dout)
        grad_num_V = eval_numerical_gradient_array(fV, V, dout)
        grad_num_by = eval_numerical_gradient_array(fby, b_y, dout)

        print('outLayer_grad_h_list error: ', rel_error(grad_num_x, outLayer_grad_h_list))
        print('grad_V error: ', rel_error(grad_num_V, grad_dict['grad_V']))
        print('grad_Wf error: ', rel_error(grad_num_by, grad_dict['grad_b_y']))


    def test_multi_classify_loss_func(self):

        np.random.seed(231)

        lstm_layer = LSTMLayer()

        def check_loss(N, T, class_num, p):
            o_list = 0.001 * np.random.randn(N, T, class_num)

            y_ba_list = 0.001 * np.random.randn(N, T, class_num)

            y = np.random.randint(class_num, size=(N, T))
            y_one_hot = ArrayUtils.one_hot_array(x=y, class_num=class_num)  # shape (N,T,class_num)

            mask = np.random.rand(N, T) <= p
            print(lstm_layer.multi_classify_loss_func(z_y_list=o_list, y_ba_list=y_ba_list, y_onehot_list=y_one_hot,
                                                     mask=mask)[0])

        check_loss(100, 1, 10, 1.0)  # Should be about 2.3
        check_loss(100, 10, 10, 1.0)  # Should be about 23
        check_loss(5000, 10, 10, 0.1)  # Should be about 2.3

        # Gradient check for temporal softmax loss
        N, T, class_num = 7, 8, 9

        o_list = np.random.randn(N, T, class_num)

        # 计算 y_ba_list
        o_list_flat = o_list.reshape(N * T, class_num)
        y_ba_flat = Activation.softmax(o_list_flat.T)  # shape (class_num,N*T)
        y_ba_list = y_ba_flat.T.reshape(N, T, class_num)

        y = np.random.randint(class_num, size=(N, T))
        y_one_hot = ArrayUtils.one_hot_array(x=y, class_num=class_num)  # shape (N,T,class_num)

        mask = (np.random.rand(N, T) > 0.5)

        loss, dx = lstm_layer.multi_classify_loss_func(z_y_list=o_list, y_ba_list=y_ba_list, y_onehot_list=y_one_hot,
                                                      mask=mask)

        dx_num = eval_numerical_gradient(lambda x:
                                         lstm_layer.multi_classify_loss_func(z_y_list=o_list, y_ba_list=y_ba_list,
                                                                            y_onehot_list=y_one_hot, mask=mask)[0],
                                         o_list, verbose=False)

        print('dx error: ', rel_error(dx, dx_num))


if __name__ == '__main__':

    test = UnitTest()

    # test.test_middle_layer_backward()

    # test.test_middle_bakwoard_propagation()

    # test.test_temporal_affine_bakward()

    # test.test_multi_classify_loss_func()
