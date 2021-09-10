#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

import math

import time

class Optimizer:
    """
    优化算法

    class BGDOptimizer 批量梯度下降(BGD)
    class MinBatchOptimizer  Mini-batch 梯度下降
    class MomentumOptimizer 带动量(Momentum)的 Mini-batch 梯度下降
    class AdamOptimizer  Adam Mini-batch 梯度下降

    Author: xrh
    Date: 2021-07-14

    """


    def __init__(self, optim_config):
        """

        :param optim_config: 优化算法的超参数

        optim_config={
        learning_rate: 学习率
        }

        """
        self.learning_rate = optim_config.setdefault('learning_rate', 0.01)



    def update_parameter(self, param , grad_param,optim_config):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param param: 模型参数
        :param grad_param: 模型参数的梯度
        :param optim_config: 超参数
        :return:
        """
        pass

class BGDOptimizer(Optimizer):


    def update_parameter(self, param , grad_param,optim_param):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param param: 模型参数
        :param grad_param: 模型参数的梯度
        :param optim_config: 字典类型, 优化算法的超参数
        :return:
        """

        param_name = param[0]
        param_value = param[1]

        param_value -= self.learning_rate * grad_param

        return param_value



class MinBatchOptimizer(Optimizer):

    def update_parameter(self, param , grad_param,optim_param):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param param: 模型参数
        :param grad_param: 模型参数的梯度
        :param optim_config: 字典类型, 优化算法的超参数
        :return:
        """

        param_name = param[0]
        param_value = param[1]

        param_value -= self.learning_rate * grad_param

        return param_value

class MomentumOptimizer(MinBatchOptimizer):

    def __init__(self, optim_config):
        """

        :param optim_config: 优化算法的超参数
        optim_config={

        learning_rate: 学习率
        beta1: 相当于定义了计算指数加权平均数时的窗口大小

        }

        eg.
        beta1=0.5  窗口大小为: 1/(1-beta1) = 2
        beta1=0.9  窗口大小为: 1/(1-beta1) = 10
        beta1=0.99 窗口大小为: 1/(1-beta1) = 100

        """

        self.learning_rate = optim_config.setdefault('learning_rate', 0.01)
        self.beta1 = optim_config.setdefault('beta1', 0.9)


    def update_parameter(self, param ,grad_param,optim_param):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param param: 模型参数
        :param grad_param: 模型参数的梯度
        :param optim_param: 优化器的参数

        :return:
        """

        param_name = param[0]
        param_value = param[1]

        v_param = optim_param.setdefault("v_"+str(param_name), np.zeros(np.shape(param_value))) # v_param 初始化为 0

        v_param = self.beta1 * v_param + (1 - self.beta1) * grad_param

        optim_param["v_"+str(param_name)] = v_param # 更新优化器的参数

        param_value -= self.learning_rate * v_param

        return param_value


class AdamOptimizer(MinBatchOptimizer):

    def __init__(self, optim_config):
        """

        :param optim_config: 优化算法的超参数
        optim_config={

        learning_rate: 学习率
        beta1: 惯性保持, 历史梯度和当前梯度的平均 (默认 0.9)
        beta2: 环境感知, 为不同的模型参数产生自适应的学习率 (默认 0.99)
               beta1, beta1 一般无需调节
        epsilon: 一个很小的数
        bias_correct: 是否使用偏差修正

        }

        eg.
        beta1=0.5  历史梯度的窗口大小为: 1/(1-beta1) = 2
        beta1=0.9  历史梯度的窗口大小为: 1/(1-beta1) = 10
        beta1=0.99 历史梯度的窗口大小为: 1/(1-beta1) = 100

        """

        self.learning_rate = optim_config.setdefault('learning_rate', 1e-3)
        self.beta1 = optim_config.setdefault('beta1', 0.9)
        self.beta2 = optim_config.setdefault('beta2', 0.999)
        self.epsilon = optim_config.setdefault('epsilon',1e-8)

        self.bias_correct = optim_config.setdefault('bias_correct',False)



    def update_parameter(self,param, grad_param, optim_param):
        """
        根据反向传播计算得到梯度信息 更新 模型参数

        :param param: 模型参数, (param_name,param_value)
        :param grad_param: 模型参数的梯度
        :param optim_param: 优化器的参数

        :return:
        """
        param_name = param[0]
        param_value = param[1]

        m_param = optim_param.setdefault("m_" + str(param_name),  np.zeros(np.shape(param_value))) # m_param 初始化为 0
        v_param = optim_param.setdefault("v_"+str(param_name), np.zeros(np.shape(param_value))) # v_param 初始化为 0
        t_param = optim_param.setdefault("t_" + str(param_name), 0)

        # 一阶矩
        m_param = self.beta1 * m_param + (1 - self.beta1) * grad_param
        # 二阶矩
        v_param = self.beta2 * v_param + (1 - self.beta2) * np.square(grad_param)

        # 更新优化器的参数
        optim_param["m_" + str(param_name)] = m_param
        optim_param["v_"+str(param_name)] = v_param

        # 使用偏差修正
        if self.bias_correct:

            alpha = self.learning_rate * np.sqrt(1 - self.beta2 ** t_param) / (1 - self.beta1 ** t_param)

        else:
            alpha = self.learning_rate

        # 更新优化器的参数
        optim_param["t_" + str(param_name)] = (t_param + 1)

        param_value -= ((alpha * m_param) / np.sqrt(v_param+self.epsilon))


        return param_value





