#!/usr/bin/python
# -*- coding: UTF-8 -*-


class TreeEvaluator:

    @staticmethod
    def ThresholdL1(w, alpha):
        """
        考虑 L1 正则化

        :param w:
        :param alpha:
        :return:
        """

        if w > alpha:
            return w - alpha

        if w < - alpha:
            return w + alpha

        return 0.0

    @staticmethod
    def CalcWeight(params, stats):
        """
        依据 xgboost 论文中的公式(5) 计算节点的权重

        :param params:
        :param stats:
        :return:
        """

        if stats.sum_hess < params['min_child_weight'] or stats.sum_hess <= 0.0:
            return 0.0

        dw = - TreeEvaluator.ThresholdL1(stats.sum_grad, params["reg_alpha"]) / (stats.sum_hess + params["reg_lambda"])

        if params['max_delta_step'] != 0 and abs(dw) > params['max_delta_step']:
            dw = params['max_delta_step']

        return dw

    @staticmethod
    def CalcLossGivenWeight(params, stats, w):
        """
        依据 xgboost 论文中的公式(6) 计算节点的 损失(目标)函数的值

        :param params:
        :param stats:
        :param w:
        :return:
        """

        if stats.sum_hess <= 0:
            return 0.0

        # CalcGainGivenWeight can significantly reduce avg floating point error.
        # 均值溢出问题:
        # eg. (a+b)/2 不如 a+ (b-a)/2 , 因为 a+b 可能溢出

        if params['max_delta_step'] == 0:
            return (TreeEvaluator.ThresholdL1(stats.sum_grad, params["reg_alpha"])) ** 2 / (
                        stats.sum_hess + params["reg_lambda"])

        return -((2.0) * stats.sum_grad * w + (stats.sum_hess + params["reg_lambda"]) * (w ** 2))

    @staticmethod
    def CalcLoss(params, stats):
        """
        依据 xgboost 论文中的公式(6) 计算节点的 损失(目标)函数的值

        :param params:
        :param stats:
        :return:
        """
        if stats.sum_hess <= 0:
            return 0.0

        return (TreeEvaluator.ThresholdL1(stats.sum_grad, params["reg_alpha"])) ** 2 / (
                stats.sum_hess + params["reg_lambda"])


    @staticmethod
    def CalcGain(params, root_loss, left_stats, right_stats):
        """
        依据 xgboost 论文中的公式(7), 计算切分后的增益

        :param params:
        :param root_loss:
        :param left_stats:
        :param right_stats:
        :return:
        """

        gain = 0.5*(TreeEvaluator.CalcLoss(params, left_stats) + TreeEvaluator.CalcLoss(params, right_stats)
                    - root_loss)

        if gain <= params['gama']:
            gain = 0

        return gain