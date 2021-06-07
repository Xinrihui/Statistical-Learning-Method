
import math

class TreeEvaluator:


    @staticmethod
    def ThresholdL1( w,  alpha):

        if w >  alpha:
            return w - alpha

        if w < - alpha:
            return w + alpha

        return  0.0

    @staticmethod
    def CalcWeight(params,stats):

        if  stats.sum_hess < params['min_child_weight'] or  stats.sum_hess <= 0.0:

            return 0.0

        dw = - TreeEvaluator.ThresholdL1(stats.sum_grad, params["reg_alpha"]) / (stats.sum_hess + params["reg_lambda"])

        if params['max_delta_step'] != 0 and  abs(dw) > params['max_delta_step'] :

            dw = params['max_delta_step']

        return dw

    @staticmethod
    def CalcGainGivenWeight(params, stats,w):

        if stats.sum_hess <= 0 :
            return 0.0

        # CalcGainGivenWeight can significantly reduce avg floating point error.
        # 均值溢出问题:
        # eg. (a+b)/2 不如 a+ (b-a)/2 , 因为 a+b 可能溢出

        if params['max_delta_step'] == 0 :
            return (TreeEvaluator.ThresholdL1(stats.sum_grad, params["reg_alpha"]))**2 / (stats.sum_hess + params["reg_lambda"])


        return -( (2.0) * stats.sum_grad * w + (stats.sum_hess + params["reg_lambda"]) * (w**2) )


    @staticmethod
    def CalcGain(params,stats):

        return TreeEvaluator.CalcGainGivenWeight(params,stats,TreeEvaluator.CalcWeight(params,stats) )
