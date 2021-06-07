
class SplitEntry:

    def __init__(self):

        # 节点分裂的增益loss变化值
        self.loss_chg=0.0

        # 分裂特征的标号
        self.sindex=0

        # 分裂的特征值
        self.split_value=0.0



class GradeStats:
    """
    梯度 统计信息

    """

    def __init__(self):

        # 一阶梯度的和
        self.sum_grad=0

        # 二阶梯度的和
        self.sum_hess=0
