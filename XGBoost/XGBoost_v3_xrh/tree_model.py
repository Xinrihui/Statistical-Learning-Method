
# 树节点
class Node:
    def __init__(self, label=None, feature=None, feature_split=None, prev_feature=None, prev_feature_split=None,
                 loss=None,
                 sample_N=None,
                 childs=None):

        self.label = label  # 叶子节点才有标签

        self.feature = feature  # 非叶子节点, 划分 子节点的特征
        self.feature_split = feature_split  # 划分 子节点的特征 的切分点

        self.prev_feature = prev_feature
        self.prev_feature_split = prev_feature_split

        self.loss=loss
        self.sample_N=sample_N

        self.childs = childs
