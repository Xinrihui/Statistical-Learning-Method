#!/usr/bin/python
# -*- coding: UTF-8 -*-

import time

import pickle
import codecs
import numpy as np
from scipy import optimize

# 将 warnings 也当做错误抛出
# import warnings
# warnings.filterwarnings('error')

# 定义负无穷大
infinite = float('-inf')

class LinearCRF(object):
    """

    线性链条件随机场

    1. 适用于中文分词任务, 参考 CRF++, 设计如下特征函数:

    (1) 状态特征, 一元特征(Unigram)
    ('U', pos, word, tag)

    (2) 转移特征, 二元特征(Bigram)
    ('B', pre_tag, tag)

    2.实现了学习算法:
      (1)带正则的梯度上升算法
      (2) 拟牛顿法 L-BGFS ( 调用 scipy.optimize)

    3.实现在已有的预训练的模型的基础上进行更进一步的训练(使用同一个数据集)

    4.实现了 基于维特比算法的 解码

    5.如果觉得 LinearCRF模型训练的慢, 可以读取 CRF++ 库训练得到的参数, 详见 类 CRFSegmentation


    ref:
    1.《统计学习方法 第二版》李航
    2. https://victorjiangxin.github.io/Chinese-Word-Segmentation/
    3. https://www.cnblogs.com/Determined22/p/6915730.html

    Author: xrh
    Date: 2021-06-26

    """
    def __init__(self):

        self.ntags = 4
        self.index_tag = {0:'B', 1:'I', 2:'E', 3:'S'} # 分词场景的 4种隐状态
        self.tag_index = {'B':0, 'I':1, 'E':2, 'S':3}

        self.tag_sets = [0, 1, 2, 3]

        self.start_tag = 'start' # 补充的句子开始状态
        self.end_tag = 'end' # 补充的句子结束状态


        self.U_feature_offset = [-2, -1, 0, 1, 2] # 模板中 向前和向后的偏移量
                                               # 若设置 self.U_feature_offset = [0] 则 在此实现中效果与HMM类似

        self.index_feature = {} # eg. {0 : ('U', 1, word, tag)}
        self.feature_index = {} #  eg. { ('U', 1, word, tag) : 0 }

        self.feature_num = 0 # 特征的个数

        self.weights = np.zeros(self.feature_num) # 所有特征的权重

        self.theta = 1e-4 # theta should in the range of (1e-6 ~ 1e-3)



    def log_M_at(self,weights, sentence, pre_tag, tag, t):
        """
        计算  log M(yi_1, yi|x) = sum{k}: W_k*F_k( y_i-1, y_i|x)

        依据公式  (11.22) (11.23)

        :param sentence: 一个句子词的词的列表
        :param pre_tag:
        :param tag:
        :param t:
        :return:
        """
        # (type , offset , word=x[t+offset] , tag=y[t])
        # eg. ('U', 1, '向', 2)
        # 'U': 特征类型为 一元(Unigram);
        #  1: 偏移量, 当前时刻t 向后偏移1个时刻 ;
        #  '向': 向后偏移1个时刻后的 x='向' ;
        #  2: y[t]=2, 当前时刻的标签, 2代表标签 'E'

        # (type,pre_tag,tag)
        # eg. ('B',1,2)
        # 'B': 特征类型为 二元(Bigram);
        # 1: y[t-1]=1 前一个时刻为状态1
        # 2: y[t]=2 当前时刻为状态2

        n=len(sentence)-1

        selectd_feature=[] # 被选中的特征列表

        if pre_tag == self.start_tag: # 句子的开始

            # 一元特征
            for offset in self.U_feature_offset:

                pos = t + offset
                if pos >= 1 and pos < n+1:  # pos 位置必须合法

                    feature = ('U', offset, sentence[pos], tag)

                    if feature in self.feature_index:
                        selectd_feature.append( self.feature_index[feature] )


        elif tag == self.end_tag: # 句子的结束

            # 结果要log化,  log(1)=0
            res=0

            return res

        else:

            # 一元特征
            for offset in self.U_feature_offset:

                pos = t + offset
                if pos >= 1 and pos < n+1:  # pos 位置必须合法

                    feature = ('U', offset, sentence[pos], tag)


                    if feature in self.feature_index:
                        selectd_feature.append( self.feature_index[feature] )

            # 二元特征
            feature = ('B', pre_tag, tag)
            if feature in self.feature_index:
                selectd_feature.append(self.feature_index[feature])

        # 结果要log化, log(exp()) 相当于啥也做, 因此直接求和
        res=np.sum(weights[selectd_feature])

        return res


    def calc_log_M(self, weights,sentence):
        """
        计算状态转移矩阵 M
        返回 log() 化之后的概率

        依据公式 (11.21)  (11.22) (11.23)

        :param sentence:
        :return:
        """
        n = len(sentence)-1

        log_M_list = np.ones((n+2,self.ntags,self.ntags))*infinite
        # 所有时刻的转移矩阵集合; shape: (L_sentence+2, ntags, ntags) 补充上开始时刻和结束时刻, 因此+2
        # 因为取log, M中为0 的元素变为 -inf

        for t in range(1,n+2): # t=1,2,.., n+1

            if t==1: # 开始
                for tag in self.tag_sets: # [0,1,2,3]
                    log_M_list[t][0][tag] = self.log_M_at(weights, sentence, self.start_tag,tag, t)

            elif  t==n+1: # 结束
                for pre_tag in self.tag_sets: # [0,1,2,3]
                    log_M_list[t][pre_tag][0] = self.log_M_at( weights,sentence, pre_tag,self.end_tag, t)

            else:
                for pre_tag in self.tag_sets:
                    for tag in self.tag_sets:
                        log_M_list[t][pre_tag][tag] = self.log_M_at(weights,sentence, pre_tag, tag, t)

        return  log_M_list

    def log_sum_exp(self, a, b):
        """

        a = [a1, a2, a3]
        b = [b1, b2, b3]

        计算 log(e^a1*e^b1+e^a2*e^b2+e^a3*e^b3)

        :param a:
        :param b:
        :return:
        """

        try:
            c = a + b
            max_value = np.max(c)

            if max_value == float('-inf'):
                res=0
            else:
                res = max_value + np.log(np.sum(np.exp(c - max_value)))

        except Warning as e: # 捕捉到 Warning
            print(e)  # debug 时 , 在此处打断点

        return res


    def calc_log_alpha(self, sentence, log_M=None):
        """
        依据公式 (11.26) (11.27) 前向算法计算 每一个时刻 t 的 alpha ;
        返回 log() 化之后的概率

        :param sentence:
        :param M:
        :return:
        """
        n = len(sentence)-1

        log_alpha_list = np.ones((n + 2, self.ntags))* float('-inf')  # 所有时刻 alpha的集合; shape: (L_sentence+2, ntags)

        # 句子的开始
        t=0
        log_alpha_list[t] = np.zeros( self.ntags ) # t=0 alpha 初始化为1, 因为取log 变为0

        # 句子中间 和 结束
        for t in range(1, n + 2):  # t=1,2,.., n+1

            for tag in self.tag_sets:  # [0,1,2,3]

                log_alpha_list[t][tag] = self.log_sum_exp(log_alpha_list[t-1], log_M[t, : ,tag])

        #  句子结束
        # t = L_sentence + 1
        # alpha_list[t][0] = self.log_sum_exp(alpha_list[t-1], M[: ,0])

        return log_alpha_list

    def calc_log_beta(self, sentence, log_M=None):
        """
        依据公式 (11.29) (11.30) 后向算法计算 每一个时刻 t 的 beta ;
        返回 log() 化之后的概率

        :param sentence:
        :param log_M:
        :return:
        """
        n = len(sentence)-1

        log_beta_list = np.ones( (n + 2, self.ntags) ) * float('-inf')  # 所有时刻的 beta集合; shape: (n+2, ntags)

        #  句子结束
        t = n + 1
        log_beta_list[t] = np.zeros(self.ntags) # t=0 beta 初始化为1, 因为取log 变为0

        # 句子中间 和 开始
        for t in range(n,-1,-1 ):  # t= L_sentence,..,1,0

            for tag in self.tag_sets:  # [0,1,2,3]

                log_beta_list[t][tag] = self.log_sum_exp(log_M[t+1,tag ,:],log_beta_list[t+1])

        return log_beta_list



    def calc_log_z(self, alpha):
        """
        依据 公式 (11.31) 下方公式 计算归一化因子 z
        返回 log() 化之后的概率

        :param alpha:
        :return:
        """
        return alpha[-1][0]


    def log_conditional(self, tag_list, log_M,log_z):
        """
        依据公式(11.24) 计算条件概率 p(y|x)

        log p(y|x) = log exp(sum(M)) - log Z(x)

        返回 log() 化之后的概率

        :param tag_list:
        :param M:
        :return:
        """
        n = len(tag_list)-1

        log_p=0

        for t in range(1, n + 2):  # t=1,2,.., n+1

            # 开头
            if t==1:

                try:
                    log_p += log_M[t][0][tag_list[t]] # 取log后 连续相乘变为连续相加
                except Exception as err:
                    print(err)  # debug 时 , 在此处打断点

            # 结尾
            elif t== n+1:
                try:
                    log_p += log_M[t][tag_list[t-1]][0]
                except Exception as err:
                    print(err)  # debug 时 , 在此处打断点

            else:
                log_p += log_M[t][tag_list[t-1]][tag_list[t]]

        return log_p-log_z # 取log后 除法变减法

    def log_marginal(self, tag_list, log_M, log_alpha, log_beta, log_z):
        """
        依据 公式 (11.33) 计算 边缘概率
        log P(yi_1, yi|x) = log alpha(i-1, yi_1) + log M(i, yi_1, yi, x) + log beta(i, yi) - log z(x)

        返回 log() 化之后的概率

        :param log_M:
        :param log_alpha:
        :param log_beta:
        :param log_z:
        :return:
        """
        T = len(tag_list)-1

        log_p = np.zeros((T+2, self.ntags, self.ntags))

        for t in range(1, T + 1):  # t=1,2,.., L_tag_list

            for pre_tag in self.tag_sets:  # [0,1,2,3]
                for tag in self.tag_sets:  # [0,1,2,3]

                    log_p[t][pre_tag][tag] = log_alpha[t-1][pre_tag] + log_M[t][pre_tag][tag] + log_beta[t][tag] - log_z

        return log_p

    def calc_gradient_part2( self,sentence,tag_list, log_M, log_alpha, log_beta, log_z):
        """
        梯度计算 公式中的第二项

        sum{i,k}: P(y_i-1, yi|x)*F_k( y_i-1, y_i|x)

        :param tag_list:
        :param log_M:
        :param log_alpha:
        :param log_beta:
        :param log_z:
        :return:
        """
        T = len(tag_list) - 1

        log_p_marginal=self.log_marginal(tag_list, log_M, log_alpha, log_beta, log_z)

        gradient_part2 = np.zeros(self.feature_num)

        p_marginal= np.exp(log_p_marginal)

        for t in range(1, T + 1):  # t=1,2,.., T

            for pre_tag in self.tag_sets:  # [0,1,2,3]
                for tag in self.tag_sets:  # [0,1,2,3]

                    selectd_feature = []

                    # 一元特征
                    for offset in self.U_feature_offset:

                        pos = t + offset
                        if pos >= 1 and pos < T+1:  # pos 位置必须合法

                            feature = ('U', offset, sentence[pos], tag)

                            if feature in self.feature_index:
                                selectd_feature.append(self.feature_index[feature])

                    if t>1: # t>1 才会考虑 二元特征

                        # 二元特征
                        feature = ('B', pre_tag, tag)
                        if feature in self.feature_index:
                            selectd_feature.append(self.feature_index[feature])

                    gradient_part2[selectd_feature] += p_marginal[t, pre_tag, tag]

        return gradient_part2

    def neg_likelihood_and_gradient(self, weights, feature_count, sentence_list):
        """
        计算 损失函数(极大化 对数似然) 和 损失函数的梯度

        返回 (-损失函数) , (-损失函数的梯度)

        :param weights:
        :param feature_count:
        :param sentence_list:
        :return:
        """
        self.weights = weights # TODO: 使用 optimize.fmin_l_bfgs_b() 需要加上此行, self.weights 才会更新

        likelihood = 0
        gradient_part2 = np.zeros(self.feature_num)

        for sentence,tag_list in sentence_list: # 遍历语料库的所有句子

            sentence = [''] + sentence # sentence 往后偏移一位
            tag_list = [-1] + tag_list # tag_list 往后偏移一位

            # 对于每一个句子, 分别计算 M, alpha, beta, z
            log_M = self.calc_log_M(weights,sentence)

            log_alpha = self.calc_log_alpha(sentence,log_M)
            log_beta = self.calc_log_beta(sentence,log_M)
            log_z = self.calc_log_z(log_alpha)

            likelihood += self.log_conditional(tag_list,log_M,log_z) # 条件概率 即为损失函数的 经验损失部分

            gradient_part2 +=  self.calc_gradient_part2( sentence,tag_list, log_M, log_alpha, log_beta, log_z) # 梯度计算 公式中的第二项

        # 加入正则化项
        likelihood = likelihood - np.dot(weights, weights) * self.theta / 2

        gradient_part1 = feature_count # 梯度计算公式中 第一项为: 对于该语料，每一特征出现的次数。

        gradient = gradient_part1 - gradient_part2  - (weights * self.theta)

        return -likelihood, -gradient


    def decode(self, sentence):
        """
        给定观测序列 X, 计算得到 P(Y|X) 取得最大值的 隐状态序列 Y

        利用维特比算法, 根据 公式 (11.54) - (11.59)

        :param sentence:
        :return:
        """
        # sentence = [''] + list(sentence)  # sentence 往后偏移一位

        sentence = list(sentence)

        tag_list =[]

        n = len(sentence)

        # 记录前一个时刻的 delta
        pre_dp = [0]*self.ntags

        #  当前时刻的状态j 是从上一时刻的状态i 转移而来的, 通过pre_state可以还原出最佳路径
        # pre_state = np.zeros((n+2,self.ntags))

        pre_state = [[0 for j in range(self.ntags)] for i in range(n)]

        # 初始时刻 t=0
        t=0
        for tag in self.tag_sets:

            selectd_feature = []  # 被选中的特征列表

            # 一元特征
            for offset in self.U_feature_offset:

                pos = t + offset
                if pos >= 1 and pos < n + 1:  # pos 位置必须合法

                    feature = ('U', offset, sentence[pos], tag)

                    if feature in self.feature_index:
                        selectd_feature.append(self.feature_index[feature])

            pre_dp[tag] = np.sum( self.weights[selectd_feature] )

        for t in range(1,n): # t=1,...,n-1

            dp = [0]*self.ntags

            for tag in self.tag_sets:

                selectd_feature = []  # 被选中的特征列表

                # 一元特征 只和当前tag 有关
                for offset in self.U_feature_offset:

                    pos = t + offset
                    if pos >= 0 and pos < n :  # pos 位置必须合法

                        feature = ('U', offset, sentence[pos], tag)

                        if feature in self.feature_index:
                            selectd_feature.append(self.feature_index[feature])

                delta_U = np.sum( self.weights[selectd_feature] )

                max_B_delta= float('-inf')
                max_B_delta_index = 0

                for pre_tag in self.tag_sets:

                    selectd_feature = []  # 被选中的特征列表

                    # 二元特征
                    feature = ('B', pre_tag, tag)
                    if feature in self.feature_index:
                        selectd_feature.append(self.feature_index[feature])

                    B_delta = np.sum( self.weights[selectd_feature] ) + pre_dp[pre_tag]

                    if B_delta > max_B_delta:
                        max_B_delta = B_delta
                        max_B_delta_index = pre_tag

                dp[tag] =  delta_U + max_B_delta
                pre_state[t][tag] = max_B_delta_index

            pre_dp = dp

        # 最后时刻 T
        state = np.argmax(pre_dp)
        tag_list.append(self.index_tag[state])

        #  时刻  T-1,...,1
        for t in range(n-1, 0, -1):
            state = pre_state[t][state]
            tag_list.append(self.index_tag[state])

        tag_list.reverse()

        return tag_list



    def get_train_data(self,train_data_dir):
        """
        准备训练数据

        分词数据集 pku_training.2col 的内容为:

        迈	B
        向	E
        充	B
        满	E

        我们将其转换为
        sentence = ['迈','向','充', '满 ']
        tag = [B,E,B,E]

        :param train_data_dir:
        :return:
        """
        dataset = []
        sentence = []  # 句子
        label = []  # 句子中每一个字的标注

        word_dict = set()  # 记录语料库中的字的种类
        tag_dict = set()  # 记录语料库中的标签的种类

        print('loading dateset ,dir: {}'.format(train_data_dir))

        with open(train_data_dir, encoding='utf-8') as f:

            for line in f.readlines():
                # 读到的每行最后都有一个\n，使用strip将最后的回车符去掉
                line = line.strip()

                line_arr = line.split('\t')  # 分隔符为 '\t'

                if len(line_arr) > 1:  # 说明还在一个句子中
                    sentence.append(line_arr[0])
                    label.append(self.tag_index[line_arr[1]])

                    word_dict.add(line_arr[0])
                    tag_dict.add(line_arr[1])

                else:  # 说明此句子已经结束
                    dataset.append((sentence, label))
                    sentence = []
                    label = []

        print('sentence Num: {}'.format(len(dataset)))  # pku 语料库中的句子个数 19056

        return dataset,word_dict,tag_dict

    def build_feature_set(self,dataset, word_dict):
        """
        建立条件随机场的分词特征

        1.状态特征, 简化为与时间 t无关:
        (type , offset , word=x[t+offset] , tag=y[t])

         x  y
        迈	B
        向	E  <-t
        充	B
        满	E

        index_tag: {0:'B', 1:'I', 2:'E', 3:'S'}

        eg.

        ('U', 0, '向', 2)
        'U': 特征类型为 一元 (Unigram);
         0 :  当前时刻 t  ;
       '向':  当前时刻 x[t]='向' ;
         2: y[t]=2, 当前时刻的标签, 2代表标签 'E'

        ('U', -1, '迈', 2)
        'U': 特征类型为 一元 (Unigram);
         -1 :  偏移量, 当前时刻 t 向前偏移1个时刻  ;
       '迈':  向前偏移1个时刻后的 x[t-1]='迈' ;
         2: y[t]=2, 当前时刻的标签, 2代表标签 'E'

        ('U', 1, '充', 2)
        'U': 特征类型为 一元 (Unigram);
         1 :  偏移量, 当前时刻 t 向后偏移1个时刻 ;
       '充': 向后偏移1个时刻后的 x[t+1]='充' ;
         2: y[t]=2, 当前时刻的标签, 2代表标签 'E'

         上述例子表明: t 时刻的隐藏状态 y[t] 不仅仅和 t时刻的观测 x[t]有关,
         还和 t-1 时刻的观测 x[t-1] 与 t+1 时刻的观测 x[t+1] 有关

        2.转移特征, 简化为与 时间 t 和 观测 x 无关

        (type,pre_tag=y[t-1],tag=y[t])

        eg. ('B',0,2)
       'B': 特征类型为 二元 (Bigram);
         1: y[t-1]=0 前一个时刻为 状态0 ,0代表标签 'B'
         2: y[t]=2 当前时刻为 状态2 ,2代表标签 'E'

        3.特征的总数量为 Ucount×Vword×m + m×m

        (1) 状态特征个数为  Ucount×Vword×m
        其中 Ucount表示 U特征提取的位置数目，比如只提取 当前文字前1个，当前文字，当前文字后一个，则此时 Ucount=3，
        Vword为语料库中文字数量，m表示 标签数。

        (2) 转移特征个数为  m×m
         二元特征表达了 前一个标签到当前标签的转移概率, 标签个数为m , 一共有 mxm 种转移


        :param dataset:
        :param word_dict:
        :return:
        """
        print('building the feature set...')

        feature_idx = 0
        index_feature = {} # eg. {0 : ('U', 1, word, tag)}
        feature_index = {} #  eg. { ('U', 1, word, tag) : 0 }

        # 1. 构建状态特征

        # for word in word_dict:
        #     for offset in self.U_feature_offset: #  [-2, -1, 0, 1, 2]
        #         for tag in self.tag_sets: # [0,1,2,3] -> {'B':0, 'I':1, 'E':2, 'S':3}
        #
        #             feature=('U', offset, word, tag)
        #             index_feature[feature_idx]=feature
        #             feature_index[feature]=feature_idx
        #             feature_idx+=1

        for sentence, tag_list in dataset: # 遍历所有的句子

            n = len(sentence)

            for t in range(n): # 遍历整个句子
                for offset in self.U_feature_offset: #  [-2, -1, 0, 1, 2]
                    for tag in self.tag_sets: #  [0, 1, 2, 3]

                        pos = t + offset
                        if pos >= 0 and pos < n:  # pos 位置必须合法

                            feature = ('U', offset, sentence[pos], tag)

                            if feature not in feature_index:
                                feature_index[feature] = feature_idx
                                index_feature[feature_idx] = feature
                                feature_idx += 1

        # 2. 构建转移特征
        for pre_tag in self.tag_sets: #  [0, 1, 2, 3]
            for tag in self.tag_sets:
                feature = ('B', pre_tag, tag)
                index_feature[feature_idx] = feature
                feature_index[feature] = feature_idx
                feature_idx += 1

        self.feature_num = len(feature_index)  # 特征的个数 93976
        print('feature_num :{}'.format(self.feature_num))

        return  feature_index,index_feature

    def statistic_feature_in_dataset(self, dataset):
        """
        统计所有特征在语料库中的出现次数

        :param dataset:
        :return:
        """

        feature_count = np.zeros(self.feature_num)  # 特征的出现次数

        for sentence, tag_list in dataset:  # 遍历语料库的所有句子

            for t in range(len(sentence)):

                # 状态特征
                # (type , offset , x[t+offset] , y[t])
                tag = tag_list[t]

                for offset in self.U_feature_offset:  # [-2, -1, 0, 1, 2]

                    pos = t + offset
                    if pos >= 0 and pos < len(sentence):  # pos 位置必须合法

                        feature = ('U', offset, sentence[pos], tag)

                        # try:
                        feature_count[self.feature_index[feature]] += 1
                        # except Exception as err:
                        #     print(err)  # debug 时 , 在此处打断点

                # 转移特征
                # (type,pre_tag,tag)
                if t - 1 >= 0:
                    pre_tag = tag_list[t - 1]
                    feature = ('B', pre_tag, tag)
                    feature_count[self.feature_index[feature]] += 1

        return feature_count

    def batch_gradient_ascent(self,feature_weights,feature_count, dataset,max_iter,learning_rate=0.01):
        """
        批量梯度上升(带正则项) 优化算法

        ref:
        https://victorjiangxin.github.io/Chinese-Word-Segmentation/

        :param feature_weights:
        :param feature_count:
        :param dataset:
        :param max_iter:
        :param learning_rate:
        :return:
        """
        for epcho in range(max_iter):

            start_time = time.time()

            likelihood, gradient = self.neg_likelihood_and_gradient(feature_weights, feature_count, dataset)

            feature_weights -= learning_rate * gradient  # 更新权重, gradient 为负, 这里负负为正

            print('epcho:{} loss: {}'.format(epcho, likelihood / 1000))

            print("Training one epcho time:{}s".format(time.time() - start_time))


    def  L_BFGS(self,feature_weights,feature_count, dataset,max_iter):
        """
        拟牛顿法 L-BFGS

        ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
        https://github.com/VictorJiangXin/Linear-CRF/blob/master/src/crf.py


        :param feature_weights:
        :param feature_count:
        :param dataset:
        :param max_iter:
        :param learning_rate:
        :return:
        """
        start_time = time.time()

        func = lambda weights: self.neg_likelihood_and_gradient(weights, feature_count, dataset)

        res = optimize.fmin_l_bfgs_b(func, x0=feature_weights, iprint=0, disp=1, maxiter=max_iter, maxls=100)
        #TODO:  def fmin_l_bfgs_b -> def _minimize_lbfgsb 中
        # task_str = task.tostring() 会导致报错, 按照提示将其改为
        # task_str = task.tobytes() 原因未知

        print("Training time:{}s".format(time.time() - start_time))

        return

    def fit(self, train_data_dir, max_iter,learning_rate=0.01, pre_train_model=None):
        """
        模型训练 主流程

        :param train_data_dir:
        :param max_iter: 迭代次数
        :param learning_rate: 梯度上升的学习率
        :return:
        """

        if pre_train_model == None:

            dataset, word_dict, tag_dict = self.get_train_data(train_data_dir)

            assert tag_dict == self.tag_index.keys() # 语料库中的标签的种类 需要与预设的一致

            self.feature_index,self.index_feature=self.build_feature_set(dataset, word_dict)

            feature_count=self.statistic_feature_in_dataset(dataset)

            feature_weights = np.random.randn(self.feature_num)  # 所有特征的权重初始化

        else: # 在预训练模型的基础上 进行训练

            self.load(pre_train_model)

            feature_weights = self.weights

            dataset, word_dict, tag_dict = self.get_train_data(train_data_dir)

            feature_count = self.statistic_feature_in_dataset(dataset)


        print("Start training...")


        # 可以看出 普通的梯度上升算法收敛地很慢, 迭代次数 max_iter=100 才会收敛
        # self.weights = self.batch_gradient_ascent(feature_weights,feature_count, dataset,max_iter,learning_rate)

        self.L_BFGS(feature_weights,feature_count, dataset,max_iter=max_iter)

        self.save(train_data_dir) # 保存训练完的模型


    def save(self, train_data_dir):
        """
        保存训练好的 CRF 模型

        :param train_data_dir:
        :return:
        """
        model_dir = 'model/'
        model_file_name = train_data_dir.split('/')[-1] + '.model' # 取数据集的名字作为模型的名字

        save_dict = {}
        save_dict['ntags'] = self.ntags
        save_dict['index_tag'] = self.index_tag
        save_dict['tag_index'] = self.tag_index
        save_dict['feature_index'] = self.feature_index
        save_dict['index_feature'] = self.index_feature
        save_dict['feature_num'] = self.feature_num
        save_dict['weights'] = self.weights
        with open(model_dir+model_file_name, 'wb') as f:
            pickle.dump(save_dict, f)

        print("Save model successful!")


    def load(self, file_path):
        """
        读取预训练的 CRF 模型

        :param file_path:
        :return:
        """

        with open(file_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.ntags = save_dict['ntags']
        self.index_tag = save_dict['index_tag']
        self.tag_index = save_dict['tag_index']
        self.feature_index = save_dict['feature_index']
        self.index_feature = save_dict['index_feature']
        self.feature_num = save_dict['feature_num']
        self.weights = save_dict['weights']

        print("Load model successful!")


    def load_crfpp_model(self, model_path):
        """
        导入通过 crf++(crfcpp) 包训练的模型

        :param model_path:
        :return:
        """

        with open(model_path, 'r',encoding='utf-8') as f:
            lines = f.readlines()

        tags_id = 0

        i = 0
        # print plus information
        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip()
            print(line)
            i += 1

        i += 1
        # get tags
        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip()
            self.tag_index[line] = tags_id
            self.index_tag[tags_id] = line
            tags_id += 1
            i += 1

        self.ntags = len(self.tag_index)
        print(self.tag_index)

        i += 1
        # map
        feature_map = {} # {'U00', -2}
        self.U_feature_offset = []
        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip()
            if line != 'B':
                feature_template = line.split(':')[0]
                pos = line.split('[')[1].split(',')[0]
                feature_map[feature_template] = int(pos)
                self.U_feature_offset.append(int(pos))
            i += 1
        print('self.U_feature_offset', self.U_feature_offset)
        print('feature_map:', feature_map)

        i += 1

        # construct feature
        feature_id = 0
        feature_id_weight_index = {}    # in model.txt weight are not in
        while i < len(lines) and lines[i] != '\n':
            weight_index = int(lines[i].strip().split()[0])
            line = lines[i].strip().split()[1]
            if line == 'B':
                for tag_pre in range(self.ntags):
                    for tag_now in range(self.ntags):
                        feature = ('B', tag_pre, tag_now)
                        self.feature_index[feature] = feature_id
                        self.index_feature[feature_id] = feature
                        feature_id_weight_index[feature_id] = weight_index
                        weight_index += 1
                        feature_id += 1
            else:
                feature_template = line.split(':')[0]
                word = line.split(':')[1]
                pos = feature_map[feature_template]
                for tag in range(self.ntags):
                    feature = ('U', pos, word, tag)
                    self.feature_index[feature] = feature_id
                    self.index_feature[feature_id] = feature
                    feature_id_weight_index[feature_id] = weight_index
                    weight_index += 1
                    feature_id += 1
            i += 1

        print('Total features:', len(self.feature_index))
        i += 1
        # read weights
        self.feature_num = len(self.feature_index)
        self.weights = np.zeros(self.feature_num)

        weights_in_file = []

        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip()
            weights_in_file.append(float(line))
            i += 1

        for feature_id in feature_id_weight_index:

            # try:
            self.weights[feature_id] = weights_in_file[feature_id_weight_index[feature_id]]

            # except Exception as err:
            #     print(err)  # debug 时 , 在此处打断点

        print('Record weights = ', feature_id)

        # print("The last feature is {}, it's weight is {}".format(
        #             self.index_feature[feature_id-1], self.weights[feature_id-1]))

        print("Load crfcpp model successful!")

class Report:

    def compare_line(self, reference, candidate):
        """
        输出标准标注结果 与 模型预测结果的重合部分的长度, 依据此可以计算出 P,R F1 值

        eg.

        标准分词 A：['结婚',' 的',' 和',' 尚未',' 结婚 ','的']

        标准区间 A：[1,2],[3,3],[4,4],[5,6],[7,8],[9,9]  一共 6个区间

        分词结果 B：['结婚',' 的','和尚','未结婚 ','的 ']

        分词区间 B：[1,2],[3,3],[4,5],[6,7,8],[9,9] 一共 5个区间

        A 和 B 的相同区间为 [1,2],[3,3],[9,9]  一共 3个区间

        ref_words_len = 6
        can_words_len = 5
        acc_word_len = 3

        Precision = 3/5
        Recall = 3/6

        ref:
        https://zhuanlan.zhihu.com/p/100552669

        :param reference: 标注的分词结果
        :param candidate: 模型预测的分词结果
        :return:
        """


        ref_words = reference.split()
        can_words = candidate.split()

        ref_words_len = len(ref_words)
        can_words_len = len(can_words)

        ref_index = []
        index = 0
        for word in ref_words:
            word_index = [index]
            index += len(word)
            word_index.append(index)
            ref_index.append(word_index)

        can_index = []
        index = 0
        for word in can_words:
            word_index = [index]
            index += len(word)
            word_index.append(index)
            can_index.append(word_index)

        tmp = [val for val in ref_index if val in can_index]
        acc_word_len = len(tmp)

        return ref_words_len, can_words_len, acc_word_len

class CRFSegmentation(object):
    """
    基于 CRF 的分词器, 功能包括:

    1.可以选择导入 LinearCRF 模型训练得到的参数, 还是 通过 CRF++ 库训练得到的参数
    2.对单个句子分词
    3.对整个文档分词并输出
    4.将分词后的结果与标注文档比较, 输出文档分词的分数

    Author: xrh
    Date: 2021-06-30

    """

    def __init__(self, model_path='lib/model/pku_model.txt',use_crfcpp=True):
        """

        :param model_path:
        :param use_crfcpp: 是否使用 CRF++ 训练的模型
        """

        self.crf = LinearCRF()

        if use_crfcpp:

            print('loading the crfcpp model, dir: {}'.format(model_path))

            self.crf.load_crfpp_model(model_path)

        else:

            print('loading model, dir: {}'.format(model_path))

            self.crf.load(model_path)

    def cut(self, sentence):
        """
        对句子进行分词, 词与词之间使用 空格间隔

        :param sentence:
        :return:
        """

        sentence=sentence.strip()

        tag_list = self.crf.decode(sentence)

        out_str = ''

        for i in range(len(tag_list)):

            try:
                out_str = out_str + sentence[i]

            except Exception as err:
                print(err)  # debug 时 , 在此处打断点

            if tag_list[i] in {'E', 'S'}:  # 词语的结束标志
                out_str = out_str + ' '

        return out_str


    def cut_doc(self, in_dir, out_dir):
        """
        对整个文档进行分词并输出

        :param in_dir:
        :param out_dir:
        :return:
        """
        # 初始化 输出
        output_lines = []

        with open(in_dir, encoding='utf-8') as f:
            # 按行读取文件
            for line in f.readlines():
                # 读到的每行最后都有一个\n，使用strip将最后的回车符去掉
                line = line.strip()
                # 将该行放入文章列表中
                output_lines.append(self.cut(line)+'\n')


        # 设置定长缓冲区
        with open(out_dir, 'w+', encoding='utf-8', buffering=200) as f:
            f.writelines(output_lines)

    def score_cut_doc(self, in_file,ref_file, out_file):
        """
        对原始文档进行分词, 并对分词结果进行评价
        评价指标包括 precison, recall, f1

        ref:
        https://zhuanlan.zhihu.com/p/100552669

        :param in_file:原始文档(未分词)
        :param ref_file: 标注文档
        :param out_file: 分词后的结果文档
        :return:
        """

        self.cut_doc(in_file,out_file)

        report = Report()

        fref = open(ref_file, 'r', encoding='utf8')
        fcan = open(out_file, 'r', encoding='utf8')

        reference_all = fref.readlines()
        candidate_all = fcan.readlines()
        fref.close()
        fcan.close()

        ref_count = 0
        can_count = 0
        acc_count = 0
        for reference, candidate in zip(reference_all, candidate_all):
            reference = reference.strip()
            candidate = candidate.strip()

            ref_words_len, can_words_len, acc_word_len = report.compare_line(reference, candidate)
            ref_count += ref_words_len
            can_count += can_words_len
            acc_count += acc_word_len

        P = acc_count / can_count
        R = acc_count / ref_count
        F1 = (2 * P * R) / (P + R)

        print('Precision:', P)
        print('Recall:', R)
        print('F1:', F1)



class Test:

    def train_model(self):

        np.random.seed(0)  # 设置随机数 种子

        crf = LinearCRF()
        # crf.fit('../dataset/ChineseCutWord/pku_training.2col.tiny',max_iter=10,learning_rate=0.01)

        # crf.fit('../dataset/ChineseCutWord/pku_training.2col.small', max_iter=100, learning_rate=0.01) # Training time:40111s

        crf.fit('../dataset/ChineseCutWord/pku_training.2col', max_iter=20, learning_rate=0.01,pre_train_model='model/pku_training.2col.model') # 原始数据集跑的太慢了


    def test_cut_sentence(self):

        seg = CRFSegmentation('model/crf_pku_small.model',use_crfcpp=False)

        # seg = CRFSegmentation()

        sen1 = '今晚的月色真美呀！'
        sen2 = '生命在于奋斗！'
        sen3 = '小哥哥，别复习了，来玩吧！'

        print('测试句:', sen1)
        print('分词后:', seg.cut(sen1))

        print('测试句:', sen2)
        print('分词后:', seg.cut(sen2))

        print('测试句:', sen3)
        print('分词后:', seg.cut(sen3))

    def test_cut_doc_and_eval(self):

        # seg = CRFSegmentation()

        seg = CRFSegmentation('model/pku_training.2col.model',use_crfcpp=False) # F1: 0.46

        seg.score_cut_doc(in_file='test/data/test_weibo.txt',
                          ref_file='test/data/weibo.txt',
                          out_file='test/result/weibo_crfcpp.txt')



if __name__ == '__main__':

    test = Test()

    test.train_model()

    # test.test_cut_sentence()

    # test.test_cut_doc_and_eval()





