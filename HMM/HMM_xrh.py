#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import time
import pickle
import random

# 定义负无穷大
infinite = -(2 ** 31)


def log_normalize(arr):
    """
    1.将数组中的元素值转换为 概率

    arr[i] = arr[i]/sum

    2.概率值可能过小, 将其转换为 对数形式

    arr[i] = log(arr[i] )

    :param arr:
    :return:
    """

    log_s = np.log(np.sum(arr))

    for i in range(len(arr)):
        if arr[i] == 0:
            arr[i] = infinite
        else:
            arr[i] = np.log(arr[i]) - log_s  # log(arr[i]/s)= log(arr[i]) - log(s)


def log_sum(arr_logprob):
    """
    类似于 scipy 中的 logsumexp

    :param arr_logprob:  对数概率 数组
    :return:
    """

    if len(arr_logprob) == 0:  # arr 为空
        return infinite

    m = np.max(arr_logprob)  # 记录 arr中的最大值

    # 将 对数化的概率 转换回概率, 然后对概率求和
    s = np.sum(np.exp(arr_logprob - m))  # np.exp(arr-m): 防止 np.exp() 出现上溢出

    return m + np.log(s)  # exp(arr-m) = exp(arr)/exp(m) ;  log(1/(exp(m)))= -m


class HMMSegment:
    '''

    利用 HMM 进行中文分词, 功能包括:

    1.有监督学习, 通过标注的 训练样本 得到 模型的参数( 转移矩阵, 发射矩阵 )
    2. 无监督学习, 利用 EM 算法 迭代得到 模型的参数
    3. 基于 维特比算法的解码器, 对中文句子进行分词

    ref:
    《统计学习方法 第二版》李航

    Author: xrh
    Date: 2021-06-13

    '''

    def __init__(self, train_data_dir=None, pre_train_model_dir='HMMSegment.bin', use_pre_train=False, mode='supervised'):

        # 1. 定义状态变量 i 的空间 Q
        # 每个字只有4种状态:
        # B：词语的开头
        # M：一个词语的中间词
        # E：一个词语的结果
        # S：非词语，单个词
        self.statuDict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        self.rev_statuDict = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}

        if not use_pre_train:

            self.mode = mode

            if self.mode == 'supervised':

                # 2. 定义状态变量 i 的初始概率

                # - 访问局部变量速度要比成员变量快很多
                PI = np.zeros(4)

                # 3. 定义状态变量 i 的转移概率矩阵
                A = np.zeros((4, 4))

                # 4. 定义观测变量 o 的发射概率矩阵
                # 因为是中文分词，使用ord(汉字)即可找到其对应编码，汉字个数的上限为65536
                B = np.zeros((4, 65536))

                PI, A, B = self.do_train_supervised(train_data_dir, PI, A, B)

                self.PI = PI
                self.A = A
                self.B = B

            else:  # self.mode == 'unsupervised'

                # 2. 定义状态变量 i 的初始概率分布
                PI = np.array([random.random() for __ in range(4)])  # random.random() 返回随机生成的一个实数，它在[0,1)范围内

                # PI 需满足 sum(PI)=1
                log_normalize(PI)

                # 3. 定义状态变量 i 的转移概率矩阵
                A = np.array([[random.random() for __ in range(4)] for __ in range(4)])

                # 3.1 加入先验知识
                # B：词语的开头
                # M：一个词语的中间词
                # E：一个词语的结果
                # S：非词语，单个词

                A[0][0] = 0  # 不可能发生 state(t-1)=B -> state(t)=B

                # 同理可得:
                A[0][3] = A[1][0] = A[1][3] \
                    = A[2][1] = A[2][2] = A[3][1] = A[3][2] = 0

                # 4. 定义观测变量 o 的发射概率矩阵
                B = np.array([[random.random() for __ in range(65536)] for __ in range(4)])

                for i in range(4):
                    # 对于每一种隐状态i, 由它转移后的状态j的概率求和为1, A 需要满足 按行求和为1
                    log_normalize(A[i])

                    # 对于每一种隐状态i, 由它发射后的观测状态j的概率求和为1,B 需要满足 按行求和为1
                    log_normalize(B[i])

                epoch = 10
                PI, A, B = self.do_train_unsupervised(train_data_dir, PI, A, B, epoch)

                self.PI = PI
                self.A = A
                self.B = B

            print('build model complete')

            # 序列化 训练好的模型参数
            with open(pre_train_model_dir, 'wb')  as f:
                pickle.dump([PI, A, B], f, 0)


        else:  # 使用 预先训练好的模型
            with open(pre_train_model_dir, 'rb')  as f:

                PI, A, B = pickle.load(f)

                self.PI = PI
                self.A = A
                self.B = B

    def do_train_unsupervised(self, train_data_dir, PI, A, B, epoch_num=10):
        """
        利用 EM 算法估计出 HMM 的参数

        :param train_data_dir:
        :param PI:
        :param A:
        :param B:
        :param epoch_num: EM 算法的迭代次数
        :return:
        """
        one_long_line = ''

        with open(train_data_dir, encoding='utf-8') as f:

            for line in f.readlines():
                # 读到的每行最后都有一个\n，使用strip将最后的回车符去掉
                line = line.strip()
                one_long_line = one_long_line + line

        T = len(one_long_line)  # 观测序列的长度
        alpha = np.zeros((4, T))  # 依据公式 (10.14)
        beta = np.zeros((4, T))  # 依据公式 (10.18)
        gamma = np.zeros((4, T))  # 依据公式 (10.23)
        ksi = np.zeros((4, 4, T-1))  # 依据公式 (10.25)

        O = list(one_long_line)  # 观测序列

        # EM 迭代
        for i in range(epoch_num):
            print('epoch :{}'.format(i))
            self.baum_welch(O, PI, A, B, alpha, beta, gamma, ksi)

        return PI, A, B

    def baum_welch(self, O, PI, A, B, alpha, beta, gamma, ksi):
        """
        《统计学习方法 第二版》 算法 10.4(BaumWelch)

        :return:
        """
        self.forward(O, alpha, PI, A, B)
        self.backward(O, beta, A, B)
        self.update_gamma(O,alpha, beta,gamma)
        self.update_ksi( O,alpha, beta,ksi, A, B)

        self.update_A(O,gamma,ksi,A)
        self.update_B(O,gamma,B)

        self.update_PI(gamma,PI)

        # return PI, A, B, alpha, beta, gamma, ksi

    def forward(self, O, alpha, PI, A, B):
        """
        前向算法

        :param O:
        :param alpha:
        :param PI:
        :param A:
        :param B:
        :return:
        """
        T = len(O)

        alpha[:, 0] = PI + B[:, ord(O[0])]  # 公式(10.15) ; alpha shape:(4,T)

        for t in range(1, T):# 1,..,T-1
            for i in range(4):  # t 时刻的状态
                temp = np.zeros(4)
                for j in range(4):  # t-1 时刻的状态

                    temp[j] = alpha[j][t - 1] + A[j][i]

                s = log_sum(temp)
                alpha[i][t] = s+B[i][ord(O[t])] # 公式(10.16)

    def backward(self, O, beta, A, B):
        """
        后向算法

        :param O:
        :param beta:
        :param A:
        :param B:
        :return:
        """
        T = len(O)

        beta[:, T-1] = 1  # 公式(10.19) ;

        for t in range(T-2, -1,-1):# T-2,...,0
            for i in range(4):  # t 时刻的状态
                temp = np.zeros(4)
                for j in range(4):  # t+1 时刻的状态
                    temp[j] = A[i][j] + B[j][ord(O[t+1])] + beta[j][t + 1]

                beta[i][t] = log_sum(temp) # 公式(10.20)

    def update_gamma(self, O, alpha, beta,gamma):
        """
        依据公式 (10.24) 更新 gamma

        :param O:
        :param alpha:
        :param beta:

        :return:
        """
        T = len(O)

        temp=np.zeros((4,T))

        for t in range(T):

            temp[:,t]= alpha[:,t]+beta[:,t]
            s= log_sum(temp[:,t])

            gamma[:,t] = temp[:,t] - s

    def update_ksi(self,O,alpha, beta,ksi, A, B):
        """
        利用公式 (10.26) 更新 ksi

        :param O:
        :param alpha:
        :param beta:
        :param ksi:
        :param A:
        :param B:
        :return:
        """
        T = len(O)

        for t in range(T-2, -1,-1):# T-2,...,0

            temp= np.zeros((4,4))

            for i in range(4):
                for j in range(4):
                    temp[i][j]= alpha[i][t] + A[i][j] + B[j][ord(O[t+1])] + beta[j][t+1] # 公式 (10.26) 分子

                    ksi[i][j][t]= temp[i][j]

            s=log_sum(temp.flatten()) # 公式 (10.26) 分母 ; .flatten() 拍平成 1维向量

            ksi[:,:,t]=ksi[:,:,t]-s # 对数化后除法变减法


    def update_A(self,O,gamma,ksi,A):
        """
        根据公式 (10.39) 更新 状态转移矩阵A
        :param O :
        :param gamma:
        :param ksi:
        :param A:
        :return:
        """
        T=len(O)

        for i in range(4):

            s=log_sum(gamma[i,:T-1]) #  公式 (10.39) 分母

            for j in range(4):
                A[i][j]= log_sum(ksi[i,j,:])-s

    def update_B(self,O, gamma,B):
        """
        根据公式 (10.40) 更新 发射矩阵B

        :param O:
        :param gamma:
        :return:
        """
        T = len(O)

        s1 = np.zeros(T)
        s2 = np.zeros(T)

        for j in range(4):
            for k in range(65536): # TODO: 效率太低, 需要优化
                # if k % 5000 == 0:
                #     print(j,k)
                valid = 0
                for t in range(T):
                    if ord(O[t]) == k:
                        s1[valid] = gamma[j][t]
                        valid += 1
                    s2[t] = gamma[j][t]
                if valid == 0:
                    B[j][k] = -log_sum(s2)  # 平滑
                else:
                    B[j][k] = log_sum(s1[:valid]) - log_sum(s2)


    def update_PI(self,gamma,PI):
        """
        根据公式 (10.41) 更新 初始概率分布 PI

        :param O:
        :param gamma:
        :param PI:
        :return:
        """
        for i in range(4):
            PI[i]=gamma[i,0]


    def __count_one_line(self, line_str, PI, A, B):
        """
        统计 单行训练样本 中 PI, A, B, 的次数

        ---------------------单行训练样本样例--------------------
         line: '深圳  有  个  打工者  阅览室'
         line_label: [BE   S   S   BME    BME]
        ------------------------------------------------------
        可以看到训练样本已经分词完毕，词语之间空格隔开，因此我们在生成统计时主要借助以下思路：
        1.先将句子按照空格隔开，例如例句中5个词语，隔开后变成一个长度为5的列表，每个元素为一个词语
        2.对每个词语长度进行判断：
              如果为1认为该词语是S，即单个字
              如果为2则第一个是B，表开头，第二个为E，表结束
              如果大于2，则第一个为B，最后一个为E，中间全部标为M，表中间词
        3.统计PI：该句第一个字的词性对应的PI中位置加1
                  例如：PI = [0， 0， 0， 0]，当本行第一个字是B，即表示开头时，PI中B对应位置为0，
                    则 PI = [1， 0， 0， 0]，全部统计结束后，按照计数值再除以总数得到概率
          统计A：对状态链中位置t和t-1的状态进行统计，在矩阵中相应位置加1，全部结束后生成概率
          统计B：对于每个字的状态以及字内容，生成状态到字的发射计数，全部结束后生成概率

        :param line_str:
        :return:
        """

        terms_arr = line_str.strip().split()  # 去掉首尾空格后用空格做切分

        if len(terms_arr) == 0:  # 此行数据为空 的异常处理:
            return

        line_label = []

        for i in range(len(terms_arr)):  # 遍历 句子中 所有的词

            term = terms_arr[i]
            term_length = len(term)

            if term_length == 1:
                term_label = ['S']

            elif term_length == 2:
                term_label = ['B', 'E']

            else:
                term_label = ['B'] + ['M'] * (term_length - 2) + ['E']

            for j in range(term_length):
                # term =     '打工者'
                # term_label=[B,M,E]

                if j == 0:  # 如果是开头第一个字，PI中对应位置加1,
                    PI[self.statuDict[term_label[j]]] += 1

                B[self.statuDict[term_label[j]]][ord(term[j])] += 1

            line_label.extend(term_label)

        # 统计 A
        for k in range(1, len(line_label)):
            # line_label: [B,E,S,S,B,M,E,B,M,E]

            A[self.statuDict[line_label[k - 1]]][self.statuDict[line_label[k]]] += 1

    def do_train_supervised(self, train_data_dir, PI, A, B):
        """
        有监督学习, 通过标注的 训练样本 得到 模型的参数( 转移矩阵, 发射矩阵 )

        :param train_data_dir:
        :return:
        """

        with open(train_data_dir, encoding='utf-8') as f:

            # 训练文本中的每一行 为一个训练样本
            for line in f.readlines():
                self.__count_one_line(line, PI, A, B)

        # 上面代码在统计上全部是统计的次数，实际运算需要使用概率，下方代码是将三个参数的次数转换为概率

        # 对PI求和，概率生成中的分母
        sum_PI = np.sum(PI)
        # 遍历PI中每一个元素，元素出现的次数/总次数即为概率
        for i in range(len(PI)):

            # 如果某元素没有出现过，该位置为0，在后续的计算中这是不被允许的
            # 比如说某个汉字在训练集中没有出现过，那在后续不同概率相乘中只要有
            # 一项为0，其他都是0了，此外整条链很长的情况下，太多0-1的概率相乘
            # 不管怎样最后的结果都会很小，很容易下溢出
            # 所以在概率上我们习惯将其转换为log对数形式，这在书上是没有讲的
            # x大的时候，log也大，x小的时候，log也相应小，我们最后比较的是不同
            # 概率的大小，所以使用log没有问题
            # 那么当单向概率为0的时候，log没有定义，因此需要单独判断
            # 如果该项为0，则手动赋予一个极小值

            if PI[i] == 0:
                PI[i] = -3.14e+100
            # 如果不为0，则计算概率，再对概率求log
            else:
                PI[i] = np.log(PI[i] / sum_PI)

        # 《统计学习方法 第二版》李航 公式(10.31)
        for i in range(len(B)):
            sum_B = np.sum(B[i])  # 对每一个 隐状态求和

            for j in range(len(B[0])):
                if B[i][j] == 0:
                    B[i][j] = -3.14e+100
                # 如果不为0，则计算概率，再对概率求log
                else:
                    B[i][j] = np.log(B[i][j] / sum_B)

        # 《统计学习方法 第二版》李航 公式(10.30)
        for i in range(len(A)):
            sum_A = np.sum(A[i])  # 对每一个 隐状态求和
            for j in range(len(A[0])):
                if A[i][j] == 0:
                    A[i][j] = -3.14e+100
                # 如果不为0，则计算概率，再对概率求log
                else:
                    A[i][j] = np.log(A[i][j] / sum_A)

        return PI, A, B

    def decode(self, sentence, PI, A, B):
        """
        基于维特比算法, 通过观测序列 解码出 出现概率最大的状态序列

        时间复杂度:  O(T*V^2)
                    T - 句子的长度
                    V - 隐状态个数

        :param sentence: 待分词的句子
        :return:
        """
        # sentence ='深圳有个打工者阅览室'

        # 当前时刻的状态j 是从上一时刻的状态i 转移而来的, 通过pre_state可以还原出最佳路径
        pre_state = [[0 for j in range(4)] for i in range(len(sentence))]

        label = []

        # 1. 初始化, 根据第1个字sentence[0] , 设置初始状态
        pre_dp = [0] * 4

        for j in range(len(pre_dp)):
            #   由于公式是概率直接相乘，但我们在求得概率时，同时取了log，取完log以后，概率的乘法
            #   也就转换为加法了，同时也简化了运算
            pre_dp[j] = PI[j] + B[j][ord(sentence[0])]

        # label.append(self.rev_statuDict[np.argmax(pre_dp)])

        # 2.递推方程
        for t in range(1, len(sentence)):

            dp = [0] * 4
            for j in range(4):  # t 时刻的状态

                max_psi = float('-inf')
                max_psi_index = 0

                for i in range(4):  # t-1 时刻的状态

                    # 《统计学习方法 第二版》公式 10.46 中的 ψ
                    psi = pre_dp[i] + A[i][j]

                    if psi >= max_psi:
                        max_psi = psi
                        max_psi_index = i

                # 公式 10.45 中的 δ
                max_delta = max_psi + B[j][ord(sentence[t])]
                dp[j] = max_delta

                pre_state[t][j] = max_psi_index

            pre_dp = dp

        # 3.回溯解

        # 最后时刻 T
        state = np.argmax(pre_dp)
        label.append(self.rev_statuDict[state])

        # 时刻  T-1,...,1
        for t in range(len(sentence) - 1, 0, -1):
            state = pre_state[t][state]
            label.append(self.rev_statuDict[state])

        label.reverse()

        return label

    def cut(self, sentence):
        """
        对句子进行分词, 词与词之间使用 空格间隔

        :param sentence:
        :return:
        """
        sentence = sentence.strip()

        label = self.decode(sentence, self.PI, self.A, self.B)

        out_str = ''

        for i in range(len(label)):

            out_str = out_str + sentence[i]

            if label[i] in {'E', 'S'}:  # 词语的结束标志
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
                output_lines.append(self.cut(line))

        # 设置定长缓冲区
        with open(out_dir, 'w+', encoding='utf-8', buffering=200) as f:
            f.writelines(output_lines)

class Test:

    def train_supervised_model(self):

        # 开始时间
        start = time.time()

        train_file_dir = '../dataset/ChineseCutWord/人民日报1998年中文标注语料库.txt'
        model_dir = 'HMMSegment.bin'

        seg = HMMSegment(train_file_dir, model_dir, use_pre_train=False,mode='supervised')

        # 结束时间
        print('build model time span:', time.time() - start)

        print(seg.cut(' 深圳有个打工者阅览室 \n'))

    def test_cut_doc(self):

        # 开始时间
        start = time.time()

        model_dir = 'HMMSegment.bin'
        seg = HMMSegment( model_dir, use_pre_train=True)

        in_dir = '../dataset/ChineseCutWord/testArtical.txt'
        out_dir = 'res_testArtical.txt'
        seg.cut_doc(in_dir, out_dir)

        print('time span:', time.time() - start)

    def train_unsupervised_model(self):

        # 开始时间
        start = time.time()

        train_file_dir = '../dataset/ChineseCutWord/novel.txt'
        model_dir = 'HMMSegment_EM.bin'

        seg = HMMSegment(train_file_dir, model_dir, use_pre_train=False,mode='unsupervised')

        # 结束时间
        print('build model time span:', time.time() - start) # TODO: 跑了 11h (真慢), 分词效果也很差, EM学了个寂寞

        print(seg.cut(' 深圳有个打工者阅览室 \n'))


    def test_cut_doc_unsupervised(self):

        # 开始时间
        start = time.time()

        model_dir = 'HMMSegment_EM.bin'
        seg = HMMSegment( pre_train_model_dir=model_dir, use_pre_train=True)

        in_dir = '../dataset/ChineseCutWord/testArtical.txt'
        out_dir = 'res_testArtical_EM.txt'
        seg.cut_doc(in_dir, out_dir)

        print('time span:', time.time() - start)

if __name__ == '__main__':

    test=Test()

    # test.test_cut_doc()

    # test.train_unsupervised_model()

    test.test_cut_doc_unsupervised()