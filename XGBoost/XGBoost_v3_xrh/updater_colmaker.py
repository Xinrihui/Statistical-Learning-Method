import numpy as np
from multiprocessing import Pool

from tree_model import *
from split_evaluator import *

from param import *


class ThreadEntry:
    """
    每一个线程中 对节点的统计信息

    """

    def __init__(self):

        #梯度 统计信息
        self.stats = GradeStats()

        # 最后扫描到的特征值
        self.last_fvalue=0

        # 最优的分裂方案(分裂特征的标号，分裂的特征值，增益loss变化)
        self.best=SplitEntry()


class NodeEntry:
    """
    节点的统计信息

    """

    def __init__(self):

        # 梯度 统计信息
        self.stats = GradeStats()

        # 节点没有分裂时的增益
        self.root_gain=0.0

        # 当前节点的 weight
        self.weight=0.0

        #最优的分裂方案(分裂特征的标号，分裂的特征值，增益loss变化)
        self.best=SplitEntry()


class Builder:
    """

    建立 CART回归树

    1.采用 非递归的 宽度优先 建树

    Author: xrh
    Date: 2021-05-29

    """

    def __init__(self, root=None,
                       gama=0,
                       reg_alpha=0,
                       reg_lambda=1,
                       max_depth=2,
                       min_sample_split=2 ,
                       max_delta_step=0,
                       min_child_weight=0,
                       tree_method='exact',
                       sketch_eps=0.3,
                       nthread =16,
                       print_log=True):

        """

        :param root: 树的根节点
        :param gama: 在树的叶节点上进行进一步 划分节点 所需的最小 增益(Gain); 若小于此阈值, 不往下分裂, 形成叶子节点 ; 越大gamma，算法将越保守。

        :param reg_alpha： 一个浮点数，是L1 正则化系数。它是xgb 的alpha 参数
        :param reg_lambda： 一个浮点数，是L2 正则化系数。它是xgb 的lambda 参数

        :param max_depth: 树的最大深度

        :param min_sample_split: 划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2

        :param max_delta_step： 每棵树的权重估计时 的最大delta step。取值范围为 [0, ) ，0 表示没有限制，默认值为 0 。

        :param min_child_weight: 搜索最佳切分点时, 若 min_child_weight < Min(HL, HR) 则放弃此切分点

        :param tree_method： 指定了构建树的算法，可以为下列的值：
                            (1)'exact'： 使用 exact greedy 算法分裂节点
                            (2)'approx'： 使用近似算法分裂节点

        :param sketch_eps： 指定了分桶的步长。其取值范围为 (0,1)， 默认值为 0.3 。
                            它仅仅用于 tree_medhod='approx'。

        :param nthread： 训练期间的 线程数

        :param print_log: 是否打印日志
        """

        self.root = root

        self.params={}

        self.params['gama']=gama # 损失的 阈值

        self.params['reg_alpha'] = reg_alpha
        self.params['reg_lambda'] = reg_lambda

        self.params['max_depth'] = max_depth# 树的最大深度

        self.params['min_sample_split'] = min_sample_split

        self.params['max_delta_step'] = max_delta_step

        self.params['min_child_weight'] = min_child_weight

        self.params['tree_method'] = tree_method

        self.params['sketch_eps'] = sketch_eps

        self.params['nthread'] = nthread

        self.params['print_log'] = print_log #是否打印日志



    def select_max_gain_split(self,Blocks):
        """

        Blocks['N']=N # 样本个数
        Blocks['m'] = N  # 特征个数
        Blocks['l'] = block_list # 块集合

        block_list=[block_0, block_1,...]  block_0: 特征0 对应的块

        block_i=[(特征值, 样本标号),... ]

        选择 最优的 切分特征 与 切分点

        :param Blocks:

        :return:
        """
        Ag = None  # 最佳特征
        Ag_split = None  # 最佳特征的切分点 , 左子节点为 <=Ag_split , 右子节点为 >Ag_split
        Ag_split_idx=None # 最佳特征的切分点的块的行标号, 方便之后对块的切分

        max_Gain = float('-inf')

        N=Blocks['N'] # 当前块 拥有的样本个数
        m= Blocks['m'] # 特征个数

        block_0 = Blocks['l'][0]
        index = block_0[:,1] # 当前 节点拥有的样本标号,  所有 block 的样本标号应该是相同的

        index = index.astype(int)

        G=np.sum(self.gArr[index])
        H=np.sum(self.hArr[index])

        H_sum_threshold = 0

        if self.tree_method == 'exact': # exact greedy 算法
            H_sum_threshold=0.0

        elif self.tree_method == 'approx': # 近似算法
            H_sum_threshold = self.sketch_eps*H  #  H_sum_threshold : 二阶梯度分桶的阈值

        loss_old = ( G**2 ) /  ( H + self.reg_lambda )  # 切分前的损失

        for i in range(m):  # 遍历 特征, i 即为特征

            # print('searching the {} feature'.format(i))

            block_i=Blocks['l'][i] # TODO:一个特征 对应一个块, 因此可以并行处理所有的块
            GL=0
            HL=0

            H_sum=0 # 二阶梯度的累计和, 用于判断是否成为分位点

            for j in range(N): # 扫描 block_i 的每一行 : (特征值, 样本标号)
                               #TODO: 两重循环的时间复杂度为 O(mN)

                GL += self.gArr[ int(block_i[j][1]) ] # block_i[j][1] 样本标号
                HL += self.hArr[ int(block_i[j][1]) ] #

                H_sum+=self.hArr[ int(block_i[j][1]) ]

                if  j+1< N and block_i[j][0]!=block_i[j+1][0] and H_sum >= H_sum_threshold:
                # block_i[j][0]!=block_i[j][0]  这一行和 下一行 的特征值 不同, 才能成为候选分位点

                # H_sum >= H_sum_threshold 达到划分桶 的 阈值 ; 若 H_sum_threshold==0 则每一个特征值都是候选切分点, 相当于 精确贪心算法

                # 分位点 增益的计算
                    GR = G - GL  # 只算左边, 右边用总和减去左边得到
                    HR = H - HL

                    if self.min_child_weight < min(HL,HR):

                        loss_new = (GL ** 2) / (HL + self.reg_lambda) + (GR ** 2) / (HR + self.reg_lambda)
                        gain = loss_new - loss_old

                        if gain >= max_Gain:

                            max_Gain = gain
                            Ag = i
                            Ag_split = block_i[j][0] # block_i[j][0] 特征值
                            Ag_split_idx = j

                        # 累计和清零
                        H_sum=0

        if Ag is None or max_Gain <= self.gama: # 未找到 最佳切分点
        # Ag is None 未找到 最佳切分点
        #    max_Gain <= self.gama 考虑 切分的最大的增益是否比 γ(gama) 大，如果小于γ则不进行分裂（预剪枝）

            split_bool=False  #

        else:
            split_bool=True

        return split_bool,Ag, Ag_split,Ag_split_idx, max_Gain



    def update_leaf_region_lable(self, gArr,hArr):
        """
        更新 叶子节点的 预测值

        (1) 需要考虑 会触发 X/0= Nan 的bug

        :param gArr: 损失函数 对 F 的一阶梯度
        :param hArr: 损失函数 对 F 的 二阶梯度
        :return:
        """

        numerator = np.sum( gArr )

        if numerator == 0:
            return 0.0

        denominator = np.sum( hArr ) + self.reg_lambda

        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return - numerator / denominator



    def InitNewNode(self, fmat ,gArr, hArr):

        ndata=fmat.N # 训练集中的样本数目

        for ridx in range(ndata): # ridx - 训练数据的行索引

            #  position < 0 表示删除节点或者不进行继续分裂的节点，存在后者是因为xgboost是按照level逐层进行分裂查找，每层的数据是全量数据，
            # 按照position来分配到expand_分裂节点id上，对于早期某层节点无法继续分裂情况，会对该节点的所有的实例设置
            # position < 0，因此需要对这部分实例进行过滤处理。
            if self.position[ridx] < 0:
                continue

            self.snode[self.position[ridx]].stats.sum_grad += gArr[ridx]
            self.snode[self.position[ridx]].stats.sum_hess += hArr[ridx]

        # evaluator=TreeEvaluator()

        # 按照论文公式计算每个qexpand_节点增益与最优值weight值
        for nid in self.qexpand:  # nid - 节点的标号

            self.snode[nid].weight=TreeEvaluator.CalcWeight(self.params ,self.snode[nid].stats) # 拆分(split)前 节点的Weight
            self.snode[nid].root_gain = TreeEvaluator.CalcGain(self.params ,self.snode[nid].stats) # 拆分(split)前 节点的Gain


    def run_process(self,fid,page,position,qexpand,snode,gArr,hArr):
        """
        执行子进程

        :param fid: 特征标号
        :param page:
        :param position:
        :param qexpand:
        :param snode:
        :param gArr:
        :param hArr:
        :return:
        """
        temp=[]

        # 子进程必须加 try-catch 否则出错了都不知道
        try:
            # 稀疏感知 的前向遍历流程
            pass


        except Exception as e:
            print('Process Error:', e)
        finally:

            return temp



    def UpdateSolution(self,batch,feat_set,gArr, hArr):
        """
        更新solution候选集

        :param batch:
        :param feat_set:
        :param gArr:
        :param hArr:
        :return:
        """
        num_features=len(feat_set)

        #  通过 Pool 进程池 进行并行处理 ( Processes 进程 threads 线程)
        #  每个进程 执行一个特征, 选出对应特征最优的分割值;
        pool = Pool(self.params["nthread"])  # 进程池, 和系统申请 nthread 个进程

        # 每个进程的 返回结果
        stemp = []

        for i in range(num_features):
            fid=feat_set[i]
            page = batch.SortedPages[fid]  # 特征 fid 对应的块

            stemp.append( pool.apply_async(self.run_process, args=( fid,page,self.position,self.qexpand,self.snode ,gArr,hArr )) )   #
            # 参数都 要拷贝一份到 子进程中, page 数据量很大, 导致开销很大
            # TODO: 考虑使用 CPython 实现

        pool.close()
        pool.join()





    def FindSplit(self,depth, gArr, hArr, p_fmat):
        """
        逐层分裂，找到 expand节点 的分裂方案

        :param depth:
        :param gArr:
        :param hArr:
        :param p_fmat:
        :return:
        """

        feat_set= list(range(p_fmat.m)) # TODO: 随机采样特征

        batch=p_fmat # TODO: 分批取训练数据

        self.UpdateSolution(batch,feat_set,gArr, hArr)



    def Update(self, p_fmat, gArr, hArr ):
        """

        :param p_fmat:  特征 X , 使用 DMatrix 表示
        :param gArr: 每一个样本的 损失函数 对 F 的一阶梯度
        :param hArr: 每一个样本的  损失函数 对 F 的 二阶梯度
        :return:
        """
        # 样本所在的树节点的位置 , position[0]=1 样本0 目前位于节点1中
        self.position = [0]*(p_fmat.N) # 初始状况, 所有样本都属于 根节点(节点标号为0)


        # 删除二阶梯度小于0的样本实例，直接 position取反，最高位为1( position[i]的值为负数 ) ，则实例在将来分裂统计会被跳过。
        for ridx in range(p_fmat.N):
            if hArr[ridx]<0:
                self.position[ridx] = ~self.position[ridx]

        # 待分裂的节点
        self.qexpand=[0] #初始状况 只有根节点

        # 节点的统计信息
        self.snode=[NodeEntry()]

        #并行计算，线程间互不影响。每个线程计算的节点分裂信息
        # self.stemp= [ [ThreadEntry()] for _ in range(self.nthread)   ] # 二维数组 , 第一维为 线程号 , 第二维为 ThreadEntry

        # 1. 计算 qexpand队列中所有候选节点的损失函数和权重
        self.InitNewNode( p_fmat , gArr, hArr)

        # 根据树的最大深度进行生长
        # 根据参数param.max_depth逐层分裂生成节点和查找分裂最优方案
        for depth  in range( self.params["max_depth"] ):

            # 2.查找最佳分裂点
            self.FindSplit(depth, gArr, hArr, p_fmat)






    def __predict(self, row):
        """
        预测 一个样本

        :param row:
        :return:
        """

        p = self.root

        while p.label == None:  # 到达 叶子节点 退出循环

            judge_feature = p.feature  # 当前节点划分的 特征
            # judge_feature_name= p.feature_name

            if row[judge_feature] <= p.feature_split:
                p = p.childs[0]
            else:
                p = p.childs[1]

        return p.label

    def predict(self, testDataArr):
        """
        推理 测试 数据集，返回预测结果

        :param test_data:
        :return:
        """

        res_list = []

        for row in testDataArr:
            res_list.append(self.__predict(row))

        return np.array(res_list)

    def score(self, testDataArr, testLabelArr):
        """
        推理 测试 数据集，返回预测 的 平方误差

        :param test_data:
        :return:
        """
        res_list = self.predict(testDataArr)

        square_loss = np.average(np.square(res_list - testLabelArr))  # 平方误差

        return square_loss


class Test:
    pass






if __name__ == '__main__':

    test=Test()

    # test.test_small_dataset()


