
## xgboost 

### 1.模型构建

#### 1.1 xgboost 单线程版本 v1 

(1) 递归建立CART回归树, 分裂树节点时, 把样本数据集也根据最佳分裂点进行切割, 分成两个子集合分别给生成的两个子节点


#### 1.2 xgboost 单线程版本 v2

(1) 对样本数据建立行索引, 并按特征分块, 在块内根据特征值对行索引排序

(2) 递归建立CART回归树,分裂叶子节点时, 对样本数据的行索引根据最佳分裂特征(包括特征值)进行切割, 将最佳特征对应的块分成两个子块

(3) 对于其他特征对应的块, 根据上一步生成的两个子块的行索引进行同步分裂

#### 1.3 xgboost 单线程版本 v3

(1) 对样本数据建立行索引, 并按特征分块, 在块内根据特征值对行索引排序

(2) 记录样本数据行与树节点的关系, 即某样本属于哪个树节点

(3) 基于层次遍历建立CART回归树(非递归), 把待分裂的节点放入队列, 每一次迭代更新(分裂)这一层的所有待分裂的节点;
每分裂一个节点完成后(它已经从队列的头弹出), 把它分裂的两个子节点从队尾加入队列, 显然这两个子节点属于下一层的待分裂节点

(4) 对于待分裂的节点, 对属于该节点的样本的梯度求和, 得到 G(一阶梯度的和)和H(二阶梯度的和)

(5) 遍历所有的特征和对应的特征值(已排序), 通过样本的行索引找到样本对应的待分裂节点,
 在待分裂节点将样本的梯度累加到该节点的 GL 与 HL , 通过 G,H,GL,HL 即可更新待分裂节点的最佳分割点


(6)单线程版本v2 和 单线程版本v3 的性能都比陈天奇版本差2个数量级, 分析原因:
  a. python 作为解释性语言与C++ 的性能差一个数量级
  b. 多线程 和 单线程的性能差一个数量级


#### 1.4 xgboost 多线程版本 v4 (待实现)

(1) 在遍历所有特征找最优切分点时, 可以使用多线程提升效率; 但是由于GIL锁, python多线程最多只能跑满一个CPU核;
 考虑使用多进程, 有下面两个思路:

  a. 在开启新的进程时, 起码要把训练数据复制一份到新的进程中, 要考虑复制数据的开销

  b. 若不复制数据, 使用共享内存, 但是我们发现共享内存效率一般(测试代码详见 /XGBoostv4/multi_process), 估计多进程的效率没有比单进程高多少


### 2.实验结果

#### 2.1 xgboost 单线程版本v2

    test1: 回归任务
    数据集：boston房价数据集
    参数: 
        error_rate_threshold=0.01, 
        max_iter=100, 
        max_depth=3,
        learning_rate=0.1,
        gama=1.0, 
        reg_lambda=1.0
    训练集数量：455
    测试集数量：51
    测试集的 MSE： 9.51
    模型训练时长：3.2s


    test2: 二分类任务
    数据集：Mnist
    参数:
      error_rate_threshold=0.01
      max_iter=40,
      max_depth=3,
      gama=1.0,
      reg_lambda=1.0,
      tree_method = 'approx',(近似算法)
      sketch_eps = 0.3
    训练集数量：6000
    测试集数量：1000
    正确率：0.981
    模型训练时长： 589s

    test3: 二分类任务
    数据集：Higgs
    参数:
      error_rate_threshold=0.01,
      objective='binary:logistic',
      max_iter=30,
      max_depth=3,
      gama=1.0,
      reg_lambda=1.0,
      tree_method='approx',(近似算法)
      sketch_eps=0.3,
    训练集数量：8000
    测试集数量：2000
    正确率：0.828
    模型训练时长： 101s

    test3: 二分类任务
    数据集：Higgs
    参数:
      error_rate_threshold=0.01,
      objective='binary:logistic',
      max_iter=30,
      max_depth=3,
      gama=1.0,
      reg_lambda=1.0,
    训练集数量：8000
    测试集数量：2000
    正确率：0.829
    模型训练时长： 124s

    对比陈天奇的 xgboost, 在同样的数据规模下, 开启多线程并行, 耗时 <5s, 正确率 Accuracy：0.833, 参数如下:
    param1 = {'objective': 'binary:logistic', "eta": 0.1, "max_depth": 3, "nthread": 16}
    num_round = 120

#### 2.2 xgboost 单线程版本v3

    test1: 回归任务
    数据集：boston房价数据集
    参数: 
        error_rate_threshold=0.01, 
        max_iter=100, 
        max_depth=3,
        learning_rate=0.1,
        gama=1.0, 
        reg_lambda=1.0
    训练集数量：455
    测试集数量：51
    测试集的 MSE：10.9
    模型训练时长：26s

    test2: 二分类任务
    数据集：Mnist
    参数:
      error_rate_threshold=0.01
      max_iter=20,
      max_depth=3,
      gama=0.0,
      reg_lambda=0.0,
      learning_rate=1.0
    训练集数量：6000
    测试集数量：1000
    正确率： 0.973
    模型训练时长： 203s
   
    对比对比陈天奇的 xgboost, 在同样的数据规模下, 开启多线程并行耗时 <2s, 正确率 Accuracy：0.975, 参数如下: 
    param= {'eval_metric':'logloss',"eta":0.5,}  num_round = 30     

    test3: 二分类任务
    数据集：Higgs
    参数:
      error_rate_threshold=0.01,
      objective='binary:logistic',
      max_iter=20,
      max_depth=3,
      gama=0.5,
      reg_lambda=0.5,
    训练集数量：8000
    测试集数量：2000
    正确率：0.822
    模型训练时长：251s
    
    (1) 使用 pycharm pro 的 profile 性能分析工具, 发现最耗时的函数为 find_split(), 占用全部时间的 98%

    test4: 多分类任务
    数据集：Mnist
    参数:
      error_rate_threshold=0.01
      max_iter=20,
      max_depth=3,
      gama=0,
      reg_lambda=0,
      learning_rate=1.0
    训练集数量：6000
    测试集数量：1000
    正确率：0.837 
    模型训练时长：7035s