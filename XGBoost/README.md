
# XGBoost in Python

![avatar](docs/images/numpy_logo.png) 

使用 Python + Numpy  实现了简化版的 XGBoost 模型 

## 项目结构
    .
    ├── dataset                 # 数据集
    ├── docs                    # 参考文献
    ├── logs                    # 记录实验结果 
    ├── notebooks               # jupyter notebook
    ├── ref                     # 参考项目
    ├── src_bak                 # 项目的历史版本的源码
    ├── src_for                 # 项目待实现的版本的源码
    ├── XGBoost_v3              # 当前版本
        ├── lib                 # 模块库
        ├── logs                # 记录实验结果 
        ├── models              # 模型的检查点
        ├── gbtree_xrh.py       # 主程序

## 1.模型设计

### 1.1 xgboost v1 (单线程) 

模型位置: [src_bak/XGBoost_v1_xrh.py](src_bak/XGBoost_v1_xrh.py)

1. 递归建立 CART 回归树, 分裂树节点时, 把样本数据集也根据最佳分裂点进行切割, 分成两个子集合分别给生成的两个子节点


### 1.2 xgboost v2 (单线程) 

模型位置: [src_bak/XGBoost_v2_xrh.py](src_bak/XGBoost_v2_xrh.py)

1. 对样本数据建立行索引, 并按特征分块, 在块内根据特征值对行索引排序

2. 递归建立 CART 回归树,分裂叶子节点时, 对样本数据的行索引根据最佳分裂特征(包括特征值)进行切割, 将最佳特征对应的块分成两个子块

3. 对于其他特征对应的块, 根据上一步生成的两个子块的行索引进行同步分裂

### 1.3 xgboost v3 (单线程) 

项目位置: [XGBoost_v3](XGBoost_v3)

实现的相关特性:

- [x] 分块有序
- [x] 精确贪心算法
- [x] 稀疏感知
- [ ] 近似分位数算法


1. 对样本数据建立行索引, 并按特征分块, 在块内根据特征值对行索引排序

2. 记录样本数据行与树节点的关系, 即某样本属于哪个树节点

3. 基于层次遍历建立 CART 回归树(非递归), 把待分裂的节点放入队列, 每一次迭代更新(分裂)这一层的所有待分裂的节点;
每分裂一个节点完成后(它已经从队列的头弹出), 把它分裂的两个子节点从队尾加入队列, 显然这两个子节点属于下一层的待分裂节点

4. 对于待分裂的节点, 对属于该节点的样本的梯度求和, 得到 G(一阶梯度的和)和H(二阶梯度的和)

5. 遍历所有的特征和对应的特征值(已排序), 通过样本的行索引找到样本对应的待分裂节点,
 在待分裂节点将样本的梯度累加到该节点的 GL 与 HL , 通过 G,H,GL,HL 即可更新待分裂节点的最佳分割点
 
6. 以上两个版本的性能都比陈天奇版差 2 个数量级, 分析原因:
  
  > a. python 作为解释性语言与C++ 的性能差一个数量级
  
  > b. 多线程 和 单线程的性能差一个数量级


### 1.4 xgboost v4 (多线程)  

(待实现)

项目位置: [src_for/XGBoost_v4](src_for/XGBoost_v4)

1. 在遍历所有特征找最优切分点时, 可以使用多线程提升效率; 但是由于GIL锁, python3 多线程最多只能跑满一个CPU核

2. 多进程的实现思路:

  > a. 在开启新的进程时, 起码要把训练数据复制一份到新的进程中, 要考虑复制数据的开销

  > b. 若不复制数据, 使用共享内存, 但是实验发现共享内存效率一般(测试代码见 [src_for/XGBoost_v4/multi_process](src_for/XGBoost_v4/multi_process), 估计多进程的效率没有比单进程高多少


## 2.实验结果


1.Boston 房价数据集 (回归)

> 训练集样本数量：455 

> 测试集样本数量：51

| 版本  |   超参数    | 测试集的 MSE | 训练时长 |
| ---------- | -----------| -----------| -----------|
| xgboost v2 |error_rate_threshold=0.01,max_iter=100,max_depth=3,learning_rate=0.1,gama=1.0,reg_lambda=1.0| 9.51 | 3.2s |
| xgboost v3 |error_rate_threshold=0.01,max_iter=100,max_depth=3,learning_rate=0.1,gama=1.0,reg_lambda=1.0| 10.9 | 26s  |

2.Mnist 数据集 (二分类)

> 训练集样本数量：6000 

> 测试集样本数量：1000

| 版本  |   超参数    | 测试集的 Accuracy | 训练时长 |
| ---------- | -----------| -----------| -----------|
| xgboost v2 |error_rate_threshold=0.01,max_iter=40,max_depth=3,gama=1.0,reg_lambda=1.0,tree_method = 'approx',sketch_eps = 0.3| 0.981 | 589s |
| xgboost v3 |error_rate_threshold=0.01,max_iter=20,max_depth=3,learning_rate=0.1,gama=0.0,reg_lambda=0.0| 0.973 | 203s  |
| xgboost Chen |param= {'eval_metric':'logloss',"eta":0.5,}  num_round = 30  | 0.975 | <2s  |

> xgboost Chen 为陈天奇版

3.Higgs 数据集 (二分类)

> 训练集样本数量：8000 

> 测试集样本数量：2000

| 版本  |   超参数    | 测试集的 Accuracy | 训练时长 |
| ---------- | -----------| -----------| -----------|
| xgboost v2 |error_rate_threshold=0.01,max_iter=30,max_depth=3,gama=1.0,reg_lambda=1.0,tree_method = 'approx',sketch_eps = 0.3| 0.828 | 101s |
| xgboost v2 |error_rate_threshold=0.01,max_iter=30,max_depth=3,gama=1.0,reg_lambda=1.0| 0.829 | 124s  |
| xgboost v3 |error_rate_threshold=0.01,max_iter=20,max_depth=3,gama=0.5,reg_lambda=0.5| 0.822 | 251s  |
| xgboost Chen |param= {"eta": 0.1, "max_depth": 3, "nthread": 16}  num_round = 120  | 0.833 | <5s  |

>  使用 pycharm pro 的 profile 性能分析工具分析 xgboost v3 , 发现最耗时的函数为 find_split(), 占用全部时间的 98%


4.Mnist 数据集 (多分类)

> 训练集样本数量：6000 

> 测试集样本数量：1000

| 版本  |   超参数    | 测试集的 Accuracy | 训练时长 |
| ---------- | -----------| -----------| -----------|
| xgboost v3 |error_rate_threshold=0.01,max_iter=20,max_depth=3,learning_rate=1.0,gama=0.0,reg_lambda=0.0| 0.837 | 7035s  |


## Ref

1. XGBoost: A Scalable Tree Boosting System
2. https://github.com/dmlc/xgboost/

## Note

1. 相关数据集下载详见: [dataset/readme.txt](dataset/readme.txt)

2. 软件环境 [Requirements](requirements.txt)