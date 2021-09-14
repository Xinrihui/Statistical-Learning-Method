# 李航老师《统计学习方法》和 深度学习 的相关算法实现

纯手写(模型实现仅使用 numpy 库), 并且和 sklearn 或者 其他机器学习库中的相关模型进行效果比较

## 第一阶段

1.树模型： ID3, CART回归, CART分类

2.提升算法： AdaBoost 回归, AdaBoost 二分类, AdaBoost 多分类(SAMME); GBDT 回归, GBDT 二分类, GBDT 多分类


## 第二阶段

3.提升算法: xgboost 回归, xgboost 二分类, xgboost 多分类

xgboost 单线程版本v1

(1) 递归建立CART回归树, 分裂树节点时, 把样本数据集也根据最佳分裂点进行切割, 分成两个子集合分别给生成的两个子节点


xgboost 单线程版本v2

(1) 对样本数据建立行索引, 并按特征分块, 在块内根据特征值对行索引排序

(2) 递归建立CART回归树,分裂叶子节点时, 对样本数据的行索引根据最佳分裂特征(包括特征值)进行切割, 将最佳特征对应的块分成两个子块

(3) 对于其他特征对应的块, 根据上一步生成的两个子块的行索引进行同步分裂

(性能比陈天奇差2个数量级, 感觉自己离写出完整的xgboost 功力尚不足, 努力向陈天奇看齐吧!)

4.xgboost 多线程版本(待完成)

(1) 对样本数据建立行索引, 并按特征分块, 在块内根据特征值对行索引排序

(2) 记录样本数据行与树节点的关系, 即某样本属于哪个树节点

(3) 基于层次遍历建立CART回归树(非递归), 把待分裂的节点放入队列, 每一次迭代更新(分裂)这一层的所有待分裂的节点;
每分裂一个节点完成后(它已经从队列的头弹出), 把它分裂的两个子节点从队尾加入队列, 显然这两个子节点属于下一层的待分裂节点

(4) 对于待分裂的节点, 对属于该节点的样本的梯度求和, 得到 G(一阶梯度的和)和H(二阶梯度的和)

(5) 遍历所有的特征和对应的特征值(已排序), 通过样本的行索引找到样本对应的待分裂节点,
 在待分裂节点将样本的梯度累加到该节点的 GL 与 HL , 通过 G,H,GL,HL 即可更新待分裂节点的最佳分割点

(6) 在遍历所有的特征时, 可以使用多线程提升效率, 但是由于GIL锁, python多线程最多只能跑满一个CPU核;
 考虑使用多进程, 有下面两个思路:

 a. 在开启新的进程时, 起码要把训练数据复制一份到新的进程中, 要考虑复制数据的开销

 b. 若不复制数据, 使用共享内存, 但是我们发现共享内存效率一般, 估计多进程的效率没有比单进程高多少


## 第三阶段

5.概率图模型: NaiveBayes, GMM, clustering( K-means, Hierachical ) ,HMM , LinearCRF

6.降维(子目录 /PCA_LDA): PCA , LDA

## 第四阶段

7. 线性模型(子目录 /Liner_Models): LinerRegression, logisticRegression

8.神经网络模型(子目录 /MLP): MLP( 二分类, 多分类 ) 并在此基础上实现了神经网络性能优化,
              包括: Xavier 初始化, dropout 正则化, BatchNormalization, 优化算法 Momentum,Adam

9.深度学习模型(子目录 /DNN): CNN, RNN, LSTM


## 重要

本项目使用的数据集下载指南：

1.Mnist数据集, 来自 https://github.com/Dod-o/Statistical-Learning-Method_Code/tree/master/Mnist 下载后进行解压,
然后使用 /transMnist 中的代码转换为 csv 的格式

各个文件夹中, ref 子目录为参考资料

## Ref

https://github.com/Dod-o/Statistical-Learning-Method_Code

https://github.com/dmlc/xgboost

https://github.com/lightaime/cs231n

https://github.com/enggen/Deep-Learning-Coursera