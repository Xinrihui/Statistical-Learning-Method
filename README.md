# 李航老师《统计学习方法》和 深度学习 的相关算法实现

纯手写(模型实现仅使用 numpy 库), 并且和 sklearn 或者 其他机器学习库中的相关模型进行效果比较

## 第一阶段

1.树模型： ID3, CART回归, CART分类

2.提升算法： AdaBoost 回归, AdaBoost 二分类, AdaBoost 多分类(SAMME); GBDT 回归, GBDT 二分类, GBDT 多分类


## 第二阶段

3.提升算法: xgboost 回归, xgboost 二分类, xgboost 多分类

xgboost 单线程版本v1

xgboost 单线程版本v2

4.xgboost 多线程版本(待完成)


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