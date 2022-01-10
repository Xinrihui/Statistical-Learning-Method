
# 李航《统计学习方法》和 深度学习 的相关算法实现

使用 Python + Numpy 实现传统的机器学习模型和深度学习模型

> 模型实现仅使用 Numpy, 并利用 sklearn 中的评价函数对模型效果进行评价


1.树模型( [DecisionTree](DecisionTree) )

> ID3 tree (分类)

> CART tree (回归, 分类)

2.基础提升算法

> AdaBoost( [AdaBoost](AdaBoost) ): 回归, 二分类, 多分类(SAMME) 

> GBDT( [GBDT](GBDT) ): 回归, 二分类, 多分类

3.高级提升算法

> xgboost( [XGBoost](XGBoost) ) :  回归, 二分类, 多分类

4.概率图模型

> NaiveBayes( [NaiveBayes](NaiveBayes) )
 
> GMM( [GMM_EM](GMM_EM) )

> Clustering( [Clustering](Clustering) ):  K-means, Hierachical  

> HMM( [HMM](HMM) )

> LinearCRF( [CRF](CRF) )

5.降维( [PCA_LDA](PCA_LDA) )

> PCA  

> LDA


6.线性模型( [Liner_Models](Liner_Models) ): LinerRegression, logisticRegression

7.神经网络模型( [MLP](MLP) )

> MLP(二分类, 多分类) 

> 神经网络的性能优化

>> Xavier 初始化

>> dropout 正则化

>> BatchNormalization

>> 优化算法: Momentum, Adam

8.深度学习模型( [DNN](DNN) ): CNN, RNN, LSTM


## Ref

https://github.com/Dod-o/Statistical-Learning-Method_Code

## Note

1. 相关数据集下载详见: [dataset/readme.txt](dataset/readme.txt)

2. 软件环境 [Requirements](requirements.txt)
