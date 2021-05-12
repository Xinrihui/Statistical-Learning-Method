李航老师《统计学习方法》的相关算法实现

第一阶段

1.树模型： ID3 , CART回归, CART分类

2.提升算法： AdaBoost 回归, AdaBoost 二分类, AdaBoost 多分类(SAMME); GBDT 回归, GBDT 二分类, GBDT 多分类

全部代码采用 numpy 库实现，并且 和 sklearn 中的相关模型进行 性能对比, 发现 sklearn 中的树是真的快....

第二阶段

3.提升算法: xgboost 回归, xgboost 二分类, xgboost 多分类

实现了 xgboost 的简化版本, 参考了 xgboost 源码 中一些数值计算的方法, 感觉自己离写出完整的xgboost 功力尚不足, 努力向陈天奇看齐吧

各个文件夹中, ref 子目录为 参考的代码, 就不在这里引用了~