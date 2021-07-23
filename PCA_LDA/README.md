
## 降维

### 1.PCA 降维

#### 1.1 实验结果

```
   1.Mnist 数据集(二分类)
    n_train = 60000
    n_test = 10000

    特征维度: 784
    降维后的特征维度: 200
    降维耗时: 1.9s
    使用 LR 模型:
            clf = LR_2Classifier(use_reg=0)
            clf.fit(X=X_reduce, y=y, max_iter=50)
    训练耗时: 85s
    测试集的 accuracy: 0.9705

    对比未使用降维
    测试集的 accuracy: 0.974
    训练时长：107s

```

```
   2.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000
    特征维度: 784
    降维后的特征维度: 200
    降维耗时: 1.9s
    使用 LR 模型:
        clf = LR_MultiClassifier(K=K,reg_lambda=0.1,use_reg=2)
        clf.fit(X_reduce, y,max_iter=50,learning_rate=0.1)

    训练耗时: 4.6s
    测试集的 accuracy: 0.83

    对比未使用降维
    测试集的 accuracy: 0.84
    训练时长：30s

```

#### 1.2 结论

PCA 降维后, 模型的训练时间减少, 准确率没有明显下降


### 2.LDA 降维

#### 2.1 模型设计

   1.避免 类内散度矩阵不可逆, 先对样本特征做PCA 的降维
   2. 对于二分类问题, LDA 降维 只能将样本特征降低为 1 维
   3. 对于K分类问题,  LDA 降维 可以选择的维度为: [1, min(K-1, 样本特征的维度)]

#### 2.2 实验结果


```
   1.Mnist 数据集(二分类)
    n_train = 60000
    n_test = 10000

    特征维度: 784
    PCA预降维后的特征维度: 200
    降维后的特征维度:1
    降维耗时: 2.6s
    使用 LR 模型:
            clf = LR_2Classifier(use_reg=0)
            clf.fit(X=X_reduce, y=y, max_iter=50)
    训练耗时: 80s
    测试集的 accuracy: 0.983

    对比未使用降维
    测试集的 accuracy: 0.974
    训练时长：107s

```

```
   2.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000
    特征维度: 784
    PCA预降维后的特征维度: 200
    降维后的特征维度:9
    降维耗时: 2.4s
    使用 LR 模型:
        clf = LR_MultiClassifier(K=K,reg_lambda=0.1,use_reg=2)
        clf.fit(X_reduce, y,max_iter=50,learning_rate=0.1)

    训练耗时: 3.8s
    测试集的 accuracy: 0.847

    对比未使用降维
    测试集的 accuracy: 0.84
    训练时长：30s

```

#### 2.3 结论

因为预先学习了样本, 使用LDA 降维后,除了加快LR模型的训练速度外, 效果还比不降维更好了, 真是6666