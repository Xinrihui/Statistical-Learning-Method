

## 1.MLP 二分类 和 多分类

### 1.1 模型设计

    1.实现了向量化的前向传播和后向传播算法

    2.实现激活函数 sigmoid, relu

    3.实现如下优化算法:

     (1) 带正则化(L1, L2)的批量梯度下降(BGD)
     (2) Mini-batch 梯度下降
     (3) 带动量(Momentum)的 Mini-batch 梯度下降
     (4) Adam Mini-batch 梯度下降


    4.实现了 dropout 正则化

    5.实现了 Xavier 模型参数随机初始化

    6.实现了 BatchNormalization 加速模型训练

### 1.2 实验结果

#### 1.2.1 二分类实验

    1.Mnist 数据集(二分类)
    n_train = 60000
    n_test = 10000

    超参数:
    MLP 各个层的维度: layers_dims=[784,50,10,1],
    加入L2正则化: use_reg=2, reg_lambda=0.7,
    learning_rate=1.0,
    max_iter=500
    正确率：0.99
    训练时长： 445s


打印混淆矩阵

```
    [[8962   58]
     [  34  946]]
```

    统计所有类别的 评价指标

```
                  precision    recall  f1-score   support

           0       1.00      0.99      0.99      9020
           1       0.94      0.97      0.95       980

    accuracy                           0.99     10000
   macro avg       0.97      0.98      0.97     10000
weighted avg       0.99      0.99      0.99     10000

```

#### 1.2.2 多分类实验

```
   1.Mnist 数据集(多分类)

    n_train = 6000
    n_test = 1000

    超参数:
    MLP 各个层的维度: layers_dims=[784,100,10],
    使用 sigmoid 激活函数,
    max_iter=5000,
    learning_rate=1

    正确率：0.915
    训练时长： 732s

```

    统计所有类别的 评价指标

```
              precision    recall  f1-score   support

           0       0.92      0.98      0.95        85
           1       0.98      0.99      0.98       126
           2       0.89      0.91      0.90       116
           3       0.91      0.90      0.90       107
           4       0.92      0.92      0.92       110
           5       0.91      0.89      0.90        87
           6       0.93      0.94      0.94        87
           7       0.88      0.87      0.87        99
           8       0.92      0.87      0.89        89
           9       0.89      0.87      0.88        94

    accuracy                           0.92      1000
   macro avg       0.91      0.91      0.91      1000
weighted avg       0.91      0.92      0.91      1000

 ```

    打印混淆矩阵

```
[[ 83   0   1   0   0   0   1   0   0   0]
 [  0 125   0   0   0   0   1   0   0   0]
 [  1   0 106   1   0   0   0   3   4   1]
 [  0   0   2  96   0   5   1   1   1   1]
 [  1   0   2   0 101   0   1   1   0   4]
 [  1   0   2   2   0  77   2   2   1   0]
 [  3   0   0   0   1   1  82   0   0   0]
 [  0   3   5   1   1   1   0  86   0   2]
 [  1   0   1   4   2   0   0   2  77   2]
 [  0   0   0   2   5   1   0   3   1  82]]

 ```

```
   2.Mnist 数据集 (多分类, 未进行二值化处理)
     对样本特征进行二范数归一化

    n_train = 6000
    n_test = 1000

    超参数:

    MLP 各个层的维度: layers_dims=[784,100,10],
    使用 sigmoid 激活函数,
    max_iter=5000,
    learning_rate=1

    正确率：0.895
    训练时长： 498s

```

```
   3.Mnist 数据集(多分类)
    n_train = 6000
    n_test = 1000

    超参数:
    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 sigmoid 激活函数,
    max_iter=2000,
    learning_rate=1.5

    正确率：0.918
    训练时长： 466s

    epcho: 0 , loss:2.30366363763326
    epcho: 100 , loss:2.299524045027727
    ...
    epcho: 1800 , loss:0.01306105662505425
    epcho: 1900 , loss:0.011085172861620318
```

```
   4.Mnist 数据集(多分类)
    n_train = 6000
    n_test = 1000

    超参数:
    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 sigmoid 激活函数,
    加入L2正则化, reg_lambda = 1.0
    max_iter=2000,
    learning_rate=1.5

    正确率：0.915
    训练时长：470s

    epcho: 0 , loss:2.305051495825731
    epcho: 100 , loss:2.300902398485821
    epcho: 200 , loss:2.215275366715295
    ...
    epcho: 1900 , loss:0.1067426462167995
```

```
   5.Mnist 数据集(多分类)
    n_train = 6000
    n_test = 1000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    max_iter=1000,
    learning_rate=0.5
    在训练时发现损失下降的不稳定(损失突然上升), 要降低学习率

    正确率(测试集)：0.931
    train accuracy : 1.0
    训练时长：206s

    参数随机初始化：
    epcho: 0 , loss:2.302552131921045
    epcho: 100 , loss:1.4967671252424792
    ...
    epcho: 900 , loss:0.0043792256852878105

    参数初始化为 0:
    epcho: 0 , loss:2.302585092994046
    epcho: 100 , loss:2.2999865286901278
    epcho: 200 , loss:2.299986528679128
    epcho: 300 , loss:2.299986528679128
    ...

    采用 Xavier参数随机初始化:
    epcho: 0 , loss:2.404079581590424
    epcho: 100 , loss:0.1395417745483575
    epcho: 200 , loss:0.05117907453760403
    epcho: 300 , loss:0.02222596027019951
    ...
    epcho: 900 , loss:0.0024177700380279507

    (1) 使用 Relu 训练时模型的收敛速度比使用 sigmoid 快, 从侧面说明 Relu 解决了sigmoid 的梯度消失问题，
        另外, 当 Relu 进入负半区的时候，梯度为 0，神经元此时不会训练，产生和L1正则化相似的稀疏性

    (2) 在训练集上的准确率比测试集高, 存在模型过拟合

    (3) 若将参数初始化为0, 出现了损失函数不收敛的现象, 这是因为参数 W 初始化为0 导致了 symmetry breaking 问题, 即所有的隐含单元都是对称的,无论你运行梯度下降多久,他
        们一直计算同样的函数

    (4) 使用 Xavier参数随机初始化, 模型的收敛速度更快了

```

```
   6.Mnist 数据集(多分类)
    n_train = 6000
    n_test = 1000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活 函数
    加入L2正则化, reg_lambda = 1.0
    max_iter=1000,
    learning_rate=0.5

    正确率(测试集)：0.929
    train accuracy : 1.0
    训练时长：198s

    epcho: 0 , loss:2.303939990113516
    epcho: 100 , loss:1.5802989343304277
    ...
    epcho: 900 , loss:0.02240750388374305

```

```

   7.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活 函数
    max_iter=1000,
    learning_rate=0.5

    正确率(测试集)： 0.9664
    train accuracy : 0.97805
    训练时长：1782s

    epcho: 0 , loss:2.302548586280129
    epcho: 100 , loss:1.6337596798433889
    ...
    epcho: 900 , loss:0.07400855057278692

```

```

   8.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数
    加入L2正则化,reg_lambda=0.5

    max_iter=1500,
    learning_rate=0.5

    正确率(测试集)： 0.9726
    train accuracy :  0.994
    训练时长：2560s

```

```
   9.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    采用 Xavier参数随机初始化,
    开启 dropout 正则化,keep_prob=0.8

    max_iter=1500,
    learning_rate=0.5

    正确率(测试集)：0.9752
    train accuracy :0.99
    训练时长：3334s

```

```

   10.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    采用 Xavier参数随机初始化,
    开启 dropout 正则化,keep_prob=0.8

    使用 min-Batch 梯度下降
    mini_batch_size = 64
    max_iter=300,
    learning_rate=0.5

    正确率(测试集)：0.9776
    train accuracy :1.0
    训练时长：1352s

    (1) 对比传统的 批量梯度下降(BGD), 使用 min-Batch 梯度下降, 在花了更少的训练时间, 得到了更好的模型

```

```
   11.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    采用 Xavier参数随机初始化,
    开启 L2 正则化, reg_lambda=0.1
    开启 dropout 正则化,keep_prob=0.8

    使用 Momentum 梯度下降,beta=0.9
    mini_batch_size = 640
    max_iter=200,
    learning_rate=0.5

    正确率(测试集)：0.98
    train accuracy :1.0
    训练时长：659s

    epcho: 0 , loss:0.529549536806474
    epcho: 10 , loss:0.17586377053254032
    epcho: 20 , loss:0.16108225052054542
    ...
    epcho: 80 , loss:0.07817365735412371
    epcho: 90 , loss:0.09528053666364614
    epcho: 100 , loss:0.07793552431861211
    ...
    epcho: 180 , loss:0.07096056088437296
    epcho: 190 , loss:0.06513401140319

```

```
   12.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    采用 Xavier参数随机初始化,
    开启 dropout 正则化, keep_prob=0.8

    使用 Adam 梯度下降,  beta1 = 0.9, beta2 = 0.99
    mini_batch_size = 640
    max_iter=100,
    learning_rate=0.01

    正确率(测试集)：0.9758
    train accuracy :0.999
    训练时长： 323s

    epcho: 0 , loss:0.19822740261529814
    ...
    epcho: 80 , loss:0.010728728459043138
    epcho: 90 , loss:0.03472642846836613

    (1) 比较
    Momentum 梯度下降 epcho: 0 , loss:0.529549536806474
    Adam 梯度下降  epcho: 0 , loss:0.19822740261529814
    可以看出Adam 的收敛速度比 Momentum 快

```

```
   13.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    采用 Xavier参数随机初始化,
    开启 dropout 正则化, keep_prob=0.8

    使用 Adam 梯度下降,  beta1 = 0.9, beta2 = 0.99
    mini_batch_size = 512
    max_iter=60,
    learning_rate=0.01

    正确率(测试集)： 0.974
    train accuracy :0.998
    训练时长： 215s
```

```
   14.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    采用 Xavier参数随机初始化,
    关闭 dropout 正则化
    开启 batchnorm , beta1 = 0.9

    使用 Adam 梯度下降,  beta1 = 0.9, beta2 = 0.99
    mini_batch_size = 512
    max_iter=40,
    learning_rate=0.01

    正确率(测试集)： 0.972
    train accuracy :0.998
    训练时长： 157 s

    epcho: 0 , loss:0.1749804396799625
    epcho: 10 , loss:0.06362727361132789
    epcho: 20 , loss:0.01902476595930249
    epcho: 30 , loss:0.06340691019349942

```

```
   15.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    MLP 各个层的维度: layers_dims=[784,200,50,10],
    使用 relu 激活函数,
    采用 Xavier参数随机初始化,
    关闭 dropout 正则化
    开启 batchnorm , beta1 = 0.9

    使用 MinBatch 梯度下降,
    mini_batch_size = 512
    max_iter=40,
    learning_rate=0.01

    正确率(测试集)： 0.95
    train accuracy :0.96
    训练时长： 115 s

    epcho: 0 , loss:0.6794630046009313
    epcho: 10 , loss:0.2413543431679431
    epcho: 20 , loss:0.17893451362036314
    epcho: 30 , loss:0.2736255554879355

```