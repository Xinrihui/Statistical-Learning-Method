
1.在 Mnist 小数据集下对 BN 的效果进行测试

n_train = 6000
n_test = 1000

1.1 关闭 batchnorm, 使用 sigmoid 激活函数, 关闭 dropout, 使用 Random 初始化, MinBatch 梯度下降

clf = MLP_MultiClassifier(K=K,
                          activation='sigmoid',
                          reg_lambda=0.1,
                          use_reg=0,
                          keep_prob=0.8,
                          use_dropout=False,
                          use_batchnorm=False,
                          model_path='model/Mnist.model',
                          use_pre_train=False)

loss_list = clf.fit(X=X, y=y, init_mode='Random',layers_dims=[784, 200, 50, 10], mini_batch_size=128, optimize_mode='MinBatch',
                    max_iter=50, learning_rate=0.1, print_log=True, print_log_step=10)


epcho: 0 , loss:2.2913362819472076
epcho: 10 , loss:2.296272214852705
epcho: 20 , loss:2.3010283340397746
epcho: 30 , loss:2.2967155371093253
epcho: 40 , loss:2.306852027878707

training cost time : 15.843713521957397
test accuracy : 0.126
train accuracy : 0.11183333333333334

可以看出, 模型无法收敛

2.开启batchnorm, 使用 sigmoid 激活函数, 使用 Random 初始化, 关闭 dropout, MinBatch 梯度下降

    clf = MLP_MultiClassifier(K=K,
                              activation='sigmoid',
                              reg_lambda=0.1,
                              use_reg=0,
                              keep_prob=0.8,
                              use_dropout=False,
                              use_batchnorm=True,
                              model_path='model/Mnist.model',
                              use_pre_train=False)

    loss_list = clf.fit(X=X, y=y, init_mode='Random',layers_dims=[784, 200, 50, 10], mini_batch_size=128, optimize_mode='MinBatch',
                        max_iter=50, learning_rate=0.1, print_log=True, print_log_step=10)

epcho: 0 , loss:1.2234454235165149
epcho: 10 , loss:0.24065881463413583
epcho: 20 , loss:0.12748146558173817
epcho: 30 , loss:0.13634269383260594
epcho: 40 , loss:0.13925107366383266

training cost time : 19.003684282302856
test accuracy : 0.875
train accuracy : 0.96

1.3 关闭batchnorm, 使用 sigmoid 激活函数, 关闭 dropout, 使用 Xavier 初始化, MinBatch 梯度下降

    clf = MLP_MultiClassifier(K=K,
                              activation='sigmoid',
                              reg_lambda=0.1,
                              use_reg=0,
                              keep_prob=0.8,
                              use_dropout=False,
                              use_batchnorm=False,
                              model_path='model/Mnist.model',
                              use_pre_train=False)

    loss_list = clf.fit(X=X, y=y, init_mode='Xavier',layers_dims=[784, 200, 50, 10], mini_batch_size=128, optimize_mode='MinBatch',
                        max_iter=50, learning_rate=0.1, print_log=True, print_log_step=10)

epcho: 0 , loss:2.2034911938311024
epcho: 10 , loss:0.8109202859900789
epcho: 20 , loss:0.598857070605953
epcho: 30 , loss:0.35368972827858564
epcho: 40 , loss:0.3853996081858158

training cost time : 15.631614923477173
test accuracy : 0.884
train accuracy : 0.93

1.4 结论

分析上述实验, 在激活函数为 sigmoid 时, BN 对于训练时模型的收敛有很好的效果, 可以避免梯度消失和梯度爆炸,
BN 与 Xavier 初始化有类似的效果


2.在 Mnist 大数据集下对 BN 的效果进行测试

n_train = 60000
n_test = 10000

在上述实验的基础上, 我们加深模型, 探索BN 的功效

2.1 不对样本特征 进行二值化处理

   关闭batchnorm, 使用 relu 激活函数, 使用 Xavier 初始化, MinBatch 梯度下降

       clf = MLP_MultiClassifier(K=K,
                              activation='relu',
                              reg_lambda=0.1,
                              use_reg=0,
                              keep_prob=0.8,
                              use_dropout=False,
                              use_batchnorm=False,
                              model_path='model/Mnist.model',
                              use_pre_train=False)

    loss_list = clf.fit(X=X, y=y, init_mode='Xavier',layers_dims=[784,100,100,100,10], mini_batch_size=256, optimize_mode='MinBatch',
                        max_iter=30, learning_rate=0.1, print_log=True, print_log_step=10)

    epcho: 0 , loss:2.299898771421963
    epcho: 10 , loss:2.2979506205947406
    epcho: 20 , loss:2.3001782744201433

    模型无法收敛, 因为样本特征的数值太大, 导致神经元失活

2.2 不对样本特征进行二值化处理, 而是对其进行二范数归一化

   关闭batchnorm, 使用 relu 激活函数, 使用 Xavier 初始化, MinBatch 梯度下降

    clf = MLP_MultiClassifier(K=K,
                              activation='relu',
                              reg_lambda=0.1,
                              use_reg=0,
                              keep_prob=0.8,
                              use_dropout=False,
                              use_batchnorm=False,
                              model_path='model/Mnist.model',
                              use_pre_train=False)

    loss_list = clf.fit(X=X, y=y, init_mode='Xavier',layers_dims=[784,100,100,100,10], mini_batch_size=256, optimize_mode='MinBatch',
                        max_iter=30, learning_rate=0.1, print_log=True, print_log_step=10)

    epcho: 0 , loss:0.664628702906637
    epcho: 10 , loss:0.10969191041721248
    epcho: 20 , loss:0.0820227704336796

    test accuracy : 0.8639
    train accuracy : 0.9747

    可以看出, 模型虽然能收敛, 但是由于对样本特征二范数归一化, 样本携带的信息出现丢失,导致模型效果较差

2.3 不对样本特征进行二值化处理

    开启batchnorm, 使用 relu 激活函数, 使用 Xavier 初始化, MinBatch 梯度下降

        clf = MLP_MultiClassifier(K=K,
                              activation='relu',
                              reg_lambda=0.1,
                              use_reg=0,
                              keep_prob=0.8,
                              use_dropout=False,
                              use_batchnorm=True,
                              model_path='model/Mnist.model',
                              use_pre_train=False)

       loss_list = clf.fit(X=X, y=y, init_mode='Xavier',layers_dims=[784,100,100,100,10], mini_batch_size=256, optimize_mode='MinBatch',
                        max_iter=30, learning_rate=0.1, print_log=True, print_log_step=10)

        epcho: 0 , loss:0.1807162868203173
        epcho: 10 , loss:0.06229439459996621
        epcho: 20 , loss:0.03642577705856489

        training cost time : 81.65531969070435
        test accuracy : 0.9721
        train accuracy : 0.9961333333333333

2.4 不对样本特征进行二值化处理, 模型再深一点

    开启batchnorm, 使用 relu 激活函数, 使用 Xavier 初始化, Adam 梯度下降

        clf = MLP_MultiClassifier(K=K,
                                  activation='relu',
                                  reg_lambda=0.1,
                                  use_reg=0,
                                  keep_prob=0.8,
                                  use_dropout=False,
                                  use_batchnorm=True,
                                  model_path='model/Mnist.model',
                                  use_pre_train=False)

        loss_list = clf.fit(X=X, y=y, init_mode='Xavier',layers_dims=[784,100,100,100,100,10], mini_batch_size=256, optimize_mode='Adam',
                            max_iter=30, learning_rate=0.01, print_log=True, print_log_step=10)

    epcho: 0 , loss:0.10746958343131718
    epcho: 10 , loss:0.077943122877863
    epcho: 20 , loss:0.04291312715154134

    training cost time : 108.79182386398315
    test accuracy : 0.9775
    train accuracy : 0.9957666666666667

    迭代次数增加为 50
        loss_list = clf.fit(X=X, y=y, init_mode='Xavier',layers_dims=[784,100,100,100,100,10], mini_batch_size=256, optimize_mode='Adam',
                            max_iter=3=50, learning_rate=0.01, print_log=True, print_log_step=10)

    epcho: 0 , loss:0.10746958343131718
    epcho: 10 , loss:0.077943122877863
    epcho: 20 , loss:0.04291312715154134
    epcho: 30 , loss:0.005910426455144666
    epcho: 40 , loss:0.01630172225441473

    training cost time : 179.952294588089
    test accuracy : 0.9796
    train accuracy : 0.9978

2.5 结论

1.使用BN , 可以保证每一层神经元不会失活, 即使是在不对样本进行归一化的情况下, BN yyds!

2.若发现到某个 epcho 损失函数不再下降后, 说明模型已经学不到东西了, 此时可以增加模型的复杂度

3.四大法宝( Xavier, Dropout, Adam, BatchNormalizaion ) 伴身 ,深度学习可以深一点, 再深一点