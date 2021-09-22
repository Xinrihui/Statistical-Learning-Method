
# 深度神经网络(DNN)模型

## 1.基于 RNN 的图片注释 (image caption)

### 1.1 模型设计

    从下往上依次为 : 词嵌入层, 图片嵌入层, RNN中间层, 时序仿射输出层

    模型结构详见 http://cs231n.stanford.edu/slides/2021/lecture_10.pdf (第74页 - 第87页)


### 1.2 实验结果

    对 Microsoft COCO 数据集 中的所有图片用CNN(VGG-16)进行特征提取, 并使用PCA 降维,本实验采用降维后的图片特征(n_p=512)作为模型的输入



## 2.基于 LSTM 的图片注释 (image caption)

### 2.1 模型设计

    从下往上依次为 : 词嵌入层, 图片嵌入层, LSTM中间层, 时序仿射输出层

### 2.2 实验结果

## 3. 基于 CNN 的图片分类器

### 3.1 模型设计

    参考 LeNet-5 网络

    0. 图片输入层 'input'
        output: shape (N,1,28,28), 28*28=784

    1. 卷积层 'conv1'

        config_conv1 = { 'f':3, 's':1, 'p':1, 'n_c':6 }
        'f' - 卷积核大小, 's' -窗口滑动步长, 'p' - padding填充的个数

                         N  C   H   W
        input :  shape ( N, 1, 28, 28)
        output : shape ( N, 6, 28, 28)

        N - 样本个数, C - 通道个数, H - 特征图的高度, W - 特征图的宽度

    2. relu 激活层 'relu1'

    3. 最大池化层 'max_pool1'

        config_pool1 = {'f':2, 's':2}
        'f' - 池化核大小, 's' -窗口滑动步长

        input :  shape (N,6,28,28)
        output : shape (N,6,14,14)


    4. 卷积层 'conv2'

    config_conv1 = { 'f':5, 's':1, 'p':0, 'n_c':16 }
    'f' - 卷积核大小, 's' -窗口滑动步长, 'p' - padding填充的个数

                     N  C   H   W
    input :  shape ( N, 6, 14, 14)
    output : shape ( N, 16, 10, 10)

    N - 样本个数, C - 通道个数, H - 特征图的高度, W - 特征图的宽度

    5. relu 激活层 'relu2'

    6. 最大池化层 'max_pool2'

        config_pool1 = {'f':2, 's':2}
        'f' - 池化核大小, 's' -窗口滑动步长

        input :  shape (N,16,10,10)
        output : shape (N,16,5,5)

    7. 全连接层 'affine1'

        input :  shape (N,16,5,5) , 15*5*5 = 400
        output : shape (N,10)

### 3.2 实验结果

   1.Mnist 数据集(多分类)
    n_train = 60000
    n_test = 10000

    使用 relu 激活函数
    使用 Adam 梯度下降, beta1 = 0.9, beta2 = 0.99
    mini_batch_size = 512
    num_epochs=5
    learning_rate= 5e-3

    test accuracy：0.9784
    train accuracy :0.9825
    训练时长： 514s



## Ref

https://github.com/jariasf/CS231n

http://cs231n.stanford.edu/schedule.html


## Note

1.Microsoft COCO 数据集

stanford cs231n 预处理过的数据集下载 http://cs231n.stanford.edu/coco_captioning.zip

标准的数据集下载 https://cocodataset.org/#download







