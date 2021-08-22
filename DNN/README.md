
# 深度神经网络(DNN)模型

## 1.基于 RNN 的图片标注 (image caption)

### 1.1 模型设计

    从下往上依次为 : 词嵌入层, 图片嵌入层, RNN中间层, 时序仿射输出层

### 1.2 实验结果

    对 Microsoft COCO 数据集 中的所有图片用CNN进行特征提取, 并使用PCA 降维,
    本实验采用降维后的图片特征(n_p=512)作为模型的输入, 详细见 http://cs231n.stanford.edu/slides/2021/lecture_10.pdf (第74页 - 第87页)


## 2.基于 LSTM 的图片标注 (image caption)

### 2.1 模型设计

    从下往上依次为 : 词嵌入层, 图片嵌入层, LSTM中间层, 时序仿射输出层



## Ref

https://github.com/jariasf/CS231n

http://cs231n.stanford.edu/schedule.html


## Note

Microsoft COCO 数据集下载 http://cs231n.stanford.edu/coco_captioning.zip








