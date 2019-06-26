# LSTM

## 使用环境

- Google Colab
- Python 3.6.7
- Pytorch 1.1.0
- GCC 8.2.0

## 参数

- device: 根据硬件条件决定是否使用GPU进行加速
- sequence_length: 时间序列
- input_size: 输入的维度
- hidden_size: 隐藏层节点数
- num_layers: 隐藏层数量
- num_classes: 分类种类
- batch_size: 每批使用的样本数量
- num_epochs: 训练的迭代次数
- learning_rate: 学习率

## 数据加载

- train_data: WHU-RS19中手动分出的训练集
- test_data: WHU-RS19中手动分出的测试集
- train_loader: 训练集
- test_loader: 测试集

## 循环神经网络的构建

本循环神经网络由一个2层的循环神经层和一个全连接层组成

- rnn: 循环神经网络层，共2层，各层节点数为1536个
- fc: 全连接层，1536->19

## 定义函数

- train: 训练函数
- test: 测试函数

## 运行过程

- 对构建的循环神经网络进行训练并测试其准确率
- 另外采用不同的learning rate和dropout值，同样进行训练和测试