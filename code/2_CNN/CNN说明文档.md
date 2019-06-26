# CNN

## 使用环境

- Google Colab
- Python 3.6.7
- Pytorch 1.1.0
- GCC 8.2.0

## 参数
* BATCH_SIZE:每批使用的样本个数
* EPOCHS:训练的总代数
* DEVICE:可能的话使用GPU进行加速

## 数据加载
* train_data: WHU-RS19中手动分出的训练集
* test_data: WHU-RS19中手动分出的测试集
* train_loader: 训练集加载
* test_loader: 测试集加载

## 神经网络的构建
网络由两个卷积层,两个全连接层组成
* conv1:卷积层1,输入通道3,输出通道10,卷积核5*5
* pool1: 池化层
* conv2:卷积层2,输入通道10,输出通道20,卷积核5*5
* pool2:池化层
* fc1:全连接层16820->1000
* fc2:全连接层1000->19
forward中
* 第一步后为 62 * 62 * 10
* 第二步后为 29  * 29 * 20
最后使用log_softmax配合下方的nll_loss函数

## 定义函数
实例化网络,使用Adam优化器
* train:训练函数
* test:测试函数

## 运行过程

* 对构建的卷积神经网络进行训练并测试其准确率
* 构建另外两个结构不同的神经网络，同样进行训练和测试