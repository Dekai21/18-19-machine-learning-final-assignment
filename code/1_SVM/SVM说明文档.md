# SVM

## 使用环境

- Google Colab
- Python 3.6.7
- Pytorch 1.1.0
- GCC 8.2.0

## 数据加载与预处理

- train_imgs: 训练集图像
- train_label: 训练集标签
- test_imgs: 测试集图像
- test_label: 测试集标签
- mean_image: 图像均值

## 定义LinearClassifier类和LinearSVM类

- train: 训练函数（加载训练集）
- predict: 预测函数
- loss: 计算损失值

## 运行过程

- 实例化SVM
- 对SVM进行训练并测试
- 尝试不同学习率和正则化参数对SVM训练效果的影响
- 可视化SVM的权值矩阵