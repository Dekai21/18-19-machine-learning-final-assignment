# kNN

## 使用环境
* Google Colab
* Python 3.6.7
* Pytorch 1.1.0
* GCC 8.2.0

## 数据加载
* train_imgs: 训练集图像
* train_label: 训练集标签
* test_imgs: 测试集图像
* test_label: 测试集标签

## 定义KNearestNeighbor类
* train: 训练函数（加载训练集）
* predict: 预测函数
* compute_distances_one_loop: 计算两幅图像的欧式距离
* predict_labels: 预测图像的标签

## 运行过程

* 实例化kNN
* 测试k = 1时的测试效果
* 尝试不同k值对分类效果的影响并绘制曲线