# 基于机器学习的遥感图像识别算法(kNN/SVM/CNN/LSTM)

随着遥感卫星成像技术的提升和机器学习的蓬勃发展，越来越多的研究人员利用机器学习的方法来进行遥感图像识别，取得了很好的效果。在本次作业中，我将利用四种机器学习算法在WHU-RS19数据集上进行遥感图像识别的尝试，这其中既包括传统的kNN和SVM，也包括近年来得到青睐的CNN和LSTM算法。本文的基本结构如下：

* 数据集
  * WHU-RS19的简单介绍
  * 数据集的预处理与索引文档的生成
* kNN
  * kNN的测试效果
  * 分析参数k对kNN的测试效果的影响
* SVM
  * SVM的测试效果
  * 分析学习率和正则化参数对SVM的测试效果的影响
  * SVM权值矩阵的可视化
* CNN
  * CNN的测试效果
  * 不同网络结构对CNN的测试结果的影响
* LSTM
  * LSTM的测试效果
  * 分析学习率和dropout值对LSTM的测试效果的影响
* 总结



## 数据集

### WHU-RS19的简单介绍

本次遥感图像识别算法采用的数据集是武汉大学提供的WHU-RS19数据集，该数据集包含了机场，海滩，桥，商业区，沙漠，农田，足球场，森林，工业区，草地，山，公园，停车场， 池塘， 港口， 火车站， 住宅区， 河流和高架桥总共19类遥感图像。图像的分辨率大都为600×600，每一个种类大约有50张图像。

<div align=center><img width="260" height="260" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/bridge_17.jpg"/></div>

<div align=center> 图1 bridge_17           </div>

<div align=center><img width="260" height="260" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/footballField_13.jpg"/></div>

<div align=center>图2 footballField_13</div>

### 数据集的预处理与索引文档的生成

在原始的数据集中，有4张分辨率不是600×600的图像已被去除。

利用 split_dataset.py 将数据集按照 0.8: 0.2 的比例分为训练集和测试集，分别置于train文件夹和test文件夹中。

利用 generate_txt.py 分别生成训练集和测试集的索引文件，索引文件中包括了图片的路径和图片的标签（0~19）。由于后面的实验在Google Colab上进行，因此我手动统一修改了图片的路径。最后得到的索引文件分别为train.txt和test.txt，其内容如下图所示：

<div align=center><img width="320" height="320" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/traintxt.png"/></div>

<div align=center>图3 train.txt</div>

在本次作业中，我采用了4种不同的机器学习方法进行遥感图像识别的尝试，分别是kNN、SVM、CNN和LSTM。以上所有的实验均在Google Colab平台上进行。

## kNN

kNN（k-邻近算法）是最为简单的机器学习算法。在kNN算法中，一个对象的分类是由其邻居的“多数表决”确定的，k个最近邻居（k为正整数，通常较小）中最常见的分类决定了赋予该对象的类别。若k = 1，则该对象的类别直接由最近的一个节点赋予。

### kNN的测试效果

kNN作为一种最简单的机器学习算法，我并未对其的测试效果报以太大的期望。在本次测试中，我先将k取为1，测试这种最简单的模式下的效果，最终其测试的准确率为16%。

### 分析参数k对kNN的测试效果的影响

采用kNN算法需要重点关注的是k值的选取。一般情况下，在分类时较大的K值能够减小噪声的影响，但会使类别之间的界限变得模糊。因此本实验分别尝试了k = 1，3，5，10，15下kNN算法的测试精度。实验的结果如下所示：

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/knn1.png"/></div>

<div align=center>图4 kNN在不同k参数下的测试效果</div>

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/knn2.png"/></div>

<div align=center>图5 kNN在不同k参数下的测试效果</div>

由实验结果可知，和预期的相同，kNN算法在WHU-RS19数据集上的表现效果非常的一般。当k = 1时，算法取得了最高的分类精度仅为16%，随着k取值的进一步增大，分类器的效果不断下降，当k = 10时，该分类器基本稳定在10%左右。本人推测，造成k值增大导致分类效果明显下降的现象的原因在于数据集中不同类别之间的界限本来就比较模糊，增大k值进一步加剧了这种现象。

## SVM

在机器学习中，SVM是一种常用的监督学习算法，其目的在于寻找一个超平面，能够以最大间隔将各类数据分开。作为传统的机器学习算法中表现非常优秀的一种算法，SVM在许多场景中都得到了应用。

### SVM的测试效果

本实验的目的在于观察SVM在WHU-RS19数据集上的表现。

首先采用了1e-7的学习率和2.5e4的正则化因子进行3500次训练，最后获得了21%的测试精度，训练过程如下所示：

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/svm1.png"/></div>
<div align=center>图6 SVM训练过程中的损失值变化与测试精度</div>

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/svm2.png"/></div>
<div align=center>图7 SVM训练过程中的损失值变化</div>

### 分析学习率和正则化参数对SVM的测试效果的影响

由于不同的学习率和正则化参数的取值会对SVM的训练结果造成明显的影响，本实验采用了[1e-08, 1e-07, 1e-06]三种学习率以及[1e04, 2.5e04, 5.0e04]三种正则化参数共9种参数组合进行测试，经过训练之后的测试精度分别如下所示：

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/svm4.png"/></div>
<div align=center>图8 9组学习率和正则化参数组合下的测试精度</div>

测试结果显示，1e-08的学习率和2.5e04正则化参数组合下的效果最好，SVM在训练后的测试精度达到了28%。

但是在后续的测试中，我也发现1e-08的学习率似乎有些过于小了，因为在训练过程中其损失值的下降往往非常缓慢，有几次也陷入了局部最小值中。因此我认为在本数据集中，1e-07的学习率是更为可取的。

### SVM权值矩阵的可视化

在SVM这部分的最后，我将SVM训练后学习到的权值矩阵进行了可视化，其效果如下所示：

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/svm3.png"/></div>
<div align=center>图9 SVM权值矩阵的可视化效果</div>

可以看出，各个权值矩阵的可视化图像反映出了该类遥感图像的一些特征。例如，森林类（Forest）的权值矩阵的可视化图像呈现出深绿色，而草地类（Meadow）则为草绿色；在足球场（Football Field）这类图像中，我们可以看出足球场的鸟瞰图的基本轮廓。

# CNN

卷积神经网络是目前图像识别中最为流行的机器学习算法，本实验的目的在于了解CNN在WHU-RS19数据集上的表现效果，并观察网络结构的变化对训练结果的影响。

## CNN的测试效果

本实验先构建了一个有两层卷积层的神经网络，各层的参数如下所示：

- conv1:卷积层1,输入通道3,输出通道10,卷积核5*5
- pool1: 池化层
- conv2:卷积层2,输入通道10,输出通道20,卷积核5*5
- pool2:池化层
- fc1:全连接层16820->1000
- fc2:全连接层1000->19

该CNN网络的训练过程如下所示：

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/cnn1.png"/></div>
<div align=center>图10 CNN网络每次迭代训练后的测试精度</div>

可以看出，经过10次迭代训练之后，CNN已经取得了55%的测试精度，这明显优于上文提及的kNN和SVM。

## 不同网络结构对CNN的测试结果的影响

考虑到CNN的网络结构会对训练结果造成明显的影响，本实验尝试构建了另外两个不同结构的神经网络：CNN_net_2和CNN_net_3。前者拥有相同的卷积层数和全连接层数，但是两个卷积层拥有更多的输出通道，即使用了数量更多的卷积核；后者则使用了3层卷积层进行训练。这两个CNN网络的参数如下所示：

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/cnn2.png"/></div>
<div align=center>图11 CNN_net_2</div>

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/cnn3.png"/></div>
<div align=center>图12 CNN_net_3</div>

但令人失望的是，这两个网络结构的表现都非常差，在迭代训练中完全没有学习到太多信息，最后的测试精度均低于随机猜测的精度（1/19 = 5.3%）

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/cnn4.png"/></div>
<div align=center>图13 CNN_net_2在每次迭代训练后的测试精度</div>

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/cnn5.png"/></div>
<div align=center>图14 CNN_net_3在每次迭代训练后的测试精度</div>

## LSTM

递归神经网络在语音识别、图像识别中得到了广泛的应用。在之前的期中作业中，我采用了RNN、LSTM和GRU等递归神经网络在MNIST、FashionMNIST、CIFAR10等比较简单的数据集上进行了测试。本实验中使用的WHU-RS19数据集的识别难度明显大于以上提及的几个数据集，通过该实验，我们也可以横向对比LSTM与其他机器学习算法的性能差异。

### LSTM的测试效果

本实验先选用0.00005的学习率和0.5的dropout值进行训练，最后获得了36.4%的测试精度。训练过程中的损失值的变化如下图所示：

<div align=center><img width="420" src="https://github.com/DICKIEZhu/Machine_learning_remote_sensing/raw/master/figure/lstm.png"/></div>
<div align=center>图15 LSTM网络在训练过程中的损失值变化</div>

### 分析学习率和dropout值对LSTM的测试效果的影响

在本次实验中，分别采用0.000005, 0.00005, 0.0005的学习率和0.3, 0.5, 0.7的dropout值总共3×3组超参数进行训练，训练的结果如下表所示：

| learning_rate | dropout | test accuracy |
| ------------- | ------- | ------------- |
| 0.000005      | 0.3     | 19.4%         |
| 0.000005      | 0.5     | 19.4%         |
| 0.000005      | 0.7     | 21.4%         |
| 0.00005       | 0.3     | 38.8%         |
| 0.00005       | 0.5     | 36.4%         |
| 0.00005       | 0.7     | 35.9%         |
| 0.0005        | 0.3     | 41.3%         |
| 0.0005        | 0.5     | 26.2%         |
| 0.0005        | 0.7     | 40.7%         |

## 总结

在以上的实验中，我在WHU-RS19数据集上采用了kNN、SVM、CNN和LSTM这四种机器学习算法进行遥感图像识别。从测试准确率来看，CNN和LSTM的效果要优于传统的kNN和SVM；相比之下，CNN（55%）的效果也比LSTM（43%）要好上不少，这可能和CNN利用卷积核更能有效地提取图像局部特征的优点有关。除此之外，我还观察了各算法中的参数调节对测试准确率的影响，并做了一些可视化的工作。

限于本人的水平，本次实验中也存在一些不足，关于网络结构和各种超参数对机器学习效果的影响，我的分析并不够深入透彻。而且本次采用的WHU-RS19数据集也偏小，如果采用更大的数据集，以上几种算法的效果应该还可以得到进一步地提升，但是由于邻近期末考试，本次实验的时间实在有限，未能尝试使用更大的数据集，这部分工作也留待未来进一步进行。

最后我想感谢何良华老师和机器学习这门选修课。作为一名汽车电子的本科生，我一直非常希望从事高级辅助驾驶（ADAS）或者自动驾驶（Autonomous Driving）这方面的研究或者工作，这要求我必须掌握足够的图像识别的知识，并拥有扎实的代码水平，但是在我的本科课程中却鲜有这方面的课程，这正是我这个学期选修机器学习这门课程的原因。通过这学期的学习和两次大作业的完成，我理解了几种常用机器学习的数学原理，也对Pytorch有了一些初步了解，特别是两次大作业，让我不得不感叹”纸上学来终觉浅，绝知此事要coding“。以上总总收获，都让我受益匪浅，相信这对我接下来研究生阶段的工作肯定有着不小的帮助。

## 链接
Google Colab链接: https://drive.google.com/drive/folders/1_cETCiwXJ3JxE_TnyzWJZwSl6wAisbMj?usp=sharing
