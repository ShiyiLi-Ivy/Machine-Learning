只用 numpy 实现的两层神经网络分类器。

代码共有三个文件。主文件 main.py 负责加载 MNIST 数据集并执行训练和测试；network.py 文件包括了定义神经网络、前向传播、后向传播以及测试的类；utils.py 文件里定义了损失函数、激活函数，以及画图函数。

在主函数中，首先加载 MNIST 数据集，然后对标签进行 one-hot 编码。接着定义了一些超参数（如学习率、批量大小、隐藏层大小等）以及一个 Network 类的实例。如果 is_train 为 True，则执行训练并保存模型参数。否则，加载模型参数并执行测试。
- 训练：设置 is_train=True。可以调整的参数包括 weight_decay（L2 正则化），hidden_size（隐藏层大小），lr（学习率），lr_decay（学习率下降）等；
- 测试：设置 is_train=False。模型加载 model_params.npz 并执行测试。

神经网络包含输入层、隐藏层和输出层，其中第一层形状为 784 * hidden_size， 第二层形状为 hidden_size * 10。使用 ReLU 和 sigmoid 作为激活函数。在前向传播中，通过将输入数据（手写数字图像）与权重矩阵相乘，加上偏置，然后通过激活函数计算每个层的输出。在反向传播中，首先计算输出层误差，然后将误差通过权重矩阵向前传递，以计算隐藏层误差。通过这些误差可以计算梯度，并更新权重和偏置。