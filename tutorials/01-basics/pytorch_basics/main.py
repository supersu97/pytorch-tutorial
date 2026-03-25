import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                            目录                                     #
# ================================================================== #

# 1. 自动求导基础示例 1                    (第 25 到 39 行)
# 2. 自动求导基础示例 2                    (第 46 到 83 行)
# 3. 从 numpy 加载数据                     (第 90 到 97 行)
# 4. 数据输入管道                           (第 104 到 129 行)
# 5. 自定义数据集的输入管道                 (第 136 到 156 行)
# 6. 预训练模型                             (第 163 到 176 行)
# 7. 保存和加载模型                         (第 183 到 189 行) 


# ================================================================== #
#                    1. 自动求导基础示例 1                              #
# ================================================================== #

# 创建张量
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# 构建计算图
y = w * x + b    # y = 2 * x + 3

# 计算梯度
y.backward()

# 打印梯度
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 


# ================================================================== #
#                    2. 自动求导基础示例 2                              #
# ================================================================== #

# 创建形状为 (10, 3) 和 (10, 2) 的张量
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 构建全连接层
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# 构建损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# 前向传播
pred = linear(x)

# 计算损失
loss = criterion(pred, y)
print('loss: ', loss.item())

# 反向传播
loss.backward()

# 打印梯度
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 执行一步梯度下降
optimizer.step()

# 你也可以在底层手动执行梯度下降
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# 一步梯度下降后打印损失
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. 从 numpy 加载数据                             #
# ================================================================== #

# 创建 numpy 数组
x = np.array([[1, 2], [3, 4]])

# 将 numpy 数组转换为 torch 张量
y = torch.from_numpy(x)

# 将 torch 张量转换为 numpy 数组
z = y.numpy()


# ================================================================== #
#                        4. 数据输入管道                                #
# ================================================================== #

# 下载并构建 CIFAR-10 数据集
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# 获取一个数据对（从磁盘读取数据）
image, label = train_dataset[0]
print (image.size())
print (label)

# 数据加载器（以非常简单的方式提供队列和线程）
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# 迭代开始时，队列和线程开始从文件加载数据
data_iter = iter(train_loader)

# 小批量图像和标签
images, labels = data_iter.next()

# 数据加载器的实际用法如下
for images, labels in train_loader:
    # 训练代码应写在这里
    pass


# ================================================================== #
#                   5. 自定义数据集的输入管道                            #
# ================================================================== #

# 你应该按照下面的方式构建自定义数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. 初始化文件路径或文件名列表
        pass
    def __getitem__(self, index):
        # TODO
        # 1. 从文件中读取一个数据（例如使用 numpy.fromfile, PIL.Image.open）
        # 2. 预处理数据（例如 torchvision.Transform）
        # 3. 返回一个数据对（例如图像和标签）
        pass
    def __len__(self):
        # 你应该将 0 改为数据集的总大小
        return 0 

# 然后你可以使用预构建的数据加载器
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)


# ================================================================== #
#                        6. 预训练模型                                 #
# ================================================================== #

# 下载并加载预训练的 ResNet-18
resnet = torchvision.models.resnet18(pretrained=True)

# 如果你只想微调模型的顶层，按如下设置
for param in resnet.parameters():
    param.requires_grad = False

# 替换顶层用于微调
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 只是一个示例

# 前向传播
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. 保存和加载模型                                #
# ================================================================== #

# 保存和加载整个模型
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# 仅保存和加载模型参数（推荐）
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
