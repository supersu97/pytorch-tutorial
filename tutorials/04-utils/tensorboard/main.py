import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST 数据集 
dataset = torchvision.datasets.MNIST(root='../../data', 
                                     train=True, 
                                     transform=transforms.ToTensor(),  
                                     download=True)

# 数据加载器
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=100, 
                                          shuffle=True)


# 包含一个隐藏层的全连接神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet().to(device)

logger = Logger('./logs')

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 50000

# 开始训练
for step in range(total_step):
    
    # 重置 data_iter
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # 获取图像和标签
    images, labels = next(data_iter)
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
    
    # 前向传播
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算准确率
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step+1) % 100 == 0:
        print ('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}' 
               .format(step+1, total_step, loss.item(), accuracy.item()))

        # ================================================================== #
        #                           Tensorboard 日志                         #
        # ================================================================== #

        # 1. 记录标量值 (标量摘要)
        info = { 'loss': loss.item(), 'accuracy': accuracy.item() }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)

        # 2. 记录参数值和梯度 (直方图摘要)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

        # 3. 记录训练图像 (图像摘要)
        info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

        for tag, images in info.items():
            logger.image_summary(tag, images, step+1)
