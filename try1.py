# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 定义深度网络类，继承自nn.Module
class MSTARNet(nn.Module):
    # 初始化函数，定义网络结构
    def __init__(self):
        # 调用父类的初始化函数
        super(MSTARNet, self).__init__()
        # 定义卷积层，使用VGG 16网络的前13层作为特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), # 输入通道为1，输出通道为64，卷积核大小为3，填充为1
            nn.ReLU(), # 激活函数为ReLU
            nn.Conv2d(64, 64, 3, padding=1), # 输入通道为64，输出通道为64，卷积核大小为3，填充为1
            nn.ReLU(), # 激活函数为ReLU
            nn.MaxPool2d(2, 2), # 最大池化层，池化核大小为2，步长为2
            nn.Conv2d(64, 128, 3, padding=1), # 输入通道为64，输出通道为128，卷积核大小为3，填充为1
            nn.ReLU(), # 激活函数为ReLU
            nn.Conv2d(128, 128, 3, padding=1), # 输入通道为128，输出通道为128，卷积核大小为3，填充为1
            nn.ReLU(), # 激活函数为ReLU
            nn.MaxPool2d(2, 2), # 最大池化层，池化核大小为2，步长为2
            nn.Conv2d(128, 256, 3, padding=1), # 输入通道为128，输出通道为256，卷积核大小为3，填充为1
            nn.ReLU(), # 激活函数为ReLU
            nn.Conv2d(256, 256, 3, padding=1), # 输入通道为256，输出通道为256，卷积核大小为3，填充为1
            nn.ReLU(), # 激活函数为ReLU
            nn.Conv2d(256, 256, 3, padding=1), # 输入通道为256，输出通道为256，卷积核大小为3，填充为1
            nn.ReLU(), # 激活函数为ReLU
            nn.MaxPool2d(2, 2), # 最大池化层，池化核大小为2，步长为2
        )
        # 定义全连接层，使用VGG 16网络的后三层作为分类器，并将最后一层的输出节点数改为10（MSTAR数据集有10个类别）
        self.classifier = nn.Sequential(
            nn.Linear(65536, 4096), # 输入维度为65536（根据卷积层的输出计算），输出维度为4096
            nn.ReLU(), # 激活函数为ReLU
            nn.Dropout(0.5), # 随机失活层，失活概率为0.5
            nn.Linear(4096, 4096), # 输入维度为4096，输出维度为4096
            nn.ReLU(), # 激活函数为ReLU
            nn.Dropout(0.5), # 随机失活层，失活概率为0.5
            nn.Linear(4096, 10), # 输入维度为4096，输出维度为10（MSTAR数据集有10个类别）
        )

    # 前向传播函数，定义网络的输出
    def forward(self, x):
        # 将输入x通过卷积层，得到特征图
        x = self.features(x)
        # 将特征图展平为一维向量
        x = x.view(x.size(0), -1)
        # 将一维向量通过全连接层，得到输出
        x = self.classifier(x)
        # 返回输出
        return x

# 定义超参数
batch_size = 8 # 批次大小
num_epochs = 10 # 训练轮数
learning_rate = 0.01 # 学习率

# 将网络移动到GPU上，如果有的话
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 检测是否有可用的GPU
# device =  "cpu"# 检测是否有可用的GPU

# 定义数据转换，将图片转换为张量，并进行归一化处理
transform = transforms.Compose([
    transforms.Resize((128, 128)), # 将图片缩放到128 x 128像素
    transforms.ToTensor(), # 将图片转换为张量，并将像素值缩放到[0, 1]区间
    transforms.Normalize((0.5,), (0.5,)) # 将张量进行归一化处理，使其均值为0.5，标准差为0.5
])

# 定义训练数据集，使用torchvision.datasets.ImageFolder类加载图片文件夹，并应用数据转换
train_dataset = torchvision.datasets.ImageFolder(root='C:/Users/VULCAN/Desktop/新建文件夹/作业二/MSTAR-SOC/train', transform=transform)

# 定义测试数据集，使用torchvision.datasets.ImageFolder类加载图片文件夹，并应用数据转换
test_dataset = torchvision.datasets.ImageFolder(root='C:/Users/VULCAN/Desktop/新建文件夹/作业二/MSTAR-SOC/test', transform=transform)

# 定义训练数据加载器，使用torch.utils.data.DataLoader类将训练数据集分批次加载，并打乱顺序
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义测试数据加载器，使用torch.utils.data.DataLoader类将测试数据集分批次加载
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# 创建深度网络对象，使用MSTARNet类的构造函数
model = MSTARNet()
model = model.to(device) # 将模型从CPU内存移动到GPU显存
# 定义损失函数，使用交叉熵损失函数nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

# 定义优化器，使用随机梯度下降优化器torch.optim.SGD()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 定义一个空列表，用于存储每轮训练的平均损失值
train_losses = []

# 定义一个空列表，用于存储每轮测试的平均准确率
test_accs = []

# 进入训练循环，循环次数为训练轮数num_epochs
for epoch in range(num_epochs):
    # 初始化一个变量，用于累计每轮训练的总损失值
    train_loss = 0.0

    # 进入训练数据加载器的迭代循环，每次迭代得到一个批次的数据和标签
    for data, labels in train_loader:
        # 将数据和标签从CPU内存移动到GPU显存（如果有GPU的话）
        data = data.to(device)
        labels = labels.to(device)

        # 将优化器的梯度清零，防止累积梯度影响优化
        optimizer.zero_grad()

        # 将数据输入网络，得到网络的输出
        outputs = model(data)

        # 计算网络的输出和标签之间的损失值
        loss = criterion(outputs, labels)

        # 反向传播损失值，计算网络参数的梯度
        loss.backward()

        # 使用优化器更新网络参数，沿着梯度的反方向进行优化
        optimizer.step()

        # 将本批次的损失值累加到总损失值中
        train_loss += loss.item()

        # 计算每轮训练的平均损失值，并添加到列表中
    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)

    # 打印每轮训练的平均损失值
    print('Epoch %d, train loss: %.3f' % (epoch + 1, train_loss))

    # 初始化一个变量，用于累计每轮测试的正确预测数
    correct = 0

    # 初始化一个变量，用于累计每轮测试的总样本数
    total = 0

    # 进入测试数据加载器的迭代循环，每次迭代得到一个批次的数据和标签
    for data, labels in test_loader:
        # 将数据和标签从CPU内存移动到GPU显存（如果有GPU的话）
        data = data.to(device)
        labels = labels.to(device)

        # 将数据输入网络，得到网络的输出
        outputs = model(data)

        # 从网络的输出中选择最大值作为预测类别
        _, predicted = torch.max(outputs.data, 1)

        # 将本批次的样本数累加到总样本数中
        total += labels.size(0)

        # 将本批次的正确预测数累加到总正确预测数中
        correct += (predicted == labels).sum().item()

    # 计算每轮测试的平均准确率，并添加到列表中
    test_acc = correct / total
    test_accs.append(test_acc)

    # 打印每轮测试的平均准确率
    print('Epoch %d, test accuracy: %.3f' % (epoch + 1, test_acc))

# 绘制训练损失曲线，使用matplotlib.pyplot库的plot函数
plt.plot(train_losses, label='Train Loss')

# 绘制测试准确率曲线，使用matplotlib.pyplot库的plot函数
plt.plot(test_accs, label='Test Accuracy')

# 添加图例，使用matplotlib.pyplot库的legend函数
plt.legend()

# 添加标题，使用matplotlib.pyplot库的title函数
plt.title('MSTAR Classification with Deep Network')

# 添加x轴标签，使用matplotlib.pyplot库的xlabel函数
plt.xlabel('Epoch')

# 添加y轴标签，使用matplotlib.pyplot库的ylabel函数
plt.ylabel('Loss/Accuracy')

# 显示图像，使用matplotlib.pyplot库的show函数
plt.show()

# 定义一个空列表，用于存储真实类别和预测类别
y_true = []
y_pred = []

# 进入测试数据加载器的迭代循环，每次迭代得到一个批次的数据和标签
for data, labels in test_loader:
    # 将数据和标签从CPU内存移动到GPU显存（如果有GPU的话）
    data = data.to(device)
    labels = labels.to(device)

    # 将数据输入网络，得到网络的输出
    outputs = model(data)

    # 从网络的输出中选择最大值作为预测类别
    _, predicted = torch.max(outputs.data, 1)

    # 将真实类别和预测类别添加到列表
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())

# 计算混淆矩阵，使用sklearn.metrics库的confusion_matrix函数
cm = confusion_matrix(y_true, y_pred)

# 打印混淆矩阵
print('Confusion matrix:')
print(cm)