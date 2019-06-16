import torch
import torch.nn as nn
from dataset import ReadData
from torch.utils.data import DataLoader
import torch.optim as optim
from Logger import log
# from cnn import CNN
from lenet5 import Lenet5

#
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        log("调用forward 函数")
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


# 初始化网络
net = CNN()
print(net)

data_train = ReadData(train=True)
data_test = ReadData(train=False)

# num_workers=8 使用多进程加载数据
data_train_loader = DataLoader(data_train, batch_size=10, shuffle=True,)
data_test_loader = DataLoader(data_test, batch_size=5, )


# if net is None:
#     net = CNN()
#     log('获得net')
#     log(net)

# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义网络优化方法
optimizer = optim.Adam(net.parameters(), lr=2e-3)


# 定义训练阶段
def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        log(i)

        # 初始0梯度
        optimizer.zero_grad()
        # 网络前向运行
        output = net(images)
        output = torch.t(output)
        # 计算网络的损失函数
        loss = criterion(output, labels)
        log("损失率"+loss)
        # 存储每一次的梯度与迭代次数
        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
        # 反向传播梯度
        loss.backward()
        # 优化更新权重
        optimizer.step()
    # 保存网络模型结构
    torch.save(net.state_dict(), 'model//' + str(epoch) + '_model.pkl')


def test():
    # 验证阶段
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    # 取消梯度，避免测试阶段out of memory
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            # 计算准确率
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

        avg_loss /= len(data_test)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    for e in range(1,10):
        train_and_test(e)


if __name__ == '__main__':
    main()
