import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np


class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):        # 普通Block简单完成两次卷积操作
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)       # 完成一次卷积
        x = self.bn2(self.conv2(x))                             # 第二次卷积不加relu激活函数
        return F.relu(x, inplace=True)                          # 添加激活函数输出


class SpecialBlock(nn.Module):                                  # 特殊Block完成两次卷积操作，以及一次升维下采样
    def __init__(self, in_channel, out_channel, stride):        # 注意这里的stride传入一个数组，shortcut和残差部分stride不同
        super(SpecialBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        return F.relu(x, inplace=True)                          # 输出卷积单元


class Net18(nn.Module):
    def __init__(self):
        super(Net18, self).__init__()
        self.prepare = nn.Sequential(           # 所有的ResNet共有的预处理==》[batch, 64, 56, 56]
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(            # layer1有点特别，由于输入输出的channel均是64，故两个CommonBlock
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(            # layer234类似，由于输入输出的channel不同，故一个SpecialBlock，一个CommonBlock
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))    # 卷积结束，通过一个自适应均值池化==》 [batch, 512, 1, 1]
        self.fc = nn.Sequential(                # 最后用于分类的全连接层，根据需要灵活变化
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)         # 这个使用CIFAR10数据集，定为10分类
        )

    def forward(self, x):
        x = self.prepare(x)         # 预处理

        x = self.layer1(x)          # 四个卷积单元
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)            # 池化
        x = x.reshape(x.shape[0], -1)   # 将x展平，输入全连接层
        x = self.fc(x)

        return x


model = Net18()
print(model)  # 打印模型结构
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 放到GPU上

batch_size = 32

train_data = datasets.CIFAR10("cifar", train=True, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]), download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,)

test_data = datasets.CIFAR10("cifar", train=False, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]), download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, )

lr = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_loss_list = []
test_loss_list = []
test_acc_list = []
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(epoch):
    train_l_sum, cnt = 0.0, 0
    model.train()
    for batch_idx, (x, label) in enumerate(train_loader, 0):
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        train_l_sum += loss.item()
        cnt += 1
    train_loss_list.append(train_l_sum / cnt)
    print("Epoch: ", epoch, "Loss is: ", train_l_sum / cnt, end='')


def test(epoch):
    correct = 0
    total = 0
    cnt = 0
    test_l_sum = 0.0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_l_sum += loss.item()
            cnt += 1
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += torch.eq(pred, labels).float().sum().item()
    test_loss_list.append(test_l_sum / cnt)
    test_acc_list.append(correct / total)
    print(" test_loss:", test_l_sum / cnt, "test_acc: %d %%" % (100 * correct / total))


if __name__ == "__main__":
    epoch_list = []
    test_acc_list = []

    epochs = 10
    for epoch in range(epochs):
        train(epoch)
        test(epoch)
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
            c = (pred == label).squeeze()

            for i in range(16):
                labels = label[i]
                class_correct[labels] += c[i].item()
                class_total[labels] += 1
        acc = total_correct / total_num
        print("Total test_acc:", acc)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    plt.figure(1)
    x = np.linspace(0, epochs - 1, epochs)
    plt.plot(x, train_loss_list, c='r', label="train_loss")
    plt.plot(x, test_loss_list, c='b', label="test_loss")
    plt.legend()
    plt.title("Train and Test loss  lr={}".format(lr))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()