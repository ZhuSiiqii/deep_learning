import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from operator import itemgetter
import gradio as gr
import os
import time


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i:min(i+batch_size, num_examples)])
        yield torch.as_tensor(np.array(itemgetter(*j)(features)), dtype=torch.float32),\
              torch.as_tensor(np.array(itemgetter(*j)(labels)), dtype=torch.float32)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()           # 32*32
        self.conv1 = nn.Conv2d(3, 32, 5, padding='same')       # 32*32
        self.pool1 = nn.MaxPool2d(2, 2)       # 16*16
        self.conv2 = nn.Conv2d(32, 64, 5, padding='same')      # 16*16
        self.pool2 = nn.MaxPool2d(2, 2)       # 8*8
#        self.conv3 = nn.Conv2d(15, 30, 3)     # 5*5

        self.fc1 = nn.Linear(64*8*8, 120)
        self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(120, 84)
        self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
#        x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)  # 展平  x.size()[0]是batch size
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.dp2(x)
        x = self.fc3(x)
        return x


def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1


train_data1 = unpickle("data_batch_1")
train_data1 = list(train_data1.get(b'data'))
train_label1 = unpickle("data_batch_1")
train_label1 = list(train_label1.get(b'labels'))
train_data2 = unpickle("data_batch_2")
train_data2 = list(train_data2.get(b'data'))
train_label2 = unpickle("data_batch_2")
train_label2 = list(train_label2.get(b'labels'))
train_data3 = unpickle("data_batch_3")
train_data3 = list(train_data3.get(b'data'))
train_label3 = unpickle("data_batch_3")
train_label3 = list(train_label3.get(b'labels'))
train_data4 = unpickle("data_batch_4")
train_data4 = list(train_data4.get(b'data'))
train_label4 = unpickle("data_batch_4")
train_label4 = list(train_label4.get(b'labels'))
valid_data = unpickle("data_batch_5")
valid_data = list(valid_data.get(b'data'))
valid_label = unpickle("data_batch_5")
valid_label = list(valid_label.get(b'labels'))
test_data1 = unpickle("test_batch")
test_data1 = list(test_data1.get(b'data'))
test_label1 = unpickle("test_batch")
test_label1 = list(test_label1.get(b'labels'))

train_data = []
train_label = []
train_data.extend(train_data1)
train_data.extend(train_data2)
train_data.extend(train_data3)
train_data.extend(train_data4)

train_label.extend(train_label1)
train_label.extend(train_label2)
train_label.extend(train_label3)
train_label.extend(train_label4)


train_data = np.array(train_data, dtype=float)
train_data.shape = (40000, 3, 32, 32)
train_label = np.array(train_label, dtype=float)
train_data = train_data/255.0

#train_data = torch.from_numpy(train_data)
#train_label = torch.from_numpy(train_label)
#train_data = train_data.type(torch.FloatTensor)
#train_label = train_label.type(torch.FloatTensor)
valid_data = np.array(valid_data, dtype=float)
valid_data.shape = (10000, 3, 32, 32)
valid_data = valid_data/255.0
valid_label = np.array(valid_label, dtype=float)
valid_data = torch.from_numpy(valid_data)
valid_label = torch.from_numpy(valid_label)
valid_data = valid_data.type(torch.FloatTensor)
valid_label = valid_label.type(torch.FloatTensor)

test_data1 = np.array(test_data1, dtype=float)
test_data1.shape = (10000, 3, 32, 32)
test_data2 = test_data1/255.0
test_data3 = test_data2
test_label1 = np.array(test_label1, dtype=float)
test_data = torch.from_numpy(test_data2)
test_label = torch.from_numpy(test_label1)
test_data = test_data.type(torch.FloatTensor)
test_label = test_label.type(torch.FloatTensor)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
print(net)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
lr = 0.001
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)


epochs = 80
batch_size = 64
train_loss_list, valid_loss_list, train_acc_list, valid_acc_list = [], [], [], []
start_time = time.time()
for epoch in range(epochs):
    train_l_sum, train_acc_sum, n, cnt, num_correct = 0.0, 0.0, 0, 0, 0.0

    for i, data in enumerate(data_iter(batch_size, train_data, train_label), 0):  # i 第几个batch     data：一个batch中的数据
        # 输入数据
        inputs, labels = data
        #print(inputs)
        # 梯度清零
        optimizer.zero_grad()
        # forward + backward
        outputs = net(inputs)
        labels = labels.long()
        #print(outputs.dtype, labels.dtype)
        loss = criterion(outputs, labels).sum()
        loss.backward()
        # 更新参数
        optimizer.step()

        train_l_sum += loss.item()
        prediction = torch.argmax(outputs, dim=1)
        num_correct += torch.sum(prediction == labels).item()
        n += labels.size(0)
        cnt += 1
    # 验证集 验证

    valid_predict = net(valid_data)
    valid_label = valid_label.long()
    valid_loss = criterion(valid_predict, valid_label)
    valid_l_sum = valid_loss.item()
    valid_prediction = torch.argmax(valid_predict, dim=1)
    valid_num_correct = torch.sum(valid_prediction == valid_label).item()
    valid_n = valid_label.size(0)

    train_loss_list.append(train_l_sum / cnt)
    train_acc_list.append(num_correct / n)
    valid_loss_list.append(valid_l_sum)
    valid_acc_list.append(valid_num_correct / valid_n)
    print('epoch %d, train_loss %.10f, train_acc %.3f, valid_loss %.10f, valid_acc %.3f' % (epoch + 1, train_l_sum / cnt,
          num_correct / n, valid_l_sum , valid_num_correct / valid_n))
tim = time.time() - start_time
print('Finished Training')
print('total time : %.3f' % tim)
# 测试集 评估
test_predict = net(test_data)
test_label = test_label.long()
test_prediction = torch.argmax(test_predict, dim=1)
test_num_correct = torch.sum(test_prediction == test_label).item()
test_n = test_label.size(0)
print('test_acc = %.3f' % (test_num_correct / test_n))

plt.figure(1)
x = np.linspace(0, epochs-1, epochs)
plt.plot(x, train_loss_list, c='r', label="train_loss")
plt.plot(x, valid_loss_list, c='b', label="valid_loss")
plt.legend()
plt.title("Train and Valid loss  lr={}".format(lr))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.figure(2)
x = np.linspace(0, epochs-1, epochs)
plt.plot(x, train_acc_list, c='r', label="train_acc")
plt.plot(x, valid_acc_list, c='b', label="valid_acc")
plt.legend()
plt.title("Train and Valid precision  lr={}".format(lr))
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


def image_mod(str):
    if str == 'train and valid_loss':
        return 'image/1.png'
    else:
        return 'image/2.png'


demo1 = gr.Interface(
    image_mod,
    gr.Dropdown(
        ["train and valid_loss", "train and valid_acc"], label="Option", info="Please choose one figure to display!"
    ),
    'image',
)


def image_convert(m):
    a = test_data3[int(m)]
    a = a.transpose(1, 2, 0)
    test_data1_tensor = torch.as_tensor(test_data3[int(m)], dtype=torch.float)
    b = test_data1_tensor.unsqueeze(0)
    test_predict_1 = net(b)
    test_predict_1 = test_predict_1.squeeze(0)
    m_predict = torch.argmax(test_predict_1, dim=-1)
    return a, classes[int(test_label1[int(m)])], classes[int(m_predict)]


demo2 = gr.Interface(
    image_convert,
    inputs=['number'],
    outputs=['image', gr.Textbox(label="实际类别"), gr.Textbox(label="预测类别")],

)
demo = gr.TabbedInterface([demo1, demo2], ["LinePlot", "Test"])
demo.launch()