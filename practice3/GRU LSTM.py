import torch
import torch.nn as nn
from torchtext import data
import torchtext
import matplotlib.pyplot as plt
import numpy as np


class GRUNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        vocab_size:词典长度           查找表形状：vocab_size*embedding_dim的矩阵是学习出来的
        embedding_dim:词向量的维度, 等价于 input_size  每个词由embedding_dim个数表示
        hidden_dim: 隐藏层维度，理论上 hidden_size=output_size,最后要转为 output_dim
        layer_dim: GRU的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM ＋ 全连接层
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embeds = self.embedding(x)
        # embeds shape (batch, seq_length, input_size)
        # r_out shape (batch, seq_length, output_size=hidden_size)
        # h_n shape (n_layers, batch, hidden_size)
        r_out, h_n = self.gru(embeds, None)   # None 表示初始的 hidden state 为0
        # 选取最后一个时间点的out输出, 多对一输出，二分类
        out = self.fc1(r_out[:, -1, :])
        return out


TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_md', batch_first=True)
LABEL = data.LabelField(dtype=torch.float)
# 下载数据，训练集、测试集划分，此时数据还是文本形式
train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)
# 建立词典
TEXT.build_vocab(train, max_size=20000, vectors='glove.6B.100d')
LABEL.build_vocab(train)

batchsz = 256
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# 此时按照batch输出的数据是编好码的数字形式，batch * seq_len
trainloader, testloader = data.BucketIterator.splits(
                                (train, test),
                                batch_size = batchsz,
                                device=device
                               )

#print(len(train[0].text))
#print(train[0].label)
# 初始化网络
vocab_size = len(TEXT.vocab.stoi)

embedding_dim = 128
hidden_dim = 128
layer_dim = 1
output_dim = 2

model = GRUNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim)
print(model)
if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
classes = ('pos', 'neg')

def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    cnt1 = 0
    model.train()
    for step,batch in enumerate(trainloader):
#        print(batch)
        x, y = batch.text,batch.label
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        y = y.long()
#        print(x)
#        print(x[0])
#        print(x.shape)
#        print(y.shape)
#        print(y_pred.shape)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
            cnt1 += 1
    #    exp_lr_scheduler.step()
    epoch_loss = running_loss / cnt1
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    cnt2 = 0
    model.eval()
    with torch.no_grad():
        for step,batch in enumerate(testloader):
            x, y = batch.text,batch.label
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            y = y.long()
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
            cnt2 += 1
    epoch_test_loss = test_running_loss / cnt2
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
          'train_loss： ', round(epoch_loss, 5),
          'accuracy:', round(epoch_acc, 5),
          'test_loss： ', round(epoch_test_loss, 5),
          'test_accuracy:', round(epoch_test_acc, 5)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


epochs = 30
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, model, trainloader, testloader)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

model.eval()
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    total_correct = 0
    total_num = 0
    for x, label in testloader:
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
for i in range(2):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

plt.figure(1)
x = np.linspace(0, epochs - 1, epochs)
plt.plot(x, train_loss, c='r', label="train_loss")
plt.plot(x, test_loss, c='b', label="test_loss")
plt.legend()
plt.title("Train and Test loss  lr={}".format(lr))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.figure(2)
x = np.linspace(0, epochs - 1, epochs)
plt.plot(x, train_acc, c='r', label="train_acc")
plt.plot(x, test_acc, c='b', label="test_acc")
plt.legend()
plt.title("Train and Test accuracy  lr={}".format(lr))
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
