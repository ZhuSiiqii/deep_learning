import pickle
import random
import numpy as np
from mxnet import autograd
from mxnet import ndarray as nd


def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1


def relu(X):
    return nd.maximum(X, 0)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i+batch_size, num_examples)])
        yield features.take(j, 0), labels.take(j)


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition


def net(X, w1, b1, w2, b2):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, w1) + b1)
    return softmax(nd.dot(H, w2) + b2)


def loss(y_hat, y):
    return -nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    return nd.array(y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += nd.array(net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


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
train_data5 = unpickle("data_batch_5")
train_data5 = list(train_data5.get(b'data'))
train_label5 = unpickle("data_batch_5")
train_label5 = list(train_label5.get(b'labels'))

train_data = []
train_label = []
train_data.extend(train_data1)
train_data.extend(train_data2)
train_data.extend(train_data3)
train_data.extend(train_data4)
train_data.extend(train_data5)
train_label.extend(train_label1)
train_label.extend(train_label2)
train_label.extend(train_label3)
train_label.extend(train_label4)
train_label.extend(train_label5)
train_data = nd.array(train_data)
train_label = nd.array(train_label)
# print(train_data)
# print(train_data.shape)
# print(train_label)

num_inputs, num_outputs, num_hiddens = 3072, 10, 256
w1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
w2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [w1, b1, w2, b2]
for param in params:
    param.attach_grad()

lr = 0.000083
num_epochs = 100
batch_size = 500
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in data_iter(batch_size, train_data, train_label):
        with autograd.record():
            y_hat = net(X, w1, b1, w2, b2)
            l = loss(y_hat, y).sum()
        l.backward()
        sgd(params, lr, batch_size)
        # print(w1, b1, w2, b2)

        y = y.astype('float32')
        train_l_sum += l.asscalar()
        train_acc_sum += nd.array(y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size
#    train_l = loss(net(train_data, w1, b1, w2, b2), train_label)
#    print(train_l)
#    print('epoch %d, loss %f' % (epoch + 1, nd.mean(train_l).asnumpy()))
    print('epoch %d, loss %.4f, train acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n))
