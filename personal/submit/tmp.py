import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os

import cnn

def load_data():
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    x_train_data = np.load('E:/homework/ML/个人大作业/X_kannada_MNIST_train.npz')['arr_0']
    y_train_data = np.load('E:/homework/ML/个人大作业/y_kannada_MNIST_train.npz')['arr_0']
    x_test_data = np.load('E:/homework/ML/个人大作业/X_kannada_MNIST_test.npz')['arr_0']

    # x_train_data = Image.fromarray(np.uint8(x_train_data))
    # TypeError: Cannot handle this data type    ?? CHW CHW ?
    # x_train_data = data_tf(x_train_data)
    # y_train_data = data_tf(y_train_data)
    # x_test_data = data_tf(x_test_data)

    # 应该x y 一起加载到Dataloader?
    x_train_data = np.array(x_train_data).reshape(x_train_data.shape[0], -1) # 变成一维
    y_train_data = np.array(([y_train_data])).T
    train_data = np.hstack((x_train_data, y_train_data))
    valid_data = train_data[50000:60000]
    train_data = train_data[0:50000]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # 50000
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True) # 10000
    test_loader = DataLoader(x_test_data, batch_size=batch_size, shuffle=False) # 10000
    print("load data finished.")
    return train_loader, test_loader, valid_loader

def printSep():
    print("***********")

def train():
    # some notes:
    # np.zeros((2, 2, 2))   -> np.zeros(shape)
    # np.vstack((arr1, arr2))
    #     ls = np.zeros((2, 2, 2))
    #     np.array: arr.size  --> int
    #     torch.Tensor: ts.size() -->torch.Size([2, 2, 2])  ts.size()[0] == ts.size(0)
    epoch = 0
    train_loss = 0
    train_acc = 0
    global step
    global learning_rate
    for train_data in train_loader:
        step += 1
        img = train_data[:, 0:-1]
        label= train_data[:, -1]
        # weight的维数：torch.Size([32, 1, 3, 3])  img与其一致
        # num, channel, height, width
        img = img.reshape(img.shape[0], 28, 28)
        img = img.unsqueeze(1)
        img = img.type(torch.FloatTensor)
        img = Variable(img).cuda()
        label = label.type(torch.LongTensor).cuda()
        # print("label:", label[0])
        out = model(img)
        print(type(out))
        input()
        loss = criterion(out, label)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        if epoch % 100 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
        if step % 10000 == 0:
            print(step)
            learning_rate = learning_rate * 0.5
        train_loss += loss.data.item() * label.size(0)
        # label.size: 32
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        train_acc += num_correct.item()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(
        train_loss / 50000,
        train_acc / 50000
    ))
    evaluate()


def result():
    model.eval()
    i = 0
    output = open("submission.csv", "w")
    print("id,label", file = output)
    for img in test_loader:
        img = img.unsqueeze(1)
        img = img.type(torch.FloatTensor)
        img = Variable(img).cuda()

        out = model(img)
        _, pred = torch.max(out, 1)
        for num in pred:
            print("%d,%d" % (i + 1, num), file=output)
            i = i + 1

def evaluate():
    # 模型评估
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for valid_data in valid_loader:
        img = valid_data[:, 0:-1]
        label = valid_data[:, -1]
        img = img.reshape(img.shape[0], 28, 28)
        img = img.unsqueeze(1)
        img = img.type(torch.FloatTensor)
        img = Variable(img).cuda()
        label = label.type(torch.LongTensor).cuda()

        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item()*label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Valid Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / 10000,
        eval_acc / 10000
    ))

if __name__ == "__main__":
    batch_size = 32 # 64
    learning_rate = 0.1 # 0.02batch_size
    num_epoches = 20
    step = 0

    model = cnn.CNN()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_loader, test_loader, valid_loader = load_data()
    for epo in range(num_epoches):
        print("epoch:", epo)
        train()
    evaluate()
    result()