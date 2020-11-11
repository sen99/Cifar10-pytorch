import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from torch.autograd import Variable
import cv2
from Utils import *
from Models import *
from torchsummary import summary
import os
import sys

parser = argparse.ArgumentParser(description='CIFAR10 Detect')
#parser.add_argument('--net', '-n', default = 'Densenet', type = str, help='Select Net for Detect')
parser.add_argument('--net', '-n', default = 'efficientnet_b0', type = str, help='Select Net for Detect')
#parser.add_argument('--net', '-n', default = 'VoVNet99', type=str, help = 'Select Net for Detect')
parser.add_argument('--resume', '-r', action = 'store_true', help='resume from checkpoint')
args = parser.parse_args()
model_name = args.net
print('arg1', args.net)

size_dict = {'darkenet53':32, 
            'Densenet':32, 
            'efficientnet_b0':224, 
            'efficientnet_b1':240, 
            'efficientnet_b2':260, 
            'efficientnet_b3':300, 
            'efficientnet_b4':380,
            'efficientnet_b5':456, 
            'efficientnet_b6':528, 
            'efficientnet_b7':600,}
print("size_dict", size_dict[model_name])
#================== Load Dataset & make Trainloader====================
train_imgs = []
train_labels = []
test_imgs = []
test_labels = []
BATCH_SIZE = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
for images, labels in trainset:
    train_imgs.append(images)
    train_labels.append(labels)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
for images, labels in testset:
    test_imgs.append(images)
    test_labels.append(labels)

class My_datasets(torch.utils.data.Dataset):
    def __init__(self, data, label, size, train='True'):
        self.data = data
        self.label = label
        self.size = size
        self.datanum = len(self.data)
        self.train = train

        if self.size == 32:
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5],  # RGB 平均
                    [0.5, 0.5, 0.5])  # RGB 標準偏差
            ])

        else:
            self.transform_train = transforms.Compose([
                transforms.Resize((self.size+32, self.size+32)),
                transforms.RandomCrop((self.size, self.size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5],  # RGB 平均
                    [0.5, 0.5, 0.5])  # RGB 標準偏差
            ])

        self.transform_test = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],  # RGB 平均
                [0.5, 0.5, 0.5])  # RGB 標準偏差
        ])

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        if self.train == "True":
            out_data = self.transform_train(out_data)
        else:
            out_data = self.transform_test(out_data)

        out_label = torch.tensor(self.label[idx])
        return out_data, out_label

train_dataset = My_datasets(train_imgs, train_labels, size_dict[model_name], train='True')
test_dataset = My_datasets(test_imgs, test_labels, size_dict[model_name], train='False')
my_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
my_testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#================== Confirm Dataset ====================
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#mean, std = get_mean_and_std(my_trainloader)
#print("mean", mean, "std", std)
#plot_hist_and_image(my_trainloader)
#plot_image_after_trans(classes, my_trainloader, my_testloader)

#================== param ====================
num_epochs = 2
start_epoch = 0
best_val_acc = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)
net = eval(model_name + str(())).to(device)
#summary(net, (3, 224, 224))
#summary(net, (3, 32, 32))

def burnin_schedule(epoch):
    if epoch < 100:
        return 0.1
    elif epoch < 200:
        return 0.01
    elif epoch < 300:
        return 0.001
    else:
        return 0.0001

criterion = torch.nn.CrossEntropyLoss()  # ロスの計算
optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=1, weight_decay=0.0005)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=burnin_schedule)

#================== resume ====================
if args.resume:
    PATH = './checkpoint/' + model_name + '.pth'
    checkpoint = torch.load(PATH)
    start_epoch = checkpoint['epoch'] + 1
    net.load_state_dict(checkpoint['net_state_dict'])
    best_val_acc = checkpoint['best_val_acc']
    print("=============checkpoint_data=============")
    print("start_epoch:", start_epoch, " / best_val_acc:", best_val_acc)
    if start_epoch == num_epochs:
        print("Finished num_epochs training, No need to resume")
        sys.exit()

#================== set mixup ====================
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    #print("pred", pred.shape, "y_a", y_a.shape, "y_b", y_b.shape, "lam", lam)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#================== train loop ====================
def train(train_loader):
    net.train()
    running_loss = 0
    train_acc = 0
    train_total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        images, labels_a, labels_b, lam = mixup_data(images, labels)
        images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))

        optimizer.zero_grad()
        det_output = net(images)
        # loss = criterion(det_output, labels)
        loss = mixup_criterion(criterion, det_output, labels_a, labels_b, lam)
        running_loss += loss.item()
        train_total += labels.size(0)
        # train_acc += (det_output.max(1)[1] == labels.to(device)).sum().item()
        # print("a",(lam * (det_output.max(1)[1] == labels_a.to(device)).cpu().sum().float()))
        # print("b",((1 - lam) * (det_output.max(1)[1] == labels_b.to(device)).cpu().sum().float()))
        train_acc += (lam * (det_output.max(1)[1] == labels_a.to(device)).cpu().sum().float() + (1 - lam) * (
                    det_output.max(1)[1] == labels_b.to(device)).cpu().sum().float())
        loss.backward()
        optimizer.step()

    output_and_label.append((det_output, images, labels))
    train_loss = running_loss / len(my_trainloader)
    train_acc = train_acc / train_total

    return train_loss, train_acc

#================== valid loop ====================
def valid(test_loader, epoch):
    global best_val_acc
    net.eval()
    running_loss = 0
    val_acc = 0
    correct = 0
    val_total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            det_output = net(images)
            loss = criterion(det_output, labels.to(device))
            running_loss += loss.item()
            val_total += labels.size(0)
            val_acc += (det_output.max(1)[1] == labels.to(device)).sum().item()
        output_and_label.append((det_output, images, labels))
    val_loss = running_loss / len(test_loader)
    val_acc = val_acc / val_total

    print("val_acc", val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        dir = './checkpoint/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        PATH = dir + model_name + '.pth'
        torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'best_val_acc': best_val_acc,
                    },PATH)
        
    return val_loss, val_acc

#================== train sequence ====================
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
output_and_label = []
for epoch in range(start_epoch, num_epochs):
    output_and_label.clear()
    # print("lr",optimizer.param_groups[0]['lr'])
    train_loss, train_acc = train(my_trainloader)
    val_loss, val_acc = valid(my_testloader, epoch)
    scheduler.step()
    print('epoch %d, loss: %.4f acc: %.4f val_loss: %.4f val_acc: %.4f' % (
    epoch, train_loss, train_acc, val_loss, val_acc))

    # logging
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
print('Finished training')

#================== save the trained model and plot ====================
dir = './saved/' + str(model_name) + str('/')
if not os.path.exists(dir):
    os.makedirs(dir)
save_path = str(dir + 'weights_tuning.pth')
torch.save(net.state_dict(), save_path)
plot_loss_and_accuracy(dir, train_loss_list, val_loss_list, train_acc_list, val_acc_list, num_epochs)


#================== Graphical accuracy of each class =====================
my_testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
load_path = str(dir + 'weights_tuning.pth')
pretrained_dict = torch.load(load_path)
net_dict = net.state_dict()
net.load_state_dict(pretrained_dict)
acc_count = [0] * 10

def test(test_loader):
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            det_output = net(images)
            for j in range(10):
                index = np.where(labels.to('cpu').numpy() == j)
                #print("j",j,"index",index)
                #print("det_output[index]", det_output[index].shape, det_output[index])
                if(det_output[index].shape[0] != 0):
                    #print("det_output[index].max(1)", type(det_output[index].max(1)), det_output[index].max(1))
                    #print("det_output[index].max(1)[1]", type(det_output[index].max(1)[1]), det_output[index].max(1)[1])
                    acc_count[j] += (det_output[index].max(1)[1] == labels[index].to(device)).sum().item() / 1000 * 100
    # plot
    plot_acc_of_each_classes(classes, acc_count)

for epoch in range(1):
    output_and_label.clear()
    test(my_testloader)

