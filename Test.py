import argparse
import torch
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

parser = argparse.ArgumentParser(description='CIFAR10 Detect')
parser.add_argument('--net', '-n', default = 'Densenet', type = str, help='Select Net for Detect')
args = parser.parse_args()
print('arg1', args.net)

#================== Load Dataset & make Trainloader====================
test_imgs = []
test_labels = []

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
for images, labels in testset:
    test_imgs.append(images)
    test_labels.append(labels)

class My_datasets(torch.utils.data.Dataset):
    def __init__(self, data, label, train='True'):
        self.data = data
        self.label = label
        self.datanum = len(self.data)
        self.train = train

        self.transform_test = transforms.Compose([
            # transforms.Resize((128, 128)),
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

test_dataset = My_datasets(test_imgs, test_labels, train='False')
my_testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

#================== Confirm Dataset ====================
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#mean, std = get_mean_and_std(my_testloader)
#print("mean", mean, "std", std)
#plot_hist_and_image(my_testloader)
#plot_image_after_trans(classes, my_trainloader, my_testloader)

#================== param ====================
best_val_acc = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)
model_name = args.net
net = eval(model_name + str(())).to(device)
#summary(net, (3, 32, 32))

#================== Graphical loss and accuracy in train ====================
dir = './saved/' + str(model_name) + str('/')
if not os.path.exists(dir):
    os.makedirs(dir)

train_loss_list = np.load(dir + "train_loss_list.npy")
train_acc_list = np.load(dir + "train_acc_list.npy")
val_loss_list = np.load(dir + "val_loss_list.npy")
val_acc_list = np.load(dir + "val_acc_list.npy")
plot_loss_and_accuracy(dir, train_loss_list, val_loss_list, train_acc_list, val_acc_list, 3)


#================== Graphical accuracy of each class =====================
my_testloader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)
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
                if(det_output[index].shape[0] != 0):
                    acc_count[j] += (det_output[index].max(1)[1] == labels[index].to(device)).sum().item() / 1000 * 100
    # plot
    plot_acc_of_each_classes(classes, acc_count)

for epoch in range(1):
    test(my_testloader)

