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
import os

# get mean value and std value of each 3h of all images in loader
def get_mean_and_std(loader):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in loader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(loader))
    std.div_(len(loader))
    return mean, std

# plot histgram and std of 1st image in loader
def plot_hist_and_image(loader):
    for images, labels in loader:
        print("images", images.shape)
        for i in range(1):
            fig, axs = plt.subplots(2, 1)
            plt.figtext(0.03, 0.9, "figure_and_hist/cdf", backgroundcolor='white')

            img = images[0] / 2 + 0.5
            image = img.numpy()
            axs[0].imshow(np.transpose(image, (1, 2, 0)))

            img = (images[0] / 2 + 0.5) * 256
            hist, bins = np.histogram(img.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            print("axs[1]", axs[1])
            print("ima.flatten()", img.flatten().shape)
            #axs[1].hist(img.flatten(), 256, [0, 256], color='r')
            axs[1].hist(img.flatten(), 256, [0, 256])
            axs[1].plot(cdf_normalized, color='b')
            plt.xlim([0, 256])
            axs[1].legend(('cdf', 'histogram'), loc='upper left')


            plt.show()

        break

# plot first 20 image in trainloader & testloader
def plot_image_after_trans(classes, trainloader, testloader):
    plt.figtext(0.03, 0.9, "trainloader_image", backgroundcolor='white')
    for images, labels in trainloader:
        for i in range(2):
            for j in range(10):
                image = images[i * 10 + j] / 2 + 0.5
                image = image.numpy()
                plt.subplot(4, 10, i * 10 + j + 1)
                plt.imshow(np.transpose(image, (1, 2, 0)))
                plt.title(classes[int(labels[i * 10 + j])])
                plt.tick_params(labelbottom=False,
                                labelleft=False,
                                labelright=False,
                                labeltop=False,
                                bottom=False,
                                left=False,
                                right=False,
                                top=False)
        break

    plt.figtext(0.03, 0.5, "testloader_image", backgroundcolor='white')

    for images, labels in testloader:
        for i in range(2):
            for j in range(10):
                image = images[i * 10 + j] / 2 + 0.5
                image = image.numpy()
                plt.subplot(4, 10, i * 10 + j + 21)
                plt.imshow(np.transpose(image, (1, 2, 0)))
                plt.title(classes[int(labels[i * 10 + j])])
                plt.tick_params(labelbottom=False,
                                labelleft=False,
                                labelright=False,
                                labeltop=False,
                                bottom=False,
                                left=False,
                                right=False,
                                top=False)

        plt.show()
        break

def plot_loss_and_accuracy(dir, train_loss_list, val_loss_list, train_acc_list, val_acc_list, num_epochs):
    print("max(val_acc_list)", max(val_acc_list))
    np.save( dir + "train_loss_list", np.array(train_loss_list))
    np.save( dir + "train_acc_list", np.array(train_acc_list))
    np.save( dir + "val_loss_list", np.array(val_loss_list))
    np.save( dir + "val_acc_list", np.array(val_acc_list))

    fig, axs = plt.subplots(2, 1)
    #plt.figtext(0.03, 0.9, "figure_and_hist/cdf", backgroundcolor='white')

    #axs[0].plot(range(num_epochs), train_loss_list, 'r-', label='train_loss')
    #axs[0].plot(range(num_epochs), val_loss_list, 'b-', label='val_loss')
    axs[0].plot(range(len(train_loss_list)), train_loss_list, 'r-', label='train_loss')
    axs[0].plot(range(len(val_loss_list)), val_loss_list, 'b-', label='val_loss')
    axs[0].legend()
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    plt.grid()

    #axs[1].plot(range(num_epochs), train_acc_list, 'y-', label='train_acc')
    #axs[1].plot(range(num_epochs), val_acc_list, 'g-', label='val_acc')
    axs[1].plot(range(len(train_acc_list)), train_acc_list, 'y-', label='train_acc')
    axs[1].plot(range(len(val_acc_list)), val_acc_list, 'g-', label='val_acc')
    axs[1].legend()
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('loss')
    plt.grid()

    plt.show()

def plot_acc_of_each_classes(classes, acc_count):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(classes, acc_count, color='#3030A0')
    ax.set_ylim(0, 100)
    ax.set_xlabel('class')
    ax.set_ylabel('accuracy %')
    ax.set_title('accuracy for each class')
    plt.show()


