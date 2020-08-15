from __future__ import print_function
import os
import torch
import numpy as np
import sys
import cv2
import scipy.io as scio
import torchvision.models as models
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
from matplotlib import pyplot as plt

import src.network as network
from src.crowd_count import CrowdCounter
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model
from collections import OrderedDict
from utils_mean import *
import seaborn as sns
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def train_DA(epoch):
    net.train()
    params = list(net.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_loss = 0
    data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=False)
    best_mae = sys.maxsize

    step = -1
    train_loss = 0
    gt_count = 0
    et_count = 0
    for blob in data_loader:
        step = step + 1
        im_data = blob['data']
        gt_data = blob['gt_density']
        dtype = torch.FloatTensor

        # certified input
        im_data = torch.from_numpy(im_data).type(dtype)
        im_data = im_data.to(device)
        im_data = random_mask_batch_one_sample(im_data, keep, reuse_noise=True)
        im_data = Variable(im_data)

        gt_data = torch.from_numpy(gt_data).type(dtype)
        gt_data = gt_data.to(device)
        gt_data = Variable(gt_data)

        density_map = net(im_data, gt_data)
        zzk_loss = net.loss
        train_loss += zzk_loss.item()

        gt_data = gt_data.data.detach().cpu().numpy()
        gt_count = np.sum(gt_data)
        density_map = density_map.data.detach().cpu().numpy()
        et_count = np.sum(density_map)
        print("gt_count: ", gt_count)
        print("et_count: ", et_count)

        optimizer.zero_grad()
        zzk_loss.backward()
        optimizer.step()

    train_loss = train_loss / data_loader.get_num_samples()

    if epoch % 100 == 0:
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
        network.save_net(save_name, net)
    return train_loss


def test_DA(epoch):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    data_path_DA = '../ShanghaiTech/part_A_final/test_data/images/'
    gt_path_DA = '../ShanghaiTech/part_A_final/test_data/after_ground_truth/'

    net.to(device)
    net.eval()
    mae = 0.0
    mse = 0.0

    data_loader_DA = ImageDataLoader(data_path_DA, gt_path_DA, shuffle=False, gt_downsample=True, pre_load=False)

    # 保存图片
    if not os.path.exists('./results_DA_ablated'):
        os.mkdir('./results_DA_ablated')
    if not os.path.exists('./results_DA_ablated/density_map_adv'):
        os.mkdir('./results_DA_ablated/density_map_adv')
    if not os.path.exists('./results_DA_ablated/images_adv'):
        os.mkdir('./results_MCNN_DA/images_adv')
    if not os.path.exists('./results_DA_ablated/images_gt'):
        os.mkdir('./results_DA_ablated/images_gt')

    correct = 0
    total = 0
    dtype = torch.FloatTensor

    # ablated test
    for blob in data_loader_DA:
        im_data = blob['data']
        gt_data = blob['gt_density']
        full_imgname = blob['fname']

        # certified input
        im_data = torch.from_numpy(im_data).type(dtype)
        im_data = im_data.to(device)
        im_data = random_mask_batch_one_sample(im_data, keep, reuse_noise=True)
        im_data = Variable(im_data)

        gt_data = torch.from_numpy(gt_data).type(dtype)
        gt_data = gt_data.to(device)
        gt_data = Variable(gt_data)

        density_map = net(im_data, gt_data)

        density_map = density_map.data.cpu().numpy()
        im_data = im_data.data.cpu().numpy()
        gt_data = gt_data.data.cpu().numpy()

        tgt_img = gt_data[0][0]
        plt.imsave('./results_DA_ablated/images_gt/IMG_{}.png'.format(full_imgname), tgt_img, format='png', cmap='gray')

        adv_tgt_img = im_data[0][0]
        plt.imsave('./results_DA_ablated/images_adv/IMG_{}.png'.format(full_imgname), adv_tgt_img, format='png',
                   cmap=plt.cm.jet)

        adv_out = density_map[0][0]
        plt.imsave('./results_DA_ablated/density_map_adv/IMG_{}.png'.format(full_imgname), adv_out, format='png', cmap='gray')

        et_count = np.sum(density_map)
        gt_count = np.sum(gt_data)

        bias = abs(et_count - gt_count)

        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))

        if bias < 10:
            correct += 1
        total += 1

    accuracy = (correct / total)*100.0
    print("correct: ", correct)
    print("total: ", total)
    mae = mae / data_loader_DA.get_num_samples()
    mse = np.sqrt(mse / data_loader_DA.get_num_samples())
    print("test_ablated_results: ")
    print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))
    print("test_ablated_accuracy: ", accuracy)

    # 保存图片
    if not os.path.exists('./results_DA_normal'):
        os.mkdir('./results_DA_normal')
    if not os.path.exists('./results_DA_normal/density_map_adv'):
        os.mkdir('./results_DA_normal/density_map_adv')
    if not os.path.exists('./results_DA_normal/images_gt'):
        os.mkdir('./results_DA_normal/images_gt')

    total = 0
    correct = 0
    mae = 0.0
    mse = 0.0

    for blob in data_loader_DA:
        im_data = blob['data']
        gt_data = blob['gt_density']
        full_imgname = blob['fname']
        tgt_img = gt_data[0][0]
        plt.imsave('./results_DA_normal/images_gt/{}'.format(full_imgname), tgt_img, format='png', cmap='gray')

        im_data = torch.from_numpy(im_data).type(dtype)
        im_data = im_data.to(device)

        gt_data = torch.from_numpy(gt_data).type(dtype)
        gt_data = gt_data.to(device)
        gt_data = Variable(gt_data)

        density_map = net(im_data, gt_data)

        density_map = density_map.data.detach().cpu().numpy()
        gt_data = gt_data.data.detach().cpu().numpy()

        adv_out = density_map[0][0]
        plt.imsave('./results_DA_normal/density_map_adv/{}'.format(full_imgname), adv_out, format='png', cmap='gray')

        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)

        bias = abs(gt_count - et_count)

        mae += abs(gt_count - et_count)
        mse += (gt_count - et_count) * (gt_count - et_count)

        # ！！！ 具体评判预测成功的指标待定！！！
        if bias < 10:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100.0
    print("correct: ", correct)
    print("total: ", total)
    mae = mae / data_loader_1.get_num_samples()
    mse = np.sqrt(mse / data_loader_1.get_num_samples())
    print("test_normal_result: ")
    print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))
    print("normal_test_accuracy: ", accuracy)


if __name__ == '__main__':
    device = torch.device('cuda:1')
    print('use cuda ==> {}'.format(device))

    method = 'DA-Net'
    dataset_name = 'Shanghai_A_certify'
    output_dir = './saved_models/'
    train_path = './data/shanghaiA_100patches/train/'
    train_gt_path = './data/shanghaiA_100patches/train_den/'

    # end_step = 1000
    start_epoch = 0
    lr = 0.00001
    momentum = 0.9
    # disp_interval = 5000
    # log_interval = 250
    pretrained_vgg16 = False
    fine_tune = False
    rand_seed = 64678
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)
    net = CrowdCounter()
    network.weights_normal_init(net, dev=0.01)
    if pretrained_vgg16:
        vgg16_model = models.vgg16(pretrained=True)
        # vgg16_model.cuda()
        net.DA_Net.copy_params_from_vgg16(vgg16_model)
    net.to(device)

    Loss_list = []
    for epoch in range(start_epoch, start_epoch + 2):
        train_loss = train_DA(epoch)
        Loss_list.append(train_loss)
        if epoch == 1:
            test_DA(epoch)

    # 记录train_loss
    train_loss_txt = open('train_loss.txt', 'w')
    for value in Loss_list:
        train_loss_txt.write(str(value))
        train_loss_txt.write('\n')
    train_loss_txt.close()
