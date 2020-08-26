from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from utils_mean import *
import numpy as np
import sys
from torch.autograd import Variable
from matplotlib import pyplot as plt
import src.network as network
from src.crowd_count import CrowdCounter
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model
from prog_bar import *
import seaborn as sns
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
from utils_adv_patch import *
from utils_mean import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:1')
    print('use cuda ==> {}'.format(device))

data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'

# 替换成我的ablated_model
model_path = './saved_models/CMTL_shtechA_900.h5'

model_name = os.path.basename(model_path).split('.')[0]

net = CrowdCounter()

trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.to(device)
net.eval()
mae = 0.0
mse = 0.0
dtype = torch.FloatTensor
keep = 45

patch_type = 'circle'
patch_size = 0.16  # 0.02   0.04      0.08     0.16
image_size = 1024  # height or width of the input image

# load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

# 保存图片
if not os.path.exists('./defense_CMTL'):
    os.mkdir('./defense_CMTL')

if not os.path.exists('./defense_CMTL/density_map_adv'):
    os.mkdir('./defense_CMTL/density_map_adv')

if not os.path.exists('./defense_CMTL/images_adv_patch'):
    os.mkdir('./defense_CMTL/images_adv_patch')

if not os.path.exists('./defense_CMTL/images_adv_ablated'):
    os.mkdir('./defense_CMTL/images_adv_ablated')


if patch_type == 'circle':
    patch, mask, patch_shape = init_patch_circle(image_size, patch_size)
    patch_init = patch.copy()
    patch_shape_orig = patch_shape

correct = 0
total = 0

for blob in data_loader:

    im_data = blob['data']
    gt_data = blob['gt_density']
    # 现在出来的im_data是numpy数组
    full_imgname = blob['fname']

    gt_count = np.sum(gt_data)

    data_shape = im_data.shape

    im_data = torch.from_numpy(im_data).type(dtype)
    im_data = im_data.to(device)
    im_data_var = Variable(im_data)

    gt_data = torch.from_numpy(gt_data).type(dtype)
    gt_data = gt_data.to(device)
    gt_data = Variable(gt_data)

    # 先进行adv_patch添加

    if patch_type == 'circle':
        patch_full, mask_full, _, rx, ry, _ = circle_transform(patch, mask, patch_init, data_shape, patch_shape)

    patch_full, mask_full = torch.FloatTensor(patch_full), torch.FloatTensor(mask_full)

    patch_full, mask_full = patch_full.to(device), mask_full.to(device)

    patch_var, mask_var = Variable(patch_full), Variable(mask_full)

    # 在image上面生成adv patch
    adv_tgt_img_var = torch.mul((1 - mask_var), im_data_var) + torch.mul(mask_var, patch_var)

    adv_img = adv_tgt_img_var.data.cpu().numpy()

# *****************************************************************
# 存adv_patch攻击的img

    adv_tgt_img = adv_img[0][0]
    plt.imsave('./defense_CMTL/images_adv_patch/{}'.format(full_imgname), adv_tgt_img
               , format='png', cmap=plt.cm.jet)

    adv_img = torch.from_numpy(adv_img).type(dtype)
    adv_img = adv_img.to(device)

    # ablated the adv image
    img_final = random_mask_batch_one_sample(adv_img, keep, reuse_noise=True)
    img_final_var = Variable(img_final)

    density_map = net(im_data, gt_data)

# *************************************************
# 存ablated   adv_patch攻击img

    im_fi = img_final_var.data.detach.cpu().numpy()
    im_fi_save = im_fi[0][0]
    plt.imsave('./defense_CMTL/images_adv_ablated/{}'.format(full_imgname), im_fi_save
               , format='png', cmap=plt.cm.jet)

    density_map = density_map.data.detach().cpu().numpy()
    adv_out = density_map[0][0]
    plt.imsave('./defense_CMTL/density_map_adv/{}'.format(full_imgname)
               , adv_out, format='png', cmap='gray')

    et_count = np.sum(density_map)
    mae += abs(gt_count - et_count)
    mse += (gt_count - et_count) * (gt_count - et_count)

    # ！！！ 具体评判预测成功的指标待定！！！
    bias = abs(gt_count - et_count)
    if bias < 10:
        correct += 1
    total += 1

# *****************************************************************************************
# 对patch重新初始化
    new_patch = np.zeros(patch_shape)

    new_mask = np.zeros(patch_shape)

    new_patch_init = np.zeros(patch_shape)

    patch = new_patch

    mask = new_mask

    patch_init = new_patch_init

    patch = zoom(patch, zoom=(1, 1, patch_shape_orig[2] / patch_shape[2], patch_shape_orig[3] / patch_shape[3]),
                 order=1)

    mask = zoom(mask, zoom=(1, 1, patch_shape_orig[2] / patch_shape[2], patch_shape_orig[3] / patch_shape[3]),
                order=0)

    patch_init = zoom(patch_init,
                      zoom=(1, 1, patch_shape_orig[2] / patch_shape[2], patch_shape_orig[3] / patch_shape[3]),
                      order=1)


accuracy = (correct / total) * 100.0
print("correct: ", correct)
print("total: ", total)
mae = mae / data_loader.get_num_samples()
mse = np.sqrt(mse / data_loader.get_num_samples())
print("defense: ")
print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))
print("defense_accuracy: ", accuracy)

# 将结果保存到txt
with open('./Defense_results.txt', 'a') as file_handle:
    file_handle.write('Ablated_MAE_:')
    file_handle.write('\n')
    file_handle.write(str(mae))
    file_handle.write('Ablated_MSE_:')
    file_handle.write('\n')
    file_handle.write(str(mse))
    file_handle.write('correct:')
    file_handle.write('\n')
    file_handle.write(str(correct))
    file_handle.write('total:')
    file_handle.write('\n')
    file_handle.write(str(total))
file_handle.close()
