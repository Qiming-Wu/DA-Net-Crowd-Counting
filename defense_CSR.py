import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
import sys
from csrnet import CSRNet
from my_dataset import CrowdDataset
import os
from utils_adv_patch import *
from utils_mean import *
from PIL import Image
from torch.autograd import Variable
from matplotlib import pyplot as plt


def cal_mae(img_root, gt_dmap_root):
    model_path = './saved_models/CSR_Shanghai_A_900.h5'

    model_name = os.path.basename(model_path).split('.')[0]

    # 保存图片
    if not os.path.exists('./defense_CSR'):
        os.mkdir('./defense_CSR')

    if not os.path.exists('./defense_CSR/density_map_adv'):
        os.mkdir('./defense_CSR/density_map_adv')

    if not os.path.exists('./defense_CSR/images_adv_patch'):
        os.mkdir('./defense_CSR/images_adv_patch')

    if not os.path.exists('./defense_CSR/images_adv_ablated'):
        os.mkdir('./defense_CSR/images_adv_ablated')

    model = CSRNet()

    # model.load_state_dict(torch.load(model_param_path))
    trained_model = os.path.join(model_path)
    load_net(trained_model, model)

    model.to(device)
    dataset = CrowdDataset(img_root,gt_dmap_root,8)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    mae = 0
    with torch.no_grad():

        for i,(img,gt_dmap) in enumerate(tqdm(dataloader)):

            patch, mask, patch_shape = init_patch_circle(image_size, patch_size)
            patch_init = patch.copy()
            patch_shape_orig = patch_shape

            img = img.to(device)
            gt_dmap = gt_dmap.to(device)

            im_data_var = Variable(img)

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
            adv_tgt_img = Image.fromarray(adv_tgt_img).convert('RGB')

            adv_tgt_img.save('./defense_CSR/images_adv_patch/{}.png'.format(full_imgname))

            # plt.imsave('./defense_MCNN/images_adv_patch/{}'.format(full_imgname), adv_tgt_img
            # , format='png', cmap='gray')

            img_final = random_mask_batch_one_sample(adv_img, keep, reuse_noise=True)
            img_final_var = Variable(img_final)

            et_dmap=model(img_final_var)

            # *************************************************
            # 存ablated   adv_patch攻击img

            im_fi = img_final_var.data.detach().cpu().numpy()
            im_fi_save = im_fi[0][0]
            plt.imsave('./defense_CSR/images_adv_ablated/{}'.format(full_imgname), im_fi_save
                       , format='png', cmap='gray')

            density_map = density_map.data.detach().cpu().numpy()
            adv_out = density_map[0][0]
            plt.imsave('./defense_CSR/density_map_adv/{}'.format(full_imgname)
                       , adv_out, format='png', cmap=cm.plt.jet)

            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()

            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader)))


def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):

    device=torch.device("cuda")
    model=CSRNet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,8)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        if i==index:
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.jet)
            break


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


if __name__=="__main__":
    device = torch.device('cuda:2')
    print('use cuda ==> {}'.format(device))

    patch_type = 'circle'
    patch_size = 0.16  # 0.02   0.04      0.08     0.16
    image_size = 1024  # height or width of the input image

    mae = 0.0
    mse = 0.0
    dtype = torch.FloatTensor
    keep = 45

    torch.backends.cudnn.enabled=False
    img_root='./data/Shanghai_part_A/test_data/images'
    gt_dmap_root='./data/Shanghai_part_A/test_data/ground_truth'
    # model_param_path='./checkpoints/epoch_124.pth'
    cal_mae(img_root,gt_dmap_root)
    # estimate_density_map(img_root,gt_dmap_root,model_param_path,3)


