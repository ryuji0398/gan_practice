import torch
from torch import nn
from torch import optim
from torch._C import HOIST_CONV_PACKED_PARAMS
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from models import Generator

from train import spade_dataget
from SPADE.models.pix2pix_model import Pix2PixModel


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)

from collections import OrderedDict
def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

import cv2

def seg_save(i, j, args, dist_seg, data_k):
    seg_path = '/home/noda/SPADE/datasets/train_inst/' + data_k['path'][0].split('/')[-1].split('.')[0]+'.png'
    im = cv2.imread(seg_path)
    cv2.imwrite(dist_seg+'/%d.png'%(i*args.batch+j), im)

def result_save(i, j, args, dist, dist_result, data_k):
    seg_path = '/home/noda/SPADE/datasets/train_inst/' + data_k['path'][0].split('/')[-1].split('.')[0]+'.png'
    im_seg = cv2.imread(seg_path)

    # g_path = dist + '/%d.png'%(i*args.batch+j)
    # breakpoint()
    g_path = dist + '/' + data_k['path'][0].split('/')[-1]
    g_img = cv2.imread(g_path)

    real_path = data_k['path'][0]
    real_img = cv2.imread(real_path)

    im_seg = cv2.resize(im_seg, dsize=(256,256))
    g_img = cv2.resize(g_img, dsize=(256,256))
    real_img = cv2.resize(real_img, dsize=(256,256))

    im_h = cv2.hconcat([g_img, im_seg, real_img])
    # im_h = hconcat_resize_min([g_img, im_seg])
    cv2.imwrite(dist_result+'/%d.png'%(i*args.batch+j), im_h)



def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


if __name__ == "__main__":
    spade_dataloader, opt = spade_dataget()
    spade_model = Pix2PixModel(opt)
    spade_data = spade_dataloader.__iter__()

    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=6)
    parser.add_argument('--end_iter', type=int, default=10)

    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=1, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=100)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=256)
    parser.set_defaults(big=False)
    args = parser.parse_args()

    args.start_iter = 5
    args.end_iter = 5

    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=args.im_size, config_txt=opt)#, big=args.big )
    net_ig.to(device)

    for epoch in [10000*i for i in range(args.start_iter, args.end_iter+1)]:
        ckpt = './models/%d.pth'%(epoch)
        checkpoint = torch.load(ckpt, map_location=lambda a,b: a)

        state_dict = fix_key(checkpoint['g'])
        net_ig.load_state_dict(state_dict)

        # net_ig.load_state_dict(checkpoint['g'])
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print('load checkpoint success, epoch %d'%epoch)

        net_ig.to(device)

        del checkpoint

        dist = 'eval_%d'%(epoch)
        dist = os.path.join(dist, 'img')
        os.makedirs(dist, exist_ok=True)

        dist_seg = 'eval_%d'%(epoch)
        dist_seg = os.path.join(dist_seg, 'seg')
        os.makedirs(dist_seg, exist_ok=True)

        dist_result = 'eval_%d'%(epoch)
        dist_result = os.path.join(dist_result, 'result')
        os.makedirs(dist_result, exist_ok=True)


        with torch.no_grad():
            for i in tqdm(range(args.n_sample//args.batch)):
                noise = torch.randn(args.batch, noise_dim).to(device)

                # segmentation_image get
                data_k = next(spade_data)
                # breakpoint()
                input_semantics, spade_real_image = spade_model.preprocess_input(data_k)

                g_imgs = net_ig(noise, input_semantics)[0]
                g_imgs = F.interpolate(g_imgs, 512)
                for j, g_img in enumerate( g_imgs ):
                    vutils.save_image(g_img.add(1).mul(0.5), 
                        # os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))
                        os.path.join(dist, data_k['path'][0].split('/')[-1]))#, normalize=True, range=(-1,1))
                    # breakpoint()

                    seg_save(i, j, args, dist_seg, data_k)
                    result_save(i, j, args, dist, dist_result, data_k)
