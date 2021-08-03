from __future__ import division

import glob
import os

import numpy as np
import torch
import torch.utils.data as data
from scipy.ndimage import imread

from datasets import pms_transforms

np.random.seed(0)


class Dragon(data.Dataset) :
    def __init__(self, args, split='test') :
        self.root = os.path.join(args.bm_dir)
        self.split = split
        self.args = args
        self.mtrl = sorted(glob.glob(os.path.join(args.bm_dir) + "/*/"))
        # self.l_dir = sorted(glob.glob(os.path.join(args.bm_dir)+"/*/"))
        print('[%s Data] \t%d mtrl  Root: %s' %
              (split, len(self.mtrl), self.root))
        self.mtrl_type = args.mtrl_type

    def _getMask(self) :
        mask = imread(os.path.join(self.root, 'mask.png'))
        if mask.ndim > 2 : mask = mask[:, :, 0]

        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def __getitem__(self, index) :
        np.random.seed(index)
        select_idx = np.arange(82)[:self.args.in_img_num]
        mtrl = self.mtrl[index]
        # intens_path = os.path.join(mtrl+"LCNet_light_ints.txt")
        intens_path = os.path.join(mtrl + "gt_light_ints.txt")
        intens_a = np.loadtxt(intens_path)
        intens = [np.diag(1 / intens_a[i]) for i in select_idx]
        # directions_path =  os.path.join(mtrl+"LCNet_light_dirs.txt")
        # directions = np.loadtxt(directions_path)
        directions = []

        imgs = []
        for i in select_idx :
            idx = '%03d' % (i)
            # light=",".join([str(x) for x in directions[i]])
            # print(index,light)
            img_name = glob.glob(f"{mtrl}/l_{idx}*.png")[0]
            img = imread(img_name).astype(np.float32) / 255.0
            # img = np.dot(img, intens[i])

            directions.append([float(x) for x in img_name.split("/")[-1].split("_")[1].split(",")[-3 :]])
            imgs.append(img)
        img = np.concatenate(imgs, 2)
        directions = np.array(directions)

        normal_path = os.path.join(self.root, "normal.png")
        normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
        mask_path = os.path.join(self.root, "mask.png")
        mask = imread(mask_path).astype(np.float32)
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)/ 255.0
        normal = normal * mask.repeat(3, 2)

        down = 4
        if mask.shape[0] % down != 0 or mask.shape[1] % down != 0 :
            pad_h = down - mask.shape[0] % down
            pad_w = down - mask.shape[1] % down
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values = ((0, 0), (0, 0), (0, 0)))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',
                          constant_values = ((0, 0), (0, 0), (0, 0)))
            normal = np.pad(normal, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',
                            constant_values = ((0, 0), (0, 0), (0, 0)))
        img = img * mask.repeat(img.shape[2], 2)
        item = {'N' : normal, 'img' : img, 'mask' : mask}

        for k in item.keys() :
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light :
            item['light'] = torch.from_numpy(directions[select_idx]).view(-1, 1, 1).float()
        item['obj'] = mtrl
        # 增加材质
        # print(self.mtrl_type)
        item['mtrl'] = torch.eye(100)[index]
        return item

    def getMtrl_type(self, mtrl_type) :
        print("get mtrl id", mtrl_type)
        self.mtrl_type = mtrl_type

    def __len__(self) :
        return len(self.mtrl)
