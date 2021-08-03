import torch.nn
from scipy.ndimage import imread
import os
import numpy as np
import torch

id2mtrl={46:"neoprene-rubber",
         91:"white-diffuse-bball",
         54:"pink-felt",
         1:"alumina-oxide"}

L1=torch.nn.L1Loss()
dic={}

# item="cowPNG"
item="pot2PNG"
f_name=[]
with open('run_model_r2/filenames.txt',"r") as f:
    for line in f:
        f_name.append(line.rstrip("\n"))
print(f_name)
for i in [46, 91, 54, 1]:

    for j in [46, 91, 54, 1]:
        img_path=f"mtrl_{i}/mtrl_{j}/"
        imgs = []
        for name in f_name:
            img_name=os.path.join("run_model_r2",img_path,item,name)
            # print(img_name)
            img=imread(img_name).astype(np.float32) / 255.0
            imgs.append(img)
        dic[f"{i}_{j}"]= np.concatenate(imgs, 2)
# print(dic)
for i in [46, 91, 54, 1]:
    img_path = f"mtrl_{i}"
    imgs = []
    for name in f_name :
        img_name = os.path.join("run_model_r2", img_path, item, name)
        # print(img_name)
        img = imread(img_name).astype(np.float32) / 255.0
        imgs.append(img)
    dic[f"{i}"] = np.concatenate(imgs, 2)
print(dic.keys())

for k in dic.keys():
    dic[k]=torch.from_numpy(dic[k])
REL={}

for i in [46, 91, 54, 1]:
    for j in [46, 91, 54, 1]:
        for k in [46, 91, 54, 1] :
            loss=L1(dic[f"{j}_{i}"],dic[f"{k}_{i}"])
            print(f"L1 between {id2mtrl[j]}->{id2mtrl[i]} and {id2mtrl[k]}->{id2mtrl[i]} :",loss.item())
