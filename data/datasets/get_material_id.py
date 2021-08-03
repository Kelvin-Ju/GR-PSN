import os
import glob
datasetName="PS_Blobby_Dataset"
# datasetName="PS_Sculpture_Dataset"
# input_path=datasetName+"/*/*/*/"
# file_name=glob.glob(input_path)
# file_name=[x.split("/")[-2] for x in file_name]
# file_name=sorted(list(set(file_name)))
# print(file_name)
# print(len(file_name))
# with open(f"./{datasetName}_material_id.txt","w") as f:
#     for i,fname in enumerate(file_name):
#         print(f'{i},{fname}',file=f)
mtrl_path=datasetName+'/mtrl.txt'
mtrl_list=[]
with open(mtrl_path,"r") as f:
    for l in f:
        # print(l)
        mtrl_list.append(l[:-1])
train_path=datasetName+'/train_mtrl.txt'
val_path=datasetName+'/val_mtrl.txt'
val_list=[]
with open(val_path,"r") as f:
    for l in f:
        # print(l)
        val_list.append(l.split("/")[0])
#sort
mtrl_list.sort()
# with open(mtrl_path,"w") as f:
#     for l in mtrl_list:
#         print(l,file=f)
#         print(l)
print(val_list)
with open(val_path,"w") as f:
    for l in mtrl_list:
        if l.split("/")[0] in val_list:
            print(l,file=f)
            print(l)

with open(train_path,"w") as f:
    for l in mtrl_list:
        if l.split("/")[0] not in val_list:
            print(l,file=f)
            print(l)