# GR-PSN
### Dependencies
GR-PSN is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install PyTorch first following the official instruction. 
- Python 3.7 
- PyTorch 1.8
- numpy
- scipy
- CUDA-11.1
- RTX 3090 (24G)

## For training our GR-PSN, you need download these two datasets:
Blobby shape dataset (4.7 GB), and Sculpture shape dataset (19 GB), via: 

```shell
sh scripts/download_synthetic_datasets.sh
```
## For testing our GR-PSN, you can download these datsets:

DiLiGenT main dataset (default) (850MB), via:
```shell
sh scripts/prepare_diligent_dataset.sh  
```
or   https://drive.google.com/file/d/1EgC3x8daOWL4uQmc6c4nXVe4mdAMJVfg/view


Light Stage Data Gallery, via:

https://vgl.ict.usc.edu/Data/LightStage/

## Results

We have shown some results of our GR-PSN in the document "results", including the estimations under 96 input images on the DiLiGenT benchmark dataset, and the rendered examples  of object "Dragon" (.mp4).


## Testing

#### Test on the DiLiGenT dataset
```shell

# Test GR_PSN on DiLiGenT main dataset using all of the 96 image-light pairs
python eval/run_model_Deligent.py --retrain data/models/GNet_checkp_1.pth.tar --in_img_num 96
 ``
```

#### You can change the number 96 to use an arbitrary number of input images. (1~96)




## Acknowledgement:

Our code is partially based on: https://github.com/guanyingc/PS-FCN.
