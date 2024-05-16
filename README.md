# LPAE
This repository is for Laplacin Pyramid-like Auto-Encoder (LPAE) introduced in the following paper.

Sangjun Han, Taeil Hur, Youngmi Hur, "Laplacian Pyramid-like Autoencoder", In: Arai, K. (eds) Intelligent Computing. SAI 2022. Lecture Notes in Networks and Systems, vol 507. Springer, Cham. https://doi.org/10.1007/978-3-031-10464-0_5  
(arXiv version : https://doi.org/10.48550/arXiv.2208.12484)

The pretrained parameters are provided by Google Drive. 
https://drive.google.com/drive/folders/1uVY4yn3K4n-2EWi1A9f0mUbQdaHgntXQ?usp=sharing

## Setting
For our codes, pytorch and torchvision is required.  
We only check the 2.0.1 version of pytorch and  in recent.
If you want to use other version of pytorch, you can find in https://pytorch.org/get-started/previous-versions/

## Test
We provide just one test code `div2k_srnet/test_real_img.py` for the super-resolution experiment of LPSR trained on DIV2K.  
This code takes images in a target directory as input, enlarges them by 2, 4, or 8 times, and then saves them.  
Generalization of this code or code for other models will be posted later.  

To run the code, use the following command:
```
python div2k_srnet/test_real_img.py --mag [Scale factor between [2, 4, 8]] --testdata [Directory path of test images] \
--trained_ae [Path for a trained Auto-encoder model] --trained_srnet [Path for a trained SR model] \
--save --save_path [Path to save result images] --workers [Number of cpu workers]
```

If you want to use GPU, then you should add arguments for gpu `--cuda`, `--ngpu [Number of GPU to use]`, and `--initial_gpu [Initial number of GPU]`. 

If you want to resize or to crop randomly input images,then you should add arguments  
`--imresize`, `--imresize_h [Height of resized image]`, and `--imresize_w [Width of resized image]` for resizing,  
`--imrandcrop`, `--imrandcrop_h [Height of randomly cropped image]`, and `--imrandcrop_w [Width of randomly cropped image]` for randomly cropping.

For example, you can enlarge Set5 images by 2 times with 1 GPU and save them:
```
python div2k_srnet/test_real_img.py --mag 2 --testdata ./datasets/Set5 \
--trained_ae ./model_save/wae_change_structure_div2k_rec_l1.pth \
--trained_srnet ./model_save/wae_srnet_structure_change_rec_l1_div2k_2mag.pth \
--save --save_path ./results --workers 4 --cuda --ngpu 1 --initial_gpu 0
```


