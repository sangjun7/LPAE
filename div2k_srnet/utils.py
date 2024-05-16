import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

#===============================================================================================================================
def gauss_noise(x, mean=0, var=0.1, use_cuda=False, device='cpu'):
    sz = x.shape
    noise = torch.normal(mean, var, size=(sz[0], sz[1], sz[2], sz[3]))
    if use_cuda:
        noise = noise.to(device)
    x = x + noise
    return x

def imgsave(torch_img, path, file_name):
    """This function takes an image with the float tensor of range [0,1] and the shape of BCHW as input, then save the image with png format by cv2 library"""
    img = torch_img.squeeze().float().cpu().clamp_(0, 1).numpy()
    img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
    img = (img * 255.0).round().astype(np.uint8)
    cv2.imwrite('/'.join([path, '{}.png'.format(file_name)]), img)
    
#===============================================================================================================================
def togray(img, ver='YCrCb_BT601'):
    """
    The type of input image must be np.float32 with [0,1] range, since model returns that precision.
    
    This function changes the dtype to np.float32 for 8 bits per sample and the range to [0,1].
    And then get the Y channel from RGB channel ( The original image has HWC order )
    
    Return : float32 ndarray with [0,255] range
    """
    img = img.astype(np.float32)           
    if ver == 'YCrCb_BT601':
        coeff = np.array([65.481, 128.553, 24.966]).reshape(1, 1, 3)
        img = img * coeff
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2] + 16
        img /= 255.0
    elif ver == 'YCrCb_BT601_single_bitshift': #CAR
        coeff = np.array([65.738, 129.057, 25.064]).reshape(1, 1, 3) 
        img = img * coeff
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2] + 16
        img /= 255.0
    elif ver == 'YPrPb_BT601':  
        coeff = np.array([0.299, 0.587, 0.114]).reshape(1, 1, 3)
        img = img * coeff
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    elif ver == 'YCrCb_BT709':
        coeff = np.array([0.2126, 0.7152, 0.0722]).reshape(1, 1, 3)
        img = img * coeff
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    elif ver == 'YCrCb_BT2020':
        coeff = np.array([0.2627, 0.678, 0.0593]).reshape(1, 1, 3)
        img = img * coeff
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    elif ver == 'YCrCb_SMPTE_240M':
        coeff = np.array([0.212, 0.701, 0.087]).reshape(1, 1, 3)
        img = img * coeff
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    img = img * 255.0
    img = img.astype(np.float32)
    
    return img

def togray_pt(img, ver='YCrCb_BT601'):
    """
    The type of input image must be torch.float32 with [0,1] range, since model returns that precision.
    And The image has the order of BCHW (generally, B=1).
    
    This function changes the dtype to torch.float32 for 8 bits per sample.
    And then get the Y channel from RGB channel ( The original image has HWC order )
    
    Return : float32 torch tensor with [0,255] range and order of BCHW
    """
    img = img.to(torch.float32)     
    if ver == 'YCrCb_BT601':
        coeff = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1).to(img.device)
        img = img * coeff
        img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :] + 16
        img /= 255.0
    elif ver == 'YCrCb_BT601_single_bitshift': #CAR
        coeff = torch.tensor([65.738, 129.057, 25.064]).reshape(1, 3, 1, 1).to(img.device)
        img = img * coeff
        img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :] + 16
        img /= 255.0
    elif ver == 'YPrPb_BT601':  
        coeff = torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).to(img.device)
        img = img * coeff
        img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]
    elif ver == 'YCrCb_BT709':
        coeff = torch.tensor([0.2126, 0.7152, 0.0722]).reshape(1, 3, 1, 1).to(img.device)
        img = img * coeff
        img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]
    elif ver == 'YCrCb_BT2020':
        coeff = torch.tensor([0.2627, 0.678, 0.0593]).reshape(1, 3, 1, 1).to(img.device)
        img = img * coeff
        img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]
    elif ver == 'YCrCb_SMPTE_240M':
        coeff = torch.tensor([0.212, 0.701, 0.087]).reshape(1, 3, 1, 1).to(img.device)
        img = img * coeff
        img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]
    img = img * 255.0
    img = img.unsqueeze(1)
    
    return img

#===============================================================================================================================
def cal_psnr(img, pred, crop_border=0, gray_scale=True, clamp=True, ver='YCrCb_BT601'):
    """
    This function takes img(target) and pred(prediction) with range [-1,1] as input.
    Inputs have the shape of BCHW (generally, B=1) and the type of torch.float32.
    
    After taking images, we change the range of images to [0,1] and device to cpu.
    Then change numpy from torch.tensor, change order to BHWC and squeeze to HWC.
    
    If want, clamping to [0,1], cropping border and converting to Y channel are implemented.
    
    To Do : For just in case, modify code to consider batch size > 1
    """
    
    img = (img + 1)/2                                # change range to 0~1 from -1~1
    img = img.cpu().numpy().transpose(0, 2, 3, 1)    # change device and to numpy and to order of BHWC
    img = img[0, ...]                               # squeeze to HWC by choosing 1 batch of index 0
        
    pred = (pred + 1)/2
    if clamp:
        pred = torch.clamp(pred,0,1)   
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
    pred = pred[0, ...]
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        pred = pred[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if gray_scale:
        img = togray(img, ver=ver)
        pred = togray(pred, ver=ver)
    else:
        img = img * 255.0
        img = img.astype(np.float32)
        pred = pred * 255.0
        pred = pred.astype(np.float32)
        
    mse = np.mean((img - pred) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10. * np.log10(255. * 255. / mse)

    return psnr

def cal_ssim(img, pred, crop_border=0, data_range=255., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, grayscale=True, clamp=True, ver='YCrCb_BT601'):
    """
    This function takes img(target) and pred(prediction) with range [-1,1] as input.
    Inputs have the shape of BCHW (generally, B=1) and the type of torch.float32.
    
    After taking images, we convert to Y channel with the range of [0,255]m then change to torch.float64.
    
    If want, clamping to [0,1], cropping border are implemented.
    """
    
    img = (img + 1)/2                                # change range to 0~1 from -1~1
    pred = (pred + 1)/2
    if clamp:
        pred = torch.clamp(pred,0,1)   
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        pred = pred[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    if grayscale:
        img = togray_pt(img, ver=ver)
        pred = togray_pt(pred, ver=ver)              # get Y channel float32 tensor with range [0,255] and order BCHW (basically, B=1)
    else:
        img = img * 255.0
        pred = pred * 255.0
        
    img = img.to(torch.float64)
    pred = pred.to(torch.float64)
        
    b, ch, h, w = img.size()
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    gauss = cv2.getGaussianKernel(filter_size, filter_sigma)
    window = np.outer(gauss, gauss.transpose())
    window = torch.from_numpy(window).reshape(1, 1, filter_size, filter_size)
    window = window.expand(ch, 1, filter_size, filter_size).to(img.dtype).to(img.device)
    
    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=ch)
    mu2 = F.conv2d(pred, window, stride=1, padding=0, groups=ch)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2
    sig1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=ch) - mu1_sq
    sig2_sq = F.conv2d(pred * pred, window, stride=1, padding=0, groups=ch) - mu2_sq
    sig12 = F.conv2d(img * pred, window, stride=1, padding=0, groups=ch) - mu12
    
    l_comp = (2*mu12 + c1)/(mu1_sq + mu2_sq + c1)
    cs_comp = (2*sig12 + c2)/(sig1_sq + sig2_sq + c2)
    ssim = l_comp * cs_comp
    ssim_mean = ssim.mean([1,2,3])                # Evaluation mean without batch. So it has order of B (basically, B=1)
    
    return ssim_mean.item()
