import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
    
def readlinesFromFile(path, datasize):
    print("Load from file %s" % path)
    f=open(path)
    data=[]    
    for idx in range(0, datasize):
        line = f.readline()
        data.append(line)      
    
    f.close()  
    return data  
    
def loadFromFile(path, datasize):
    if path is None:
        return None, None
      
    print("Load from file %s" % path)
    f=open(path)
    data=[]
    label=[]
    for idx in range(0, datasize):
        line = f.readline().split()
        data.append(line[0])         
        label.append(line[1])
       
    f.close()  
    return data, label     
    
def load_video_image(file_path, input_height=None, input_width=None, output_height=128, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_flip=True,
              is_gray=False, scale=1.0, is_scale_back=False):
    
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height
    
    img = Image.open(file_path)
    if is_gray is False and img.mode != 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode != 'L':
        img = img.convert('L')
      
    if is_mirror and random.randint(0,1) == 0:
        img = ImageOps.mirror(img)    
        
    if is_flip and random.randint(0,1) == 0:
        img = ImageOps.flip(img)    
      
    if input_height is not None:
        img = img.resize((input_width, input_height),Image.BICUBIC)
      
    if crop_height is not None:
        [w, h] = img.size
        if is_random_crop:
        #print([w,cropSize])
            cx1 = random.randint(0, w-crop_width)
            cx2 = w - crop_width - cx1
            cy1 = random.randint(0, h-crop_height) 
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w-crop_width)/2.))
            cy2 = cy1 = int(round((h-crop_height)/2.))
        img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))
      
    #print(scale)
    img = img.resize((output_width, output_height),Image.BICUBIC)
    img_lr = img.resize((int(output_width/scale),int(output_height/scale)),Image.BICUBIC)
    if is_scale_back:
        return img_lr.resize((output_width, output_height),Image.BICUBIC), img
    else:
        return img_lr, img
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, root_path, input_height=None, input_width=None, output_height=128, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_flip=True,
              is_gray=False, upscale=1.0, is_scale_back=False):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list        
        self.upscale = upscale
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.is_flip = is_flip
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_scale_back = is_scale_back
        self.is_gray = is_gray
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])

    def __getitem__(self, index):
    
        if self.is_mirror:
            is_mirror = (random.randint(0,1) == 0)
        else:
            is_mirror = False
            
        if self.is_flip:
            is_flip = (random.randint(0,1) == 0)
        else:
            is_flip = False
          
        lr, hr = load_video_image(join(self.root_path, self.image_filenames[index]), 
                                  self.input_height, self.input_width, self.output_height, self.output_width,
                                  self.crop_height, self.crop_width, self.is_random_crop, is_mirror, is_flip,
                                  self.is_gray, self.upscale, self.is_scale_back)
        
        
        input = self.input_transform(lr)
        target = self.input_transform(hr)
        
        return input, target

    def __len__(self):
        return len(self.image_filenames)
        
#=======================================================================================================================================
def load_test_image(img_path, is_resize=False, resize_h=144, resize_w=None, is_rcrop=False, crop_h=128, crop_w=None, grayscale=False):

    #open img
    img = Image.open(img_path)
    if grayscale and img.mode != 'L' :
        img = img.convert('L')
    if grayscale is False and img.mode != 'RGB' :
        img = img.convert('RGB')
    
    #augmentation
    if is_resize :
        if resize_w is None :
            resize_w = resize_h
        img = img.resize((resize_w, resize_h), Image.BICUBIC)
    if is_rcrop :
        [w, h] = img.size
        if crop_w is None :
            crop_w = crop_h
        cx1 = random.randint(0, w-crop_w)
        cx2 = cx1 + crop_w
        cy1 = random.randint(0, h-crop_h)
        cy2 = cy1 + crop_h
        
        img = img.crop((cx1, cy1, cx2, cy2))
            
    return img

class RealImgTest(data.Dataset):
    def __init__(self, root_path, is_resize=False, resize_h=144, resize_w=None, is_rcrop=False, crop_h=128, crop_w=None, grayscale=False):
        super(RealImgTest, self).__init__()
        
        self.img_path = root_path
        self.img_lst = listdir(root_path)
        self.is_resize = is_resize
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.is_rcrop = is_rcrop
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.grayscale = grayscale          
        self.transform = transforms.ToTensor()
                               
    def __getitem__(self, index):
                               
        path = '/'.join([self.img_path, self.img_lst[index]])
        img = load_test_image(path, is_resize=self.is_resize, resize_h=self.resize_h, resize_w=self.resize_w, 
                         is_rcrop=self.is_rcrop, crop_h=self.crop_h, crop_w=self.crop_w, grayscale=self.grayscale)
        img = self.transform(img)
        
        return img
        
    def __len__(self):
        
        return len(self.img_lst)
