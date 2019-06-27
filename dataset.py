import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from random import randrange
import os.path

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]

        for i in seq:
            index = int(filepath[char_len-7:char_len-4])-i
            file_name=filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png'
            
            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
    
    return target, input, neigbor

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        if scale == 1:
            input = target
        else:
            input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])+i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.png'
            
            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale)
                if scale > 1:
                    temp = temp.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame %s is not exist' % file_name1)
                temp=input
                neigbor.append(temp)
            
    else:
        target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo)
    iw = iw - (iw%modulo)
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, img_nn, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))#[:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))#[:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_nn] #[:, iy:iy + ip, ix:ix + ip]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch

def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class DatasetFromFolder():
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame
        self.indexes = np.arange(len(self.image_filenames))
        np.random.shuffle(self.indexes)
        self.index = 0

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)

        if self.patch_size != 0:
            input, target, neigbor, _ = get_patch(input,target,neigbor,self.patch_size, self.upscale_factor, self.nFrames)
        
        if self.data_augmentation:
            input, target, neigbor, _ = augment(input, target, neigbor)
            
        flow = [get_flow(input,j) for j in neigbor]
            
        # bicubic = rescale_img(input, self.upscale_factor)
        
        target = np.asarray(target, dtype='float32') / 255.0
        input = np.asarray(input, dtype='float32') / 255.0
        neigbor = [np.asarray(i, dtype='float32') / 255.0 for i in neigbor]
        flow = [np.asarray(i, dtype='float32') for i in flow]
        # bicubic = np.asarray(bicubic)

        return input, neigbor, flow, target

    def __len__(self):
        return len(self.image_filenames)
    
    def batch(self, size):
        inputs = []
        neighbors = []
        flows = []
        y = []
        for i in range(size):
            input, neighbor, flow, target = self.__getitem__(self.indexes[self.index])
            self.index = (self.index+1) % len(self.indexes)
            inputs.append(input)
            neighbors.append(neighbor)
            flows.append(flow)
            y.append(target)
        neighbors = np.array(neighbors).swapaxes(0,1)
        flows = np.array(flows).swapaxes(0,1)
        x = [np.array(inputs)] + [n for n in neighbors] + [f for f in flows]
        y = np.array(y)
        return x, y
        

class DatasetFromFolderTest():
    def __init__(self, image_dir, label_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame):
        ''' Make sure the sequences in file_list for image_dir & label_dir are the same '''
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.label_filenames = None
        if label_dir:
            alist = [line.rstrip() for line in open(join(label_dir,file_list))]
            self.label_filenames = [join(label_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.other_dataset = other_dataset
        self.future_frame = future_frame
        self.index = 0

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)

        if self.label_filenames:
            target = Image.open(self.label_filenames[index]).convert('RGB')

        flow = [get_flow(input,j) for j in neigbor]

        # bicubic = rescale_img(input, self.upscale_factor)
        
        target = np.asarray(target, dtype='float32') / 255.0
        input = np.asarray(input, dtype='float32') / 255.0
        neigbor = [np.asarray(i, dtype='float32') / 255.0 for i in neigbor]
        flow = [np.asarray(i) for i in flow]

        return input, neigbor, flow, target
      
    def batch(self, size=1, withName=False):
        inputs = []
        neighbors = []
        flows = []
        y = []
        filenames = []
        for i in range(size):
            input, neighbor, flow, target = self.__getitem__(self.index)
            if withName:
                filenames.append(self.image_filenames[self.index])
            self.index = (self.index+1) % len(self.image_filenames)
            inputs.append(input)
            neighbors.append(neighbor)
            flows.append(flow)
            y.append(target)
        neighbors = np.array(neighbors).swapaxes(0,1)
        flows = np.array(flows).swapaxes(0,1)
        x = [np.array(inputs)] + [n for n in neighbors] + [f for f in flows]
        y = np.array(y)
        return x, y, filenames

    def __len__(self):
        return len(self.image_filenames)
