import numpy as np
from PIL import Image
import os

scale=4
input_size=48
dataset_train_x='dataset/set5/input'
dataset_train_y='dataset/set5/label'
dataset_val_x='dataset/set5/input'
dataset_val_y='dataset/set5/label'

def get_files_in_directory(path):
    if not path.endswith('/'): path += '/'
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.startswith('.')]

def image2nparray(image):
    ret = np.asarray(image, 'float32')
    ret = ret[:,:,:1]
    return ret

def read_image(path, resize=False):
    im = Image.open(path).convert('YCbCr')
    width = im.width
    height = im.height

    if resize:
        imb = im.resize((width*scale, height*scale), Image.BICUBIC)
        imb = image2nparray(imb)
        im = image2nparray(im)
        return im, imb
    else:
        im = image2nparray(im)
        return im

def divide_image(imi, imb, iml):
    stride = input_size // 2
    w, h = imi.shape[0], imi.shape[1]
    sx = np.random.randint(w % stride) if w % stride != 0 else 0
    ex = w - (w%stride - sx) - input_size + 1
    sy = np.random.randint(h % stride) if w % stride != 0 else 0
    ey = h - (h%stride - sy) - input_size + 1
    imis = []
    imbs = []
    imls = []
    for x in range(sx, ex, stride):
        for y in range(sy, ey, stride):
            imis.append(imi[x:x+input_size, y:y+input_size])
            imbs.append(imb[x*scale:(x+input_size)*scale, y*scale:(y+input_size)*scale])
            imls.append(iml[x*scale:(x+input_size)*scale, y*scale:(y+input_size)*scale])
    return imis, imbs, imls

def load_images():
    print('Loading images', end='', flush=True)
    x_train = []
    x_train_b = []
    y_train = []
    x_val = []
    x_val_b = []
    y_val = []

    filenames = get_files_in_directory(dataset_train_x)
    for f in filenames:
        imi, imb = read_image(os.path.join(dataset_train_x, f), True)
        iml = read_image(os.path.join(dataset_train_y, f))
        imis, imbs, imls = divide_image(imi, imb, iml)
        x_train += imis
        x_train_b += imbs
        y_train += imls
        print('.', end='', flush=True)
    
    filenames = get_files_in_directory(dataset_val_x)
    for f in filenames[:1]:
        imi, imb = read_image(os.path.join(dataset_val_x, f), True)
        x_val.append(imi)
        x_val_b.append(imb)
        y_val.append(read_image(os.path.join(dataset_val_y, f)))
        print('.', end='', flush=True)

    print('finished. Total number: ', len(x_train)+len(y_train) + len(x_val)+len(y_val))
    x_train = np.stack(x_train)
    x_train_b = np.stack(x_train_b)
    y_train = np.stack(y_train)
    x_val = np.stack(x_val)
    x_val_b = np.stack(x_val_b)
    y_val = np.stack(y_val)
    print('Data shape:', x_train.shape, x_train_b.shape, y_train.shape, x_val.shape, x_val_b.shape, y_val.shape)
    return x_train, x_train_b, y_train, x_val, x_val_b, y_val