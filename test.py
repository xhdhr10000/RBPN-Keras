import sys
import numpy as np
import keras
from keras.models import load_model
from PIL import Image
from data import read_image
from train import loss, psnr

def cal_psnr(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    return 10. * np.log10(255.0*255.0 / np.mean(np.square(y_pred - y_true)))

x, xb = read_image(sys.argv[1], True)
y_true = read_image(sys.argv[2])
x = np.stack([x])
xb = np.stack([xb])

model = load_model('models/model.hdf5', custom_objects={'loss':loss, 'psnr':psnr})
y = model.predict([x,xb])
print(cal_psnr(y_true, y[0]))
Image.fromarray(y[0,:,:,0].astype(np.uint8), mode='L').show()
Image.fromarray(y_true[:,:,0].astype(np.uint8), mode='L').show()