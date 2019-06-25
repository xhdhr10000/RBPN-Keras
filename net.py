import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Conv2DTranspose, Flatten, Concatenate, Input, PReLU, Add, Subtract, ReLU, LeakyReLU

"""
DCSCN Model
"""
def SRModel():
    x = Input(shape=(None, None, 1))
    bx = Input(shape=(None, None, 1))
    conv1 = Conv2D(96, 3, padding='same')(x)
    relu1 = PReLU(shared_axes=[1,2])(conv1)
    conv2 = Conv2D(80, 3, padding='same')(relu1)
    relu2 = PReLU(shared_axes=[1,2])(conv2)
    conv3 = Conv2D(64, 3, padding='same')(relu2)
    relu3 = PReLU(shared_axes=[1,2])(conv3)
    conv4 = Conv2D(48, 3, padding='same')(relu3)
    relu4 = PReLU(shared_axes=[1,2])(conv4)
    conv5 = Conv2D(32, 3, padding='same')(relu4)
    relu5 = PReLU(shared_axes=[1,2])(conv5)

    concat = Concatenate()([relu1, relu2, relu3, relu4, relu5])

    C = Conv2D(32, 1)(concat)
    tcnn = Conv2DTranspose(32, 3, strides=(4,4))(C)
    rcnn = Conv2D(1, 3, padding='same')(tcnn)
    y = Add()([rcnn, bx])
    return Model(inputs=[x, bx], outputs=y)


"""
RBPN Model
"""
def UpBlock():
    return Sequential([
        Conv2DTranspose(64, 8, strides=(4,4), padding='same'),
        PReLU(shared_axes=[1,2]),
        Conv2D(64, 8, strides=(4,4), padding='same'),
        PReLU(shared_axes=[1,2]),
        Conv2DTranspose(64, 8, strides=(4,4), padding='same'),
        PReLU(shared_axes=[1,2]),
    ])

def DownBlock():
    return Sequential([
        Conv2D(64, 8, strides=(4,4), padding='same'),
        PReLU(shared_axes=[1,2]),
        Conv2DTranspose(64, 8, strides=(4,4), padding='same'),
        PReLU(shared_axes=[1,2]),
        Conv2D(64, 8, strides=(4,4), padding='same'),
        PReLU(shared_axes=[1,2]),
    ])

def DBPN(x):
    feat1 = Conv2D(64, 1, padding='same')(x)
    act1 = PReLU(shared_axes=[1,2])(feat1)

    up1 = UpBlock()(act1)
    down1 = DownBlock()(up1)
    up2 = UpBlock()(down1)
    down2 = DownBlock()(up2)
    up3 = UpBlock()(down2)

    cat = Concatenate()([up3, up2, up1])
    output = Conv2D(64, 1, padding='same')(cat)
    return output

def ResnetBlock(x, filters):
    conv1 = Conv2D(filters, 3, padding='same')(x)
    act1 = PReLU(shared_axes=[1,2])(conv1)
    conv2 = Conv2D(filters, 3, padding='same')(act1)
    add = Add()([conv2, x])
    act2 = PReLU(shared_axes=[1,2])(add)
    return act2

def res_feat1(x):
    out = x
    for i in range(5):
        out = ResnetBlock(out, 256)
    out = Conv2DTranspose(64, 8, strides=(4,4), padding='same')(out)
    out = PReLU(shared_axes=[1,2])(out)
    return out

def res_feat2(x):
    out = x
    for i in range(5):
        out = ResnetBlock(out, 64)
    out = Conv2D(64, 3, padding='same')(out)
    out = PReLU(shared_axes=[1,2])(out)
    return out

def res_feat3(x):
    out = x
    for i in range(5):
        out = ResnetBlock(out, 64)
    out = Conv2D(256, 8, strides=(4,4), padding='same')(out)
    out = PReLU(shared_axes=[1,2])(out)
    return out

def RBPN():
    x = Input(shape=(None, None, 3))
    neighbor = [Input(shape=(None, None, 3)) for i in range(6)]
    flow = [Input(shape=(None, None, 2)) for i in range(6)]

    feat0 = Conv2D(256, 3, padding='same')(x)
    feat_input = PReLU(shared_axes=[1,2])(feat0)

    Ht = []
    for i in range(len(neighbor)):
        h0 = DBPN(feat_input)
        cat_input = Concatenate()([x, neighbor[i], flow[i]])
        feat1 = Conv2D(256, 3, padding='same')(cat_input)
        act1 = PReLU(shared_axes=[1,2])(feat1)
        h1 = res_feat1(act1)
        e = Subtract()([h0, h1])
        e = res_feat2(e)
        h = Add()([h0, e])
        Ht.append(h)
        feat_input = res_feat3(h)
    
    cat_output = Concatenate()(Ht)
    y = Conv2D(3, 3, padding='same')(cat_output)
    xs = [x] + neighbor + flow
    return Model(inputs=xs, outputs=y)