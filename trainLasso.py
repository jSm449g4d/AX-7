
import os
import sys
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

import optuna
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))


def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        
def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

def img2np(dir=[],img_len=64):
    img=[]
    for x in dir:
        try:img.append(cv2.imread(x))
        except:continue
        if img_len!=0:img[-1]=cv2.resize(img[-1],(img_len,img_len))
        elif img[-1].shape!=img[0].shape:img.pop(-1);continue#Leave only the same shape
        img[-1] = img[-1].astype(np.float32)/ 256
    return np.stack(img, axis=0)
    
def GEN(input_shape=(64,64,3,)):
    mod=mod_inp = Input(shape=input_shape)
    mod=Flatten()(mod)
    mod=Dense(2,activation="softmax",kernel_regularizer=regularizers.l1(0.001),)(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)

def main():
    batchSize=16
    tmg=ImageDataGenerator(rescale=1./255)
    tdg=tmg.flow_from_directory(batch_size=batchSize,
                            directory=os.path.join("./", 'apple2orange',"train"),
                            shuffle=True,
                            target_size=(64, 64),
                            class_mode='categorical')
    tedg=tmg.flow_from_directory(batch_size=batchSize,
                            directory=os.path.join("./", 'apple2orange',"test"),
                            shuffle=True,
                            target_size=(64, 64),
                            class_mode='categorical')
    
    model=GEN()
    model.compile(optimizer=optimizers.SGD(lr=0.01),
                  loss=keras.losses.mean_squared_error)
    model.summary()
    cbks=[keras.callbacks.TensorBoard(log_dir='logsRegularizers', histogram_freq=1)]
    
    model.fit(tdg,steps_per_epoch=len(tdg),epochs=30,validation_data=tedg,validation_steps=len(tedg),callbacks=cbks)
    model.save('modelRegularizers.h5')

if __name__ == "__main__":
    tf_ini()
    main()
    