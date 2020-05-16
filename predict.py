
import os
import sys
import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras as keras
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))

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

def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    

def main():
    batchSize=32
    tmg=ImageDataGenerator(rescale=1./255)
    
    tedg=tmg.flow_from_directory(batch_size=batchSize,
                            directory=os.path.join("./", 'apple2orange',"test"),
                            shuffle=True,
                            target_size=(64, 64),
                            class_mode='categorical')
    tedly=ffzk(os.path.join("./", 'apple2orange','predIn'))
    testX=img2np(tedly[0:16])
    print(testX.shape)
    
    model = keras.models.load_model('modelRegularizers.h5')
    model.summary()
    predY=model.predict(testX)
    print(predY)
    

if __name__ == "__main__":
    tf_ini()
    main()
    