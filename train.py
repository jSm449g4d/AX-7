
import os
import sys
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))


def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    
def GEN(input_shape=(64,64,3,)):
    mod=mod_inp = Input(shape=input_shape)
    mod=Conv2D(8,3,2,padding="same")(mod)
    mod=Conv2D(8,3,2,padding="same")(mod)
    mod=Activation("relu")(mod)
    mod=Dropout(0.1)(mod)
    mod=Conv2D(8,3,2,padding="same")(mod)
    mod=Conv2D(8,3,2,padding="same")(mod)
    mod=Activation("relu")(mod)
    mod=Dropout(0.1)(mod)
    mod=Flatten()(mod)
    mod=Dense(2,activation="softmax")(mod)
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
    model.compile(optimizer='adam',
                loss=keras.losses.binary_crossentropy)
    model.summary()
    cbks=[keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)]
    
    model.fit(tdg,steps_per_epoch=len(tdg),epochs=20,validation_data=tedg,validation_steps=len(tedg),callbacks=cbks)
    model.save('myModel.h5')

if __name__ == "__main__":
    tf_ini()
    main()
    