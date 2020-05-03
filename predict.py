
import os
import sys
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras as keras
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))


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
    
    model = keras.models.load_model('myModel.h5')
    model.summary()
    testX,testY=next(tedg)
    predY=model.predict(testX)
    print(testY.argmax(axis = 1),"\n==\n",predY.argmax(axis = 1))
    

if __name__ == "__main__":
    tf_ini()
    main()
    