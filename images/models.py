from django.db import models
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from django.conf import settings
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from tensorflow.python import ops


#import 2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

 
# Create your models here.
 
 
class Image(models.Model):
    picture = models.ImageField()
    classified = models.CharField(max_length=10, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)
 
    def __str__(self):
        return f"Image classified as {self.uploaded.strftime('%Y-%m-%d %H:%M')}"
 
    def save(self, *args, **kwargs):
        LABELS = ['A', 'B']
        img = image.load_img(self.picture, target_size=(32, 32))
        img_array = image.img_to_array(img)
        to_pred = np.expand_dims(img_array, axis=0)
        print(to_pred)
        try:
            #####################
            SIZE = 32
            ###2 conv and pool layers. with some normalization and drops in between.

            INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)

            model = Sequential()
            model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(64))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])
            
            print("model compiled")

            ##########################
            file_model = os.path.join(settings.BASE_DIR, 'malaria_augmented_model.h5')
            print(f'file_model as {file_model}')
            graph = ops.get_default_graph()
 
            with graph.as_default():
                model = load_model(file_model)
                pred = LABELS[model.predict_classes(to_pred)[0]]
                self.classified = str(pred)
                print("graph loaded")
                print(f'classified as {pred}')
        except:
            print('failed to classify')
            self.classified = 'failed to classify'
        super().save(*args, **kwargs)