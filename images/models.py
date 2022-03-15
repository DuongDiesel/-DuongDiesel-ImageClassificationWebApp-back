from django.db import models
import pandas as pd
import numpy as np
#import keras
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from django.conf import settings
#from keras.preprocessing import image
#from tensorflow.keras.models import load_model
import os
from tensorflow.python import ops
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input
#from tensorflow.keras.applications.densenet.DenseNet121 import DenseNet121, decode_predictions, preprocess_input

#import 2
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import Activation, Dropout, Flatten, Dense

#import 3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

 
# Create your models here.
 
 
class Image(models.Model):
    picture = models.ImageField()
    classified = models.CharField(max_length=10, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)
 
    def __str__(self):
        return f"Image classified as {self.uploaded.strftime('%Y-%m-%d %H:%M')}"
 
    def save(self, *args, **kwargs):

        try:
            img = load_img(self.picture, target_size=(224,224))
            img_arry = img_to_array(img)
            to_pred = np.expand_dims(img_arry, axis=0) #(1, 299, 299, 3)
            prep = preprocess_input(to_pred)
            model = ResNet50(weights='imagenet')
            prediction = model.predict(prep)
            decoded = decode_predictions(prediction)[0][0][1]
            self.classified = str(decoded)
            print('success')
        except:
            print('failed to classify')
            self.classified = 'failed to classify'
        super().save(*args, **kwargs)