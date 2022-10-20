from django.db import models

import io
import os
import json
from torchvision import transforms
import numpy as np
from PIL import Image
import torch 
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

try:
    from torch.hub import load_state_dict_from_url
except:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torch import Tensor
from torch.jit.annotations import List



# load as global variable here, to avoid expensive reloads with each request
model = models.densenet121(pretrained=True)
model.eval()

 
 
class Image(models.Model):
    picture = models.ImageField()
    classified = models.CharField(max_length=10, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)

    def transform_image(image_bytes):
        """
        Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
        Return the corresponding tensor.
        """
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)

    def get_prediction(image_bytes):
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return imagenet_class_index[predicted_idx]

    def __str__(self):
        return f"Image classified as {self.uploaded.strftime('%Y-%m-%d %H:%M')}"
 
    def save(self, *args, **kwargs):
        try:
            #transform
            image = Image.open(io.BytesIO(image_bytes))

            model = ResNet50(weights='imagenet')
            prediction = model.predict(prep)

            decoded = decode_predictions(prediction)[0][0][1]

            self.classified = str(decoded)
            print('success')
            # Find way to clear cache after predic image
        except:
            print('failed to classify')
            self.classified = 'failed to classify'
        
        super().save(*args, **kwargs)


