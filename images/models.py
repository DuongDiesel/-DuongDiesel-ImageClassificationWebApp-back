from django.db import models

import io
import os
import json

from torchvision import transforms


 

 
 
class Image(models.Model):
    picture = models.ImageField()
    classified = models.CharField(max_length=10, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)
 
    def __str__(self):
        return f"Image classified as {self.uploaded.strftime('%Y-%m-%d %H:%M')}"
 
    def save(self, *args, **kwargs):

        
        super().save(*args, **kwargs)


