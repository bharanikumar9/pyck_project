from django.db import models
from django.views.generic.dates import timezone_today

# Create your models here.

class File(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField(upload_to='files/')

    def __str__(self) -> str:
        return self.title