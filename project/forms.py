from  django import forms
from django.db.models import fields

from .models import File

class FileForm(forms.ModelForm):
    class Meta:
        model = File
        fields = ['title','file']