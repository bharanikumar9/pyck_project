from django.urls import path

from . import views

urlpatterns = [
    path('', views.Home.as_view(), name='home'),
    path('index', views.index, name='index'),
    path('files', views.files, name='files'),
    path('upload', views.upload_file, name='upload'),
    path('upv',views.upload_vid,name='upload_vid'),
    path('cam',views.cam,name='cam'),
]