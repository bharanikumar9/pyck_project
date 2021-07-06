from .models import File
from django.shortcuts import redirect, render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
# Create your views here.
from django.http import HttpResponse

from .forms import FileForm
from .models import File
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

import io

from PIL import Image 
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import keras
from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
from keras.backend import expand_dims
from matplotlib.patches import Rectangle
import cv2
import subprocess
import glob
def runv(frame,width,height,outvdo):
    input_w, input_h = 416, 416
    temp=frame
    # photo_filename = os.path.join(BASE_DIR,"files/"+file_name)
    image= load_image_pixelsv(temp, (input_w, input_h))
    image_w, image_h =width,height
    yhat = model.predict(image)
    boxes = list()
    for i in range(len(yhat)):
        boxes += decode_netout(yhat[i][0], anchors[i], 0.6, input_h, input_w)
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    do_nms(boxes, 0.6)

    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, 0.6)
    # for i in range(len(v_boxes)):
    #     print(v_labels[i], v_scores[i])
    draw_boxesv(frame, v_boxes, v_labels, v_scores,outvdo)

def handle_vid(v):
    fs = FileSystemStorage()
    name = fs.save('vid.mp4',v)
    cap = cv2.VideoCapture(os.path.join(BASE_DIR,"files/"+name))
    if (cap.isOpened() == False): 
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    i=0
    while(True):
            ret, frame = cap.read()
            if ret == True: 
                # frame=cv2.resize(frame,(412,412))
                runv(frame,width,height,i)
                # outvdo.write(frame)
                i=i+1
                if(i%10==0):
                    print(i)
            else:
                break
            if(i==30):  #sonnand ga ikkada chudu, idid number of frames limit petesa
                break  
    fs.delete(name)
    cap.release()
    cv2.destroyAllWindows()
    os.chdir(os.path.join(BASE_DIR,"files/ff"))
    try:
        os.remove("ans.mp4")
    except:
        print("Something went wrong")
    subprocess.call([
            'ffmpeg', '-framerate', str(fps), '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'ans.mp4'
        ]) 
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    os.chdir(os.path.join(BASE_DIR))

    return fs.url("ff/ans.mp4")


def upload_vid(request):
    if (request.method == 'POST') :
        form = FileForm(request.POST,request.FILES)
        if form.is_valid():
            r = handle_vid(request.FILES['file'])
            return render(request,'video.html',{
                'url' : r
            })
    else:
        form = FileForm()
    return render(request,'upv.html',{
        'form' : form
    })

def upload_file(request):
    if request.method == 'POST':
        form = FileForm(request.POST,request.FILES)
        # for k,v in request.POST.items():
        #     print(k)
        #     print(v)
        # print(request.POST)
        # print(request.POST.get('title'))
        # print(request.FILES)
        if form.is_valid():
            # print("OKKKKK")
            handle_uploaded_file(request.FILES['file'])
            # form.save()
            return redirect('files')
    else:
        form = FileForm()
    return render(request,'upload.html',{
    })

def cam(request):
    if(request.method == 'POST'):
        form = FileForm(request.POST,request.FILES)
        for k,v in request.POST.items():
            print(k)
            print(v)
        print(request.POST)
        print(request.POST.get('title'))
        print(request.FILES)
        if form.is_valid():
            print("OKKKKK")
            handle_uploaded_file(request.FILES['file'])
            # form.save()
            return redirect('files')
        else:
            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            form = FileForm()

    return render(request,'cam.html',{
        })
