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
def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, target_size=shape)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)
    # print(image.size())
    return image, width, height

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1
 
	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
 
		return self.label
 
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
 
		return self.score
def _sigmoid(x):
	return 1. / (1. + np.exp(-x))
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w
			y = (row + y) / grid_h 
			w = anchors[2 * b + 0] * np.exp(w) / net_w
			h = anchors[2 * b + 1] * np.exp(h) / net_h 
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
def interval_overlap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2
    if x3 < x1:
        return 0 if x4 < x1 else (min(x2,x4) - x1)
    else:
        return 0 if x2 < x3 else (min(x2,x4) - x3)
def bbox_iou(box1, box2):
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect_area = intersect_w * intersect_h
    
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union_area = w1*h1 + w2*h2 - intersect_area
    return float(intersect_area) / union_area
def do_nms(boxes, nms_thresh): 
    if len(boxes) > 0: 
        pass 
    else: 
        return 

    for c in range(1): 
        sorted_indices = np.argsort([-box.classes[c] for box in boxes]) 
 
        for i in range(len(sorted_indices)): 
            index_i = sorted_indices[i] 
            
            if boxes[index_i].classes[c] == 0: continue 
 
            for j in range(i+1, len(sorted_indices)): 
                index_j = sorted_indices[j] 
 
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh: 
                    boxes[index_j].classes[c] = 0
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    for box in boxes:
        for i in range(len(labels)):
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    return v_boxes, v_labels, v_scores


# def draw_boxesd(data, v_boxes, v_labels, v_scores):

#     plt.imshow(data)
#     ax = plt.gca()
#     for i in range(len(v_boxes)):
#         box = v_boxes[i]
#         y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
#         width, height = x2 - x1, y2 - y1
#         rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#         ax.add_patch(rect)

#     plt.imsave("")
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    p=[]
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='green')
        ax.add_patch(rect)
        # label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        # plt.text(x1, y1, label, color='white')
        if(v_labels[i]=="person"):
            p.append([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1])
    for id in range(len(p)):
        for id2 in range(id+1,len(p)):
            # plt.plot([p[id][0],p[id2][0]],[p[id][1],p[id2][1]])
            # print((p[id][0]-p[id2][0])**2+(p[id][1]-p[id2][1])**2,1.6*(min(p[id][2],p[id2][2])**2)/max(p[id][3]/p[id2][3],p[id2][3]/p[id][3]),sep=" ")
            if((p[id][0]-p[id2][0])**2+(p[id][1]-p[id2][1])**2<=2*(min(p[id][2],p[id2][2])**2)/(max(p[id][3]/p[id2][3],p[id2][3]/p[id][3])**2)):
                plt.plot([p[id][0],p[id2][0]],[p[id][1],p[id2][1]],"r")
                # print("222222222222")   
    # plt.show()
    plt.savefig(os.path.join(BASE_DIR,"files/ans.jpg"))
    plt.close()

labels = ["person"]

model = load_model(os.path.join(BASE_DIR,"files/model.h5"))
def run(file_name):
    input_w, input_h = 416, 416
    photo_filename = os.path.join(BASE_DIR,"files/"+file_name)
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
    yhat = model.predict(image)
    boxes = list()
    for i in range(len(yhat)):
        boxes += decode_netout(yhat[i][0], anchors[i], 0.6, input_h, input_w)
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    do_nms(boxes, 0.6)

    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, 0.6)
    # for i in range(len(v_boxes)):
    #     print(v_labels[i], v_scores[i])
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)

def handle_uploaded_file(f):
    im1=Image.open(f)
    # rgb_im = im1.convert('RGB')
    im1.save("files/input.jpg")
    run("input.jpg")


class Home(TemplateView):
    template_name = 'home.html'

def index(request):
    return render(request,'home.html')

def files(request):
    # files = File.objects.get(title="answer")
    return render(request,'files.html',{
        'title' : "modified"
    })



def load_image_pixelsv(frame,shape):
    cvt_image =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    image = im_pil.resize(shape)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)
    return image

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def draw_boxesv(frame, v_boxes, v_labels, v_scores,outvdo):
    frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)    
    plt.imshow(frame)
    ax = plt.gca()
    p=[]
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='green')
        ax.add_patch(rect)
        # label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        # plt.text(x1, y1, label, color='white')
    #     if(v_labels[i]=="person"):
    #         p.append([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1])
    for id in range(len(p)):
        for id2 in range(id+1,len(p)):
            # plt.plot([p[id][0],p[id2][0]],[p[id][1],p[id2][1]])
            # print((p[id][0]-p[id2][0])**2+(p[id][1]-p[id2][1])**2,1.6*(min(p[id][2],p[id2][2])**2)/max(p[id][3]/p[id2][3],p[id2][3]/p[id][3]),sep=" ")
            if((p[id][0]-p[id2][0])**2+(p[id][1]-p[id2][1])**2<=1.6*(min(p[id][2],p[id2][2])**2)/max(p[id][3]/p[id2][3],p[id2][3]/p[id][3])):
                plt.plot([p[id][0],p[id2][0]],[p[id][1],p[id2][1]],"g")
    #             print("222222222222")   
    plt.savefig(os.path.join(BASE_DIR,"files/ff")+"/file%02d.png" % outvdo)
    plt.close()


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
