import os
import torch
import yaml
from ultralytics import YOLO  
from QtFusion.path import abs_path  # get path file
device = "0" if torch.cuda.is_available() else "cpu"
workers = 1  # number of work processes
batch = 8  # number of pictures
data_name = "Violence"
data_path = abs_path(f'datasets/{data_name}/{data_name}.yaml', path_type='current')
unix_style_path = data_path.replace(os.sep, '/')
directory_path = os.path.dirname(unix_style_path)
with open(data_path, 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

if 'path' in data:
    data['path'] = directory_path
    with open(data_path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)
model = YOLO(abs_path('./weights/yolov5nu.pt', path_type='current'), task='detect')  
# model = YOLO('./weights/yolov5.yaml', task='detect').load('./weights/yolov5nu.pt')  
# Training.
results = model.train(  # train model
    data=data_path,  
    device=device,  
    workers=workers,  
    imgsz=640,  # picture size 640x640
    epochs=120,  
    batch=batch,  # niumbre of  pictures = 8
    name='train_v5_' + data_name  # name of training task
)
model = YOLO(abs_path('./weights/yolov8n.pt'), task='detect')  #  pre-training YOLOv8
results2 = model.train(  # train model
    data=data_path,  
    device=device,  
    workers=workers,  
    imgsz=640,  # picture size 640x640
    epochs=120,  
    batch=batch,  # niumbre of  pictures = 8
    name='train_v8_' + data_name  # name of training task
)
