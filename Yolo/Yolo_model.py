import cv2
import torch
from QtFusion.models import Detector
from datasets.label_name import Chinese_name
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
ini_params = {
    'device': device,
    'conf': 0.25,
    'iou': 0.5,
    'classes': None,
    'verbose': False
}
def count_classes(det_info, class_names):
    count_dict = {name: 0 for name in class_names}
    for info in det_info:
        class_name = info['class_name']
        if class_name in count_dict:
            count_dict[class_name] += 1
    count_list = [count_dict[name] for name in class_names]
    return count_list
class YOLOv8v5Detector(Detector):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = None
        self.img = None
        self.names = list(Chinese_name.values())
        self.params = params if params else ini_params
	def load_model(self, model_path):
	    self.device = select_device(self.params['device'])
	    self.model = YOLO(model_path)
	    names_dict = self.model.names
	    self.names = [Chinese_name[v] if v in Chinese_name else v for v in names_dict.values()]
	    self.model(torch.zeros(1, 3, *[self.imgsz] * 2).to(self.device).type_as(next(self.model.model.parameters())))
	def preprocess(self, img):
	    self.img = img
	    return img
	
	def predict(self, img):
	    results = self.model(img, **ini_params)
	    return results
	
	def postprocess(self, pred):
	    results = []
	    for res in pred[0].boxes:
	        for box in res:
	            class_id = int(box.cls.cpu())
	            bbox = box.xyxy.cpu().squeeze().tolist()
	            bbox = [int(coord) for coord in bbox]
	            result = {
	                "class_name": self.names[class_id],
	                "bbox": bbox,
	                "score": box.conf.cpu().squeeze().item(),
	                "class_id": class_id,
	            }
	            results.append(result)
	    return results
	    
    def set_param(self, params):
        self.params.update(params)
