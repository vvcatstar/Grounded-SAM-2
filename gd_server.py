import os
import glob
import json
import pickle
import yaml

import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image

import supervision as sv
import tqdm
import random
import time 

from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from torchvision.ops import box_convert
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from flask import Flask, request, jsonify

device = "cuda" if torch.cuda.is_available() else "cpu"


class GroundingDino:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)
        self.load_models(self.configs)
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.box_annotator = sv.BoundingBoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        
    def show_bbox(self, image_path, detections, labels, output_image):
        img = cv2.imread(image_path)
        annotated_frame = self.box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)
        cv2.imwrite(output_image, annotated_frame)
    def load_models(self, config):
        '''
        use for load groundeddino and sam2 models
        input:
        config[dict]: a config dict format data 
        output:
        outputmodel[dict]: GroundedDINO and Sam2 models
        '''
        model_id = config['model_id']
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    def process_image(self, image_path, text_str, output_root, score_thr=0.05, text_thr=0.3, show_result=False):
        image_name = image_path.split('/')[-1].split('.')[0]
        output_result = os.path.join(output_root, image_name+'_gd_detection.json')
        os.makedirs(output_root, exist_ok=True)
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        input_label = text_str.split('.')
            
        text_str = text_str.replace('.', ' .')
        # text = []
        # prompts = text_str.split('.')
        # for prompt in prompts:
        #     text.append([prompt])
        # text.append([' '])
        inputs = self.processor(images=image, text=text_str, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=score_thr,
            text_threshold=text_thr,
            target_sizes=[image.size[::-1]]
        )
        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_names_len = len(class_names)
        class_ids = np.array(list(range(len(class_names))))
        ## get annotate file
        img = cv2.imread(image_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            class_id=class_ids,
            confidence=np.array(confidences),
        )
        # detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        # class_id = detections.class_id
        # labels = [labels[i] for i in class_id]
        # class_names = [class_names[i] for i in class_id]
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        class_ids = detections.class_id
        confidences = detections.confidence
        class_names = [class_names[i] for i in class_ids]
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        if show_result:
            output_image = os.path.join(output_root, image_name+'_gd.png')
            self.show_bbox(image_path, detections, labels, output_image)
        outputs = {
            'image_path': image_path,
            'xyxy': detections.xyxy.tolist(),
            'confidence': detections.confidence.tolist(),
            'class_name': class_names,
        }
        with open(output_result, 'w') as f:
            json.dump(outputs, f)
        return output_result, outputs
    
    
gd_config_file = './grounding_server_config.yaml'
gd_det = GroundingDino(gd_config_file)
app = Flask(__name__)
@app.route('/gd_detection', methods=['POST'])
def gd_detection():
    start_time = time.time()
    data = request.get_json()
    image_path = data['image_path']
    task = data['task']
    output_root = os.path.dirname(image_path)
    output_root = data.get('output_root', output_root)
    text_prompt = data.get('text_prompt', '.')
    score_thr = float(data.get('score_thr', 0.05))
    text_thr = float(data.get('text_thr', 0.3))
    output_file, outputs = gd_det.process_image(image_path, text_prompt, output_root, score_thr=score_thr, text_thr=text_thr,show_result=True)
    response = {}
    response['output_file'] = output_file
    response['outputs'] = outputs
    end_time = time.time()
    use_time = round(end_time - start_time, 3)
    print(use_time)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10004)