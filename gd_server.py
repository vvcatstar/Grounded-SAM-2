import os
import yaml

import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image

import supervision as sv
import time 

from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from torchvision.ops import box_convert
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from flask import Flask, request, jsonify
from IPython import embed 

class GroundingDino:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)
        self.device = self.configs['device']
        if 'npu' in self.device:
            import torch
            import torch_npu 
            from torch_npu.contrib import transfer_to_npu 
        self.load_models(self.configs['gd_server'])
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.box_annotator = sv.BoundingBoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        with open('../config.yaml','r') as f:
            self.post_config = yaml.load(f, Loader=yaml.Loader)
        self.post_processer = PostProcess(self.post_config)
        
    def inference(self, image, text_prompt, score_thr, text_thr):
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=score_thr,
            text_threshold=text_thr,
            target_sizes=[image.size[::-1]]
        )
        return results[0]
    def post_boat(self, image, results):
        input_boxes = results['boxes']
        class_names = results['labels']
        person_details = []
        for index in range(len(input_boxes)):
            if 'rowboat' in class_names[index] or 'raft' in class_names[index]:
                # class_names[index] = 'rowboat'
                bbox = input_boxes[index]
                x_min, y_min, x_max, y_max = map(int, bbox)
                width = x_max - x_min
                height = y_max - y_min
                new_x_min = max(x_min - width , 0)
                new_y_min = max(y_min - height , 0)
                new_x_max = min(x_max + width , image.size[0])
                new_y_max = min(y_max + height , image.size[1])
                cropped_image = image.crop((new_x_min, new_y_min, new_x_max, new_y_max))
                person_result = self.inference(cropped_image, 'person.', 0.35, 0.3)
                for j in range(len(person_result["boxes"])):
                    p_x1 = float(person_result["boxes"][j][0].detach().cpu().numpy() + new_x_min)
                    p_y1 = float(person_result["boxes"][j][1].detach().cpu().numpy() + new_y_min)
                    p_x2 = float(person_result["boxes"][j][2].detach().cpu().numpy() + new_x_min)
                    p_y2 = float(person_result["boxes"][j][3].detach().cpu().numpy() + new_y_min)
                    conf = float(person_result["scores"][j].detach().cpu().numpy())
                    person_details.append([p_x1, p_y1, p_x2, p_y2, conf, 'person'])
        for person in person_details:
            results['boxes'].append(person[:4])
            results['labels'].append(person[5])
            results['scores'].append(person[4])
        return results 
    
        
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
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
    
    def process_image(self, image_path, text_str, output_root, task,  score_thr=0.05, text_thr=0.3, show_result=False, bboxInfos=[]):
        image_name = image_path.split('/')[-1].split('.')[0]
        if type(image_path) == str:
            image = Image.open(image_path)
        else:
            image = Image.fromarray(np.array(image_path))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        input_label = text_str.split('.')
            
        text_str = text_str.replace('.', ' . ')
        results = self.inference(image, text_str, score_thr, text_thr)
        results["boxes"] = results["boxes"].cpu().numpy().tolist()
        results["scores"] = results["scores"].cpu().numpy().tolist()
        # results = self.post_boat(image, results)
        input_boxes = results["boxes"]
        confidences = results["scores"]
        class_names = results["labels"]
        class_ids = np.array(list(range(len(class_names))))
        outputs = {
            'image_path': image_path,
            'xyxy': input_boxes,
            'confidence': confidences,
            'class_name': class_names,
        }
        output_result = os.path.join(output_root, image_name+'_gd_detection.json')
        return output_result, outputs
    
    
gd_config_file = '../config.yaml'
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
    bboxInfos = data.get('bboxInfos', [])
    output_file, outputs = gd_det.process_image(image_path, text_prompt, output_root, task, score_thr=score_thr, text_thr=text_thr,show_result=False, bboxInfos=bboxInfos)
    # outputs, annote_image = gd_det.post_processer.post_task(task, outputs)
    # with open(output_file, 'w') as f:
    #     json.dump(outputs, f)
    # cv2.imwrite(output_file.replace('.json', '.jpg'), annote_image)
    end_time = time.time()
    use_time = round(end_time - start_time, 3)
    print(use_time)
    return jsonify(outputs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10004)