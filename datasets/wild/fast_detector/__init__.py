"""
This code performs a real-time face and landmark detections
1. Use a light-weight face detector (ONNX): https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
2. Use mobilefacenet as a light-weight landmark detector (OpenVINO: 10 times faster than ONNX)
Date: 09/27/2020 by Cunjian Chen (ccunjian@gmail.com)
"""
from pygments.unistring import No
import torch
from facenet_pytorch import InceptionResnetV1
from torch._C import dtype
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import cv2
import numpy as np
import onnx
import os
import onnxruntime as ort
import onnx
from torchvision.transforms.functional import crop
import cv2
from openvino.inference_engine import IECore

try:
    from .vision.utils import box_utils_numpy as box_utils; from .common.utils import BBox
except:
    from vision.utils import box_utils_numpy as box_utils; from common.utils import BBox


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class FaceDetector:
    MODEL_BASE_PATH = 'fast_detector/models'


    def __init__(self, batch_size=32, threshold=0.9, target_face_embedding=None) -> None:
        ie = IECore()
        model_bin = os.path.splitext(f"{self.MODEL_BASE_PATH}/mobilefacenet.xml")[0] + ".bin"
        net = ie.read_network(model=f"{self.MODEL_BASE_PATH}/mobilefacenet.xml", weights=model_bin)
        self.input_blob = next(iter(net.inputs))

        self.exec_net = ie.load_network(network=net, device_name="CPU")

        onnx_path = f"{self.MODEL_BASE_PATH}/version-RFB-320.onnx"
        
        self.ort_session = ort.InferenceSession(onnx_path)
        self.fd_name = self.ort_session.get_inputs()[0].name

        self.batch_size = batch_size
        self.threshold = threshold

        self.target_face_embedding = target_face_embedding
        if self.target_face_embedding is not None:
            self.face_recognition = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            self.face_recog_resize = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Lambda(lambda im: (im.float() - 127.5) / 128.0),
            ])


    def __call__(self, input):
        images = (input - torch.tensor([127, 127, 127])) / 128
        images = images.permute(0, 3, 1, 2)
        
        N, C, H, W = images.shape
        resized_images = F.interpolate(images.float(), (240, 320), mode='bicubic', align_corners=True).numpy()

        input = input.numpy()

        faces = list()
        for i in range(resized_images.shape[0]):
            resized_image = resized_images[i: i + 1]
            confidences, boxes = self.ort_session.run(None, {self.fd_name: resized_image})
            
            boxes, labels, probs = self.predict(W, H, confidences, boxes, self.threshold)
            
            if self.target_face_embedding is None:
                box = self.get_center_face(W, H, boxes)
            else:   
                cropped_faces = list() 
                fr_compatible_boxes = list()
                for box in boxes:
                    left, top, right, bottom = box
                    fr_compatible_boxes.append([top, right, bottom, left])

                    x1, y1, x2, y2 = box
                    cropped_faces.append(self.face_recog_resize(torch.from_numpy(input[i, y1: y2, x1: x2]).permute(2, 0, 1)))

                if len(cropped_faces) == 0:
                    return None
            
                cropped_faces = torch.stack(cropped_faces).to(device)
                with torch.no_grad():
                    face_encodings = self.face_recognition(cropped_faces)
                                
                face_distances = np.array([(self.target_face_embedding - e2).norm().cpu().numpy() for e2 in face_encodings])
                
                if not np.any(face_distances < 0.9):
                    return None
                
                result_idx = np.argmin(face_distances)
                box = boxes[result_idx]


            if box is None: 
                faces.append(None)
                continue
            
            
            landmark = self.predict_landmarks(input[i], box)
            
            landmark = landmark.astype(dtype=np.int)
            box = np.array(box, dtype=np.int)
            box[box < 0] = 0

            faces.append([box, landmark])

        return faces

    def get_center_face(self, W, H, boxes):
        if len(boxes) == 0:
            return None
        if len(boxes) == 1:
            return boxes[0]
        
        CX = W // 2
        CY = H // 2

        min_distance, min_idx = 1e6, 0
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1     

            cx = x1 + w // 2
            cy = y1 + h // 2
            
            d = (CX - cx) ** 2 + (CY - cy) ** 2

            if min_distance > d:
                d = min_distance
                min_idx = idx
        
        box = boxes[min_idx]
    
        return box

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def predict_landmarks(self, image, box):
        height, width, _ = image.shape
        x1, y1, x2, y2 = box

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(max([w, h]))
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)   

        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        
        face = image[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            face = cv2.copyMakeBorder(face, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
        
        cropped_face = cv2.resize(face, (112, 112))

        if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
            return None

        test_face = cropped_face.astype(dtype=np.float32) / 255.0
        test_face = test_face.transpose((2, 0, 1))
        test_face = test_face.reshape((1,) + test_face.shape)  
        
        # OpenVINO Inference             
        outputs = self.exec_net.infer(inputs={self.input_blob: test_face})
        key = list(outputs.keys())[0]
        output = outputs[key]

        landmark = output[0].reshape(-1,2)
        landmark = new_bbox.reprojectLandmark(landmark)
        
        return landmark


def main():
    FaceDetector.MODEL_BASE_PATH = 'models'
    fd = FaceDetector()
    cap = cv2.VideoCapture('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech/test/_0kBqBMNfOc_90.000000_97.520000.mp4')  # capture from camera
    
    images = list()
    while True:
        ret, image = cap.read()
        if image is None:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        
    
    images = np.array(images)
    
    # images = np.transpose(images, [0, 3, 1, 2])
    images = torch.from_numpy(images)

    t = time.time()
    fd(images)
    print(time.time() - t)

    threshold = 0.7


if __name__ == '__main__':
    main()