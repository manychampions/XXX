import os
import cv2
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

import torch.utils.data
from backbone import mobilefacenet, resnet, arcfacenet, cbam
from mtcnnalign.align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnnalign.mtcnn.detector import detect_faces
import torchvision.transforms as transforms
from sklearn.externals import joblib
import pickle


def detectFace(img):
    """
        input : image numpy array
        output : two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
                 bounding boxes and facial landmarks.
    """
    img = Image.fromarray(img).convert('RGB')
    bounding_boxes, facial5points = detect_faces(img)
    return bounding_boxes, facial5points

def alignFace(raw, bounding_boxes, facial5points):
    """
        input : raw : numpy array, bounding boxes and facial landmarks.
        output : align img
    """
    if len(facial5points) == 0:
        align_img = raw
    else:
        facial5points = np.reshape(facial5points[0], (2, 5))
        crop_size = (112, 112)

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        output_size = (112, 112)

        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)

        align_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
        return align_img

# load model
model_checkpoint_path = './model/IMFDB_MOBILEFACE_20190510_142512_Align_1.000/Iter_006000_net.ckpt'
model = mobilefacenet.MobileFaceNet()
model.load_state_dict(torch.load(model_checkpoint_path)['net_state_dict'])
model.eval()

# transform for input net
transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
])

# 读取视频
cap = cv2.VideoCapture("/home/lab404/Desktop/myOpenCV/3idots.avi")
fps = cap.get(cv2.CAP_PROP_FPS)
totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps)
print(totalFrameNumber)
COUNT = 0

#load knn trained model.
knn_model = joblib.load('./model/knn_classifier.model')

#load star dict to get idx for star name
fid = open("./model/star_dict.pkl","rb")
star_dict = pickle.load(fid)

while COUNT < totalFrameNumber:
    ret, frame = cap.read()  #BGR

    #bbox use to display, facial5points use to align face
    bounding_boxes, facial5points = detectFace(frame)

    #fliter the invaild bboxes
    #vaild_bboxes = [bbox for bbox in bounding_boxes if bbox is not None]
    if len(bounding_boxes) == 0 or len(facial5points)==0 :
        cv2.imshow('video', frame)
        COUNT = COUNT + 1
        cv2.waitKey(1)
    else:
        #align_img use to face recognition
        for idx in range(len(bounding_boxes)):
            bbox = bounding_boxes[idx]
            facial5 = facial5points[idx]
            x1 = int(bbox[0]);y1 = int(bbox[1]); x2 = int(bbox[2]); y2 = int(bbox[3]); prob = bbox[4]
            print(prob)

            align_img = alignFace(frame, bbox, facial5points)
            align_img_tensor = transform(align_img)
            align_img_tensor = torch.unsqueeze(align_img_tensor, 0)
            feature = model(align_img_tensor)
            predict = knn_model.predict(feature.detach().numpy())
            star_name = star_dict[predict[0]]

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, star_name, (x1-5,y1-5), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        cv2.imshow('video', frame)
        COUNT = COUNT + 1
        cv2.waitKey(1)
cap.release();
