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
from tqdm import tqdm
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pickle

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


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



def makeTrainSet(root):
    #load model
    model_checkpoint_path = './model/IMFDB_MOBILEFACE_20190510_142512_Align_1.000/Iter_006000_net.ckpt'
    model = mobilefacenet.MobileFaceNet()
    model.load_state_dict(torch.load(model_checkpoint_path)['net_state_dict'])
    model.eval()

    # transform for input net
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    #label index to star name
    star_dict = {}
    #make ndarray to save features and labels
    FEATURE = []
    LABEL = []

    star_list = os.listdir(root)
    for idx, star in enumerate(tqdm(star_list)):
        image_list = os.listdir(os.path.join(root,star))
        for image in image_list:
            img = cv2.imread(os.path.join(root, star, image))
            #convert gray to rgb
            if len(img.shape) == 2:
                img = to_rgb(img)

            bounding_boxes, facial5points = detectFace(img)

            #can not detect faces
            if len(bounding_boxes) == 0 or len(facial5points)==0 :
                continue
            #detect faces
            else:
                #align_img use to face recognition
                align_img = alignFace(img, bounding_boxes, facial5points)
                align_img_tensor = transform(align_img)
                align_img_tensor = torch.unsqueeze(align_img_tensor, 0)
                feature = model(align_img_tensor)
                FEATURE.append(feature.detach().numpy())
                LABEL.append(idx)
                if idx not in star_dict.keys():
                    star_dict[idx] = star

    FEATURE = np.array(FEATURE)
    LABEL = np.array(LABEL)
    np.save('feature_train.npy', FEATURE)
    np.save('label_train.npy', LABEL)
    fid = open('star_dict.pkl', 'wb')
    pickle.dump(star_dict, fid, pickle.HIGHEST_PROTOCOL)

# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    joblib.dump(model, './model/knn_classifier.model')
    return model

def knn_predict(train_x_path, train_y_path):
    train_x = np.load(train_x_path)
    train_y = np.load(train_y_path)
    train_x = np.squeeze(train_x)
    #split train test set
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.3, random_state=42)
    classifiers = knn_classifier(X_train, y_train)
    predict = classifiers.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predict)
    print ('accuracy: %.2f%%' % (100 * accuracy))

if __name__ == "__main__":
    root = "/home/lab404/Documents/PyData/IMFDB_final_rename"
    train_x_path = "./model/feature_train.npy"
    train_y_path = "./model/label_train.npy"
    #makeTrainSet(root)
    knn_predict(train_x_path, train_y_path)
