import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import detect_faces

def singleTest(img_fn, align_path):
    print("[ALIGN IMAGE] : {}".format(img_fn))
    raw = cv2.imread(img_fn, True)
    img = Image.open(img_fn).convert('RGB')
    _, facial5points = detect_faces(img)
    if len(facial5points) == 0:
        cv2.imwrite(align_path, raw)  #save src image
    else:
        facial5points = np.reshape(facial5points[0], (2, 5))
        crop_size = (112, 112)

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        output_size = (112, 112)

        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)

        dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
        cv2.imwrite(align_path, dst_img)

def powerattackAligh(root, align_root):
    nodetectList = []
    for root, dirs, files in os.walk(root):
        for file in files:
            img_fn = os.path.join(root,file)
            print("[ALIGN IMAGE] : {}".format(img_fn))
            raw = cv2.imread(img_fn, True)
            img = Image.open(img_fn).convert('RGB')
            _, facial5points = detect_faces(img)
            if len(facial5points) == 0:
                cv2.imwrite(os.path.join(align_root, file), raw)  #save src image
                nodetectList.append(file)
            else:
                facial5points = np.reshape(facial5points[0], (2, 5))
                crop_size = (112, 112)

                default_square = True
                inner_padding_factor = 0.25
                outer_padding = (0, 0)
                output_size = (112, 112)

                # get the reference 5 landmarks position in the crop settings
                reference_5pts = get_reference_facial_points(
                    output_size, inner_padding_factor, outer_padding, default_square)

                # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
                dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
                cv2.imwrite(os.path.join(align_root, file), dst_img)
                #img = cv.resize(raw, (250, 250))
                #cv.imwrite('images/{}_img.jpg'.format(i), img)
    name = ['NoDetect']
    nodetect_dataframe = pd.DataFrame(columns=name, data=nodetectList)
    nodetect_dataframe.to_csv('./nodetectList.csv',encoding='gbk')

def powerrattackrenameAligh(rename_root, rename_align_root):
    nodetectList = []
    for root, dirs, files in os.walk(rename_root):
        for file in files:
            img_fn = os.path.join(root,file)
            print("[ALIGN IMAGE] : {}".format(img_fn))
            raw = cv2.imread(img_fn, True)
            img = Image.open(img_fn).convert('RGB')
            _, facial5points = detect_faces(img)
            if len(facial5points) == 0:
                align_folder = os.path.join(rename_align_root, root.split("/")[-1])
                if not os.path.exists(align_folder):
                    os.makedirs(align_folder)
                cv2.imwrite(os.path.join(align_folder, file), raw)  #save src image
                nodetectList.append(file)
            else:
                facial5points = np.reshape(facial5points[0], (2, 5))
                crop_size = (112, 112)

                default_square = True
                inner_padding_factor = 0.25
                outer_padding = (0, 0)
                output_size = (112, 112)

                reference_5pts = get_reference_facial_points(
                    output_size, inner_padding_factor, outer_padding, default_square)

                dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
                align_folder = os.path.join(rename_align_root, root.split("/")[-1])
                if not os.path.exists(align_folder):
                    os.makedirs(align_folder)
                cv2.imwrite(os.path.join(align_folder, file), dst_img)

    name = ['NoDetect']
    nodetect_dataframe = pd.DataFrame(columns=name, data=nodetectList)
    nodetect_dataframe.to_csv('./nodetectList.csv',encoding='gbk')



if __name__ == "__main__":
    root = "D:/YeJQ/IMFDB_final_rename/"
    align_root = "D:/YeJQ/IMFDB_final_rename_Align/"
    powerrattackrenameAligh(root, align_root)
