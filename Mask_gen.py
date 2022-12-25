import numpy as np
import os
import cv2
import openslide
import sys
import argparse
import logging
from PIL import Image

'''
Generating binary mask of digital slides based on the original tif and given annotation file of tumor or normal area.
Annotation file is in txt form
'''

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '')
print(sys.path)
parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('--level', default=0, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')

def file2array(path, delimiter='  '   ):
    #file2array converts the coordinates of the annotations in each TXT document into a 3-D array
    fp = open(path, 'r', encoding='utf-8')
    string = fp.read()  #a string that contains all the contents of the file
    fp.close()
    row_list = string.splitlines()
    data_list = [[float(i) for i in row.strip().split(delimiter)] for row in row_list]
    return len(data_list),np.array(data_list)

'''
extracting_focus: to extract the outline area
contours is a set of outlined annotations and is a 4-dimensional array, the same as the contours structure in opencv
wsi_wholepath: the path of a single slice to process
args is the parser created above, and the level parameter determines how large the slice should be
out_tissue_mark: the output path
The script processes each slice individually, so there is only one slice to process in the wsi_path file
'''
def extracting_focus(contours,wsi_wholepath,args,out_tissue_mark,out_tissue_tif):
    logging.basicConfig(level=logging.INFO)
    slide = openslide.open_slide(wsi_wholepath)
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                           args.level,
                           slide.level_dimensions[args.level]).convert('RGB')),
                           axes=[1, 0, 2]).copy() #Gets an RGB picture at a specific zoom level
    img_RGB=np.transpose(img_RGB, (1, 0, 2)).copy()
    cv2.drawContours(img_RGB, contours, -1, (0, 0, 0), -1) #Dye the inside of each contour enclosed by a curve black
    hsv = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV) #Convert image to HSV space
    lower_hsv = np.array([0,0,0])
    upper_hsv = np.array([180,255,46])
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv) #get the mask for the black area
    print(mask.shape)
    np.save(out_tissue_mark,mask)
    image = Image.fromarray(mask)
    image.save(out_tissue_tif)


def main():
    contours = []
    annotation_path = r'D:\self_study\medical_imaging\annotation\positive'
    #These are all TXT documents in annotation_path, and each TXT document stores all the coordinates of an Annotation
    for annotation in os.listdir(annotation_path):
    #Integrate the coordinates of all the annotations that need to be processed in the slice into contours
        annotation_wholepath = os.path.join(annotation_path, annotation)
        length, data = file2array(annotation_wholepath)
        annotation = np.reshape(data, (length, 1, 2))
        annotation1=np.array(annotation).reshape((-1, 1, 2)).astype(np.int32)
        contours.append(annotation1)

    args = parser.parse_args()
    wsi_path = r'D:\self_study\medical_imaging\digital_slide_sample'
    for wsi in os.listdir(wsi_path):
        wsi_wholepath = os.path.join(wsi_path, wsi)
        (wsi_path, wsi_extname) = os.path.split(wsi_wholepath)
        (wsi_name, extension) = os.path.splitext(wsi_extname)
        out_tissue_mark = r'D:\self_study\medical_imaging\mask\testslide.npy'
        out_tissue_tif = r'D:\self_study\medical_imaging\mask\testslide.tif'#% wsi_name
        extracting_focus(contours, wsi_wholepath, args, out_tissue_mark, out_tissue_tif)
        print("finish,{}\n".format(wsi_name))


if __name__ == '__main__':
    main()