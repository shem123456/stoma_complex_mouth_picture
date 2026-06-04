# 气孔图像识别
# 参数：气孔数量，开闭数量
# 使用yolov3-tiny-open-close进行气孔状态开闭识别
# 增加结果输出，保存为csv文件

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
from tqdm import tqdm
from utils.utils import yolo_detect
import sys
import pandas as pd
import time
from unet import Unet
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取开闭状态的yolo-tiny-open-close模型参数
weightsPath1 = "yolov3_tiny_open_close/stoma-yolov3-tiny_40000.weights"
configPath1 ="yolov3_tiny_open_close/stoma-yolov3-tiny.cfg"
labelsPath1 = "yolov3_tiny_open_close/stomata_class.txt"

# 设置气孔复合体和气孔开口语义分割模型
unet = Unet()
name_classes  = ["_background_","stoma","mouth"]
# 设置统计每个类的像素点个数
count         = True

# opencv和PIL转换
def cv2PIL(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))

def PIL2cv(img_pil):
    return cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)

#轮廓面积计算函数
def areaCal(contour):

    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour[i])

    return area

#轮廓周长计算函数
def perimeterCal(contour):

    area = 0
    for i in range(len(contour)):
        area += cv2.arcLength(contour[i],closed = True)

    return area


def detect_img(image):
    # 语义分割
    image_PIL = cv2PIL(image)
    # 获得掩模图和每个类像素点个数
    r_image,class_number = unet.detect_image(image_PIL, count=count, name_classes=name_classes)
    r_image = PIL2cv(r_image)
    # print(r_image.shape)

    # 寻找轮廓，计算周长
    # 转灰度
    r_image_gray = cv2.cvtColor(r_image,cv2.COLOR_BGR2GRAY)
    # 二值化
    thred1,binary1 = cv2.threshold(r_image_gray,0,255,cv2.THRESH_BINARY)
    # cv2.imshow("binary1",binary1)
    # 取气孔开口（2,2,2）
    thred2,binary2 = cv2.threshold(r_image_gray,3,255,cv2.THRESH_BINARY_INV)  
    # 去交集
    binary2 = cv2.bitwise_and(binary1, binary2)
    # cv2.imshow("binary2",binary2)
  
    # 查找轮廓
    # 气孔复合体
    contours1 ,hierarchy1 = cv2.findContours(binary1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # 气孔开口
    contours2 ,hierarchy2 = cv2.findContours(binary2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 复合体面积周长
    area_stoma_complex = areaCal(contours1)
    perimeter_stoma_complex = perimeterCal(contours1)
    # print("area_stoma_complex:",area_stoma_complex)
    # print("perimeter_stoma_complex:",perimeter_stoma_complex)

    # 开口面积周长
    area_stoma_mouth = areaCal(contours2)
    perimeter_stoma_mouth = perimeterCal(contours2)
    # print("area_stoma_mouth:",area_stoma_mouth)
    # print("perimeter_stoma_mouth:",perimeter_stoma_mouth)

    # 绘制轮廓
    cv2.drawContours(r_image,contours1,-1,(211,123,255),2)
    cv2.drawContours(r_image,contours2,-1,(111,123,255),2)
    # cv2.imshow("r_image",r_image)

    #初始化标签值
    LABELS = ["close","open"]

    color = (255,255,20)

    # 识别气孔开闭
    match_result1 = yolo_detect(image,weightsPath1,configPath1,labelsPath1)

    # 初始化开闭气孔的个数
    close = 0
    open = 0

    # 画矩形
    for tracker in match_result1:
        # print(tracker)
        # 统计开闭气孔数量
        if tracker[-1] == 0:
            close += 1
        else:
            open += 1

        color_tracker = (255,255,0)
        cv2.rectangle(image, (int(tracker[0])-10, int(tracker[1])-10), (int(tracker[2])+10, int(tracker[3])+10), 
                      color_tracker, 1)
        text = "{}".format(LABELS[tracker[-1]])
        cv2.putText(image, text, (tracker[0], tracker[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # 混合检测和语义分割图像
    image_mixed = cv2.addWeighted(image,1,r_image,0.5,0)

    return image_mixed,close,open,class_number,perimeter_stoma_complex,perimeter_stoma_mouth


if __name__ == "__main__":
    path = r"D:\zxn\pho\DR-V16\Picture"#要改路径
    print(f"Current working directory: {os.getcwd()}")
    print(f"Reading from directory: {os.path.abspath(path)}")
    image_list = os.listdir(path)
    print(f"Files in directory: {image_list}")
    result = []
    for image_path in image_list:
        print("image name:",image_path)
        # 使用PIL读取图像以支持中文路径
        try:
            image_pil = Image.open(os.path.join(path, image_path))
            image = PIL2cv(image_pil)
        except Exception as e:
            print(f"Failed to read image: {os.path.join(path, image_path)}")
            print(f"Error: {e}")
            continue
        image_mixed,close,open,class_number,perimeter1,perimeter2 = detect_img(image)
        temp_dict = {
                     "image_name":image_path,
                     "stoma_number":close+open,
                     "close_stoma":close,
                     "open_stoma":open,
                     "stoma_complex_area":class_number[1]+class_number[2],
                     "stoma_mouth_area":class_number[2],
                     "stoma_complex_perimeter":perimeter1,
                     "stoma_mouth_perimeter":perimeter2
                     }
        result.append(temp_dict)
        output_dir = r"D:\zxn\pho\DR-V16\Picture-out"#要改路径
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, image_path)
        print(f"Saving to: {output_path}")
        # Use PIL to save the image
        image_pil = cv2PIL(image_mixed)
        try:
            image_pil.save(output_path)
            print("Save successful: True")
        except Exception as e:
            print(f"Save failed: {e}")
    
    df = pd.DataFrame(result)
    df.to_csv(r"D:\zxn\pho\DR-V16\Picture-out\result.csv",index=False)#要改路径
    



