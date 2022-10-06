# YOLOv4-tiny-attention-Pytorch
YOLOV4-Tiny：You Only Look Once-Tiny
## YOLOV4-Tiny：You Only Look Once-Tiny
---

## 目录

1. [Environment](#Environment)
2. [How to train](#How to train)

## Environment
torch==1.2.0
## How to train

### Train
1. make your data sets  
labels in VOCdevkit/VOC2007/Annotation   
images in VOCdevkit/VOC2007/JPEGImages   

2. run voc_annotation.py to get 2007_train.txt and 2007_val.txt for training

3. Training by running train.py  

4. Predicting by yolo.py and predict.py
