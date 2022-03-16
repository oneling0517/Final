# Final - Kaggle contest

## Introduction
Topic : [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)
We Use classification model and objection detection model to finish this task.

## Training 
procedure:
1. Crop the training data into many training images by using the (label.json)
2. Train the classification model by training images
3. Train the objection detection model by training images
4. Detect the boundary box of fishes in testing data and crop into testing images
5. Predict the class of testing images
6. Ensemble models and clip the final results

Crop the training data into many training images by using the (label.json) 
```
python crop_dataset.py 
```
Train the classification model by training images
```
python train_crop.py --model=resnet50
python train_crop.py --model=regnet_x_8gf 
```
Train the objection detection model by training images
Clone the source of yolov5 
```
git clone https://github.com/ultralytics/yolov5  
```
Install the environment
```
pip install -r requirements.txt    
```
Remember to modify the dir of training images in get_label.py
5. Detect the boundary box of fishes in testing data and crop into testing images
6. Predict the class of testing images
7. Ensemble models and clip the final results

We need to change our version in order to match the version in [requirements.txt](https://github.com/matterport/Mask_RCNN/blob/master/requirements.txt).
First uninstall keras.
```
pip uninstall keras
```
Install keras 2.0.8 and tensorflow 1.15.2
```
%tensorflow_version 1.x
pip install keras==2.0.8
pip install tensorflow-gpu==1.15.2
```
I don't know why there is an error sometimes. If there is an error, you can run this again.
```
pip install keras==2.0.8
```
We also need to install the elder version of h5py
```
pip uninstall h5py
pip install h5py==2.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
## Dataset Download
```
os.chdir("/content/Mask_RCNN")
!gdown --id '1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG' --output dataset.zip

!apt-get install unzi
!unzip -q 'dataset.zip' -d dataset
```

## Training
Use Mask R-CNN resnet101
```
os.chdir("/content/Mask_RCNN/samples/VRDL_HW3")
python3 nucleus.py train --dataset=/content/Mask_RCNN/dataset/dataset --subset=train --weights=imagenet
```

## Validation
```
os.chdir("/content/Mask_RCNN/samples/VRDL_HW3")
python3 nucleus.py detect --dataset=/content/Mask_RCNN/dataset/dataset --subset=val --weights=/content/Mask_RCNN/log/mask_rcnn_nucleus_0019.h5
```

## Testing
Use the weights from [Google Drive](https://drive.google.com/file/d/1Apj1jhAVYkVR-SDFrIpeDchNBDkPjfMd/view?usp=sharing).
```
os.chdir("/content/Mask_RCNN")
!gdown --id '1Apj1jhAVYkVR-SDFrIpeDchNBDkPjfMd' --output weights19.zip

!apt-get install unzi
!unzip -q 'weights19.zip' -d log
```
```
os.chdir("/content/Mask_RCNN/samples/VRDL_HW3")
python3 nucleus.py detect --dataset=/content/Mask_RCNN/dataset/dataset --subset=test --weights=/content/Mask_RCNN/log/mask_rcnn_nucleus_0019.h5
```

## Inference

You can click [Inference.ipynb](https://colab.research.google.com/drive/13vLcOs_x6R_ALSdEjlYYxuOcER0Xr-gd?usp=sharing).

## Reference
https://github.com/matterport/Mask_RCNN/tree/master/samples/nucleus
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
