# Final - Kaggle contest

## Introduction
Topic : [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring).

We Use classification model and objection detection model to finish this task.

## Training 
procedure:
1. Crop the training data into many training images by using the json file in datasets folder.
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
1. Clone the source of yolov5 
```
git clone https://github.com/ultralytics/yolov5  
```
2. Install the environment
```
pip install -r requirements.txt    
```
3. Remember to modify the dir of training images in get_label.py

4. Execute the (label_mask.py) to obtain the training labels in txt (N txt for N images)
```
python label_mask.py  
```
5. Execute the (train.py) to training the model
```
python train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt
```

## Inference
You can download [model weight](https://drive.google.com/drive/folders/104ZJATHoQJcIoAiDLS3PKJUOY3oAirGN?usp=sharing) included yolov5, ResNet50 and RegNet. And put the model weights in `model` directory.

Detect the boundary box of fishes in testing data and crop into testing images:

```
python detect.py --weights=model/best.pt --source=test_stg1/ --save-crop
python detect.py --weights=model/best.pt --source=test_stg2/ --save-crop
```
          
Predict the class of testing images:

```
python inference_crop.py --model=regnet_x_8gf --output=regnet_x_8gf_crop.csv
python inference_crop.py --model=resnet50 --output=resnet50_epoch10_crop.csv
```

Ensemble models and clip the final results:

```
python ensemble.py
```

## Reference
1. yolov5: https://github.com/ultralytics/yolov5  
2. object detection label dataset1: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/25428  
3. object detection label dataset2: https://github.com/autoliuweijie/Kaggle/tree/master/NCFM/datasets  
4. K-fold Cross-Validation: https://github.com/lidxhaha/Kaggle_NCFM  
5. InceptionV3 network : https://github.com/pengpaiSH/Kaggle_NCFM
6. Contributors : https://github.com/gyes00205/NYCU_VRDL_Final.git


