Attribute-guided Feature Extraction and Augmentation Robust Learning for Vehicle Re-identification
=================
In AIcity2020 Track2, we rank number 6(team ID 44) among all the teams with the mAP of 0.6683 on server.

# Authors
Chaoran Zhuge, Yujie Peng, Yadong Li, Jiangbo Ai, Junru Chen

# our origin results
our origin results on a deep learning framework without open source:
|backbone|mAP server|
|---|---|
|res50|0.614|
|res50+res50attr|0.638|
|res50+res50attr+dense161bs|0.657|
|res50+res50attr+dense161bs+dense161|0.662|
|res50+res50attr+dense161bs+dense161+hrnetw18c|0.6683|

# our reproducing results on pytorch
We have already reproduced our method on pytorch.The final result is a little lower than our origin results,which may caused by the difference of deep learning frameworks. 
|backbone|mAP local|mAP server|
|---|---|---|
|res50|0.617|0.6016|
|res50+res50attr|0.6436|0.6286|
|res50+res50attr+hrnetw18c|0.6527|0.6373|
|res50+res50attr+hrnetw18c+dense161bs|0.6667|0.6506|
|res50+res50attr+hrnetw18c+dense161bs+dense161|0.6730|0.6568|

# requirements
```
1.pytorch==1.4.0
2.torchvision==0.5.0
3.python==3.6
4.opencv-python==4.1.1.264
5.apex==0.1
```
if you use python3.7,please use the follwing orders to install apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
if you use python 3.6,please use the follwing orders to install apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```
Using apex to train can reduce video memory usage by 25% -30%,and the trained model has the same performance as not using apex.

# Preparing the dataset
If you want to reproduce our imagenet pretrained models,you need download ILSVRC2012 dataset,and make sure the folder architecture as follows:
```
ILSVRC2012
|
|-----train----1000 sub classes folders
|
|-----val----1000 sub classes folders
Please make sure the same class has same class folder name in train and val folders.
```
Or,you can use our pretrained models from here: https://drive.google.com/open?id=1nXHXbrmWuOCzDsMumWoXy9Atv60G0Guy.

If you want to reproduce our aicity track2 result,you need download AIC20_ReID dataset and AIC20_ReID_Simulation dataset,and make sure the folder architecture as follows:
```
AIC20_track2
|
|                |----image_train
|----AIC20_ReID--|----image_query
|                |----image_test
|
|----AIC20_ReID_Simulation----image_train
```
Also,you need download our dataset pkls to train and test.You can download it from here: https://drive.google.com/open?id=1zMvJhIl05X6goGu9SvxaXWEkMt99LDsF.
Unzip it,you will get track2_train_pytorch.pkl,track2_simu_train_pytorch.pkl and benchmark_pytorch.pkl.
## track2_train_pytorch.pkl
all images information in AIC20_ReID/image_train/.
```
track2_train_pytorch.pkl is a python list,each element is a python dictionary for a image.
keys of each dictionary item:
'vehicle_id'
'camera_id'
'bbox':predicted by a object detection model trained on COCO
'mask':predicted by a object segmentation model trained on COCO
'color':manually annotated vehicle color attributes
'type':manually annotated vehicle type attributes
'image_path':image relative path
```

## track2_simu_train_pytorch.pkl
all images information in AIC20_ReID_Simulation/image_train/.
```
track2_simu_train_pytorch.pkl is a python list, each element is a python dictionary for a image.
keys of each dictionary item:
'vehicle_id'
'camera_id'
'bbox':predicted by a object detection model trained on COCO
'mask':predicted by a object segmentation model trained on COCO
'color':manually annotated vehicle color attributes
'type':manually annotated vehicle type attributes
'image_path':image relative path
```

## benchmark_pytorch.pkl
all data information in AIC20_ReID/image_query/ and AIC20_ReID/image_test/.
```
track2_simu_train_pytorch.pkl is a python dictionary which has two key:'reid_query' and 'reid_gallery', each key's value is a python list of corresponding sets.
'reid_query' value has all images information in AIC20_ReID/image_query/, and 'reid_gallery' has all images information in AIC20_ReID/image_test/.
each image key:
'vehicle_id'
'track_id':According to AIC20_ReID/test_track_id.txt,We numbered track_id for reid_gallery images,this track id is different from track_id in training set.
'bbox':predicted by a object detection model trained on COCO
'pred_color':predicted by a color classification model trained on AIC20_ReID/image_train/
'pred_type':redicted by a type classification model trained on AIC20_ReID/image_train/
'pred_color_prob':color prediction prob.
'pred_type_prob':type prediction prob.
'image_path':image relative path
```

# pretrained models training
We use a densenet161 model pretrained on imagenet with input size 320x320 and a resnet50+last_stride+ibn with input size 320x320.
our pretrained models performance:
|backbone|epoch|top1 acc|
|---|---|---|
|res50+last stride+ibn 320x320|100|78.292|
|densenet161 320x320|100|79.208|

If you want to reproduce our pretrained models.You can find the training code in imagenet_experiments folder.
For example,if you want to reproduce resnet50+last_stride+ibn,just enter imagenet_experiments/resnet50_320_ibn,then run:
```
./train.sh
```
our training needs 8 2080ti gpus.

# five feature extractors training
In vehicle_reid_experiments,we trained five feature extractors and ensemble all feature extractors dists and features to get the final ensemble result.
You can find each feature extractor training code in vehicle_reid_experiments folder.In our training code,We test per 2 epoch model after epoch 60 and save the best model, dist,features,txt for each feature extractor.
For example,if you want to reproduce our res50 feature extractor,just enter vehicle_reid_experiments/res50,and run:
```
./train.sh
```
Please make sure you can load the pretrained models and our dataset_pkls correctly.
# single feature extractor testing and ensemble testing
For each feature extractor training code,you can get a best model and related dist pkl, query features pkl,gallery features pkl,and txt for server commit.
You can find Each feature extractor bese model performance at the end of training_folder/log/__main__.info.log.Just like:
```
finish training, best_map: 0.xxxx, best_top1:0.xxxx
```
This result are local result eval from our val function,same rules as server.But for some unknown reason,the results always higher 1.0%-1.3% map than server result.
If you want to get ensemble results,just enter public/reid/,and run:
```
./val_ensemble1.sh
```
Please ensure that the order of each pkl in dists/query features/gallery features are same.

# Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={Attribute-guided Feature Extraction and Augmentation Robust Learning for Vehicle Re-identification},
 author={Chaoran Zhuge, Yujie Peng, Yadong Li, Jiangbo Ai, Junru Chen},
 booktitle={AI City Challenge 2020 CVPR Workshop},
 year={2020}
}
```