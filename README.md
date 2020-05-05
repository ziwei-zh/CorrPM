# Correlating Edge, Pose with Parsing

This is a Pytorch implementation of our paper [Correlating Edge, Pose with Parsing](https://arxiv.org/pdf/2005.01431.pdf) accepted by CVPR2020. We propose a Correlation Parsing Machine (CorrPM) utilizing a Heterogeneous Non-Local (HNL) network to capture the correlations among features from human edge, pose and parsing.

## Requirements
Pytorch 0.4.1

Python 3.6

The compile of InPlace-ABN is based on [CE2P](https://github.com/liutinglt/CE2P).

## Implementation
### Dataset
Please download [LIP](http://sysu-hcp.net/lip/overview.php) dataset and make them follow this structure:
```
|-- LIP
    |-- TrainVal_pose_annotations/LIP_SP_TRAIN_annotations.json
    |-- images_labels
        |-- train_images
        |-- train_segmentations
        |-- val_images
        |-- val_segmentations
        |-- train_id.txt
        |-- val_id.txt
```
    

Pose annotation file can be downloaded here [Google drive](https://drive.google.com/open?id=1qlTED6vDHevfl3sr9t8WLVNDEkPahfyK).

### Train
Please download the pre-trained ResNet-101 from [Google drive](https://drive.google.com/open?id=1uTf0wNLS5y0l8jIy06Tewdg8XF0TMSOq) or [Baidu drive](https://pan.baidu.com/s/1Lzjvqpafw9VUO45TcPvhBA) 

TO BE FINISHED

### Test
```bash
./run_eval.sh
```
Our trained model can be downloaded from  [Google drive](https://drive.google.com/open?id=1skvx6qVjh31a0Bff6ad06I82jRTtO-1T) or [Baidu drive](https://pan.baidu.com/s/1XEXfR7--9eqUIn_LnJTlYA).

