---
layout: post
title: "Graphonomy-Panoptic setting on ubuntu server"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.2. C++, 1.2. Artificial Intelligence, 1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker]
---

Server SPEC

|OS/Version|CPU|GPU/Version|
|----------|---|-----------|
|Ubuntu-server/20.04|AMD Ryzen Threadripper PRO 5955WX|A6000/515.65.01|

---

### [Graphonomy-Panoptic - Repository](https://github.com/Gaoyiminggithub/Graphonomy-Panoptic)

---

### Install Container

```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A6000    Off  | 00000000:41:00.0 Off |                  Off |
| 30%   57C    P0    87W / 300W |      0MiB / 49140MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

# https://hub.docker.com/r/nvidia/cuda

docker pull nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

docker run -i -t --gpus all --shm-size 16gb --name Graphonomy-Panoptic nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04
```

### Setup in Container

```bash
# Apt Updata && Upgrade && install
apt-get update && apt-get -y dist-upgrade

apt-get install -y wget git vim build-essential python3 python3-pip zip libgl1-mesa-glx libglib2.0-0
```

### Install package

[Graphonomy-Panoptic/INSTALL.md](https://github.com/Gaoyiminggithub/Graphonomy-Panoptic/blob/main/INSTALL.md)

```
# pip3 install package
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 && pip3 install opencv-python

# Build Detectron2 from Source
mkdir /workspace && cd /workspace

git clone https://github.com/Gaoyiminggithub/Graphonomy-Panoptic.git

cd Graphonomy-Panoptic

pip3 install -e detectron2

pip3 install git+https://github.com/cocodataset/panopticapi.git
```

### [Data Preparation](https://github.com/Gaoyiminggithub/Graphonomy#getting-started)

[Graphonomy-Panoptic/GETTING_STARTED.md](https://github.com/Gaoyiminggithub/Graphonomy-Panoptic/blob/main/GETTING_STARTED.md)

[COCO dataset](https://github.com/Gaoyiminggithub/Graphonomy-Panoptic/blob/main/GETTING_STARTED.md#coco-dataset)

```
mkdir /workspace/Graphonomy-Panoptic/detectron2/datasets/coco && cd /workspace/Graphonomy-Panoptic/detectron2/datasets/coco

# 1. Download and extract COCO 2017 train and val images with annotations from http://cocodataset.org.
# Dataset -> Download -> Images -> 2017 Val images [5K/1GB], 2017 Train images [118K/18GB]
wget http://images.cocodataset.org/zips/val2017.zip && wget http://images.cocodataset.org/zips/train2017.zip
unzip val2017.zip && unzip train2017.zip

# 2. Download panoptic annotations from COCO website.
# Dataset -> Download -> Annotations -> 2017 Panoptic Train/Val annotations [821MB]
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
mv annotations/panoptic_train2017.zip annotations/panoptic_val2017.zip ./
unzip panoptic_train2017.zip && unzip panoptic_val2017.zip

# 3. Download the pre-processing panopitc segmentation annotations from here(Google Drive).
# Download on Google Drive file PanopticStuffAnno.tar(672M)
tar -xvf PanopticStuffAnno.tar
mv PanopticAnnotation/* annotations

rm -rf panoptic_annotations_trainval2017.zip panoptic_train2017.zip panoptic_val2017.zip val2017.zip train2017.zip PanopticStuffAnno.tar PanopticAnnotation

# 4. prepare the data as the following structure:
# detectron2/
#    datasets/
#         coco/
#           {train,val}2017/
#           panoptic_{train,val}2017/  # png annotations
#           annotations/
#             panoptic_{train,val}2017.json
#             panoptic_{train,val}2017_trans/  # pre-processing panoptic segmentation png annotations
```

[ADE20K dataset](https://github.com/Gaoyiminggithub/Graphonomy-Panoptic/blob/main/GETTING_STARTED.md#ade20k-dataset)

```
mkdir /workspace/Graphonomy-Panoptic/detectron2/datasets/ADE20K_2017/ && cd /workspace/Graphonomy-Panoptic/detectron2/datasets/ADE20K_2017/

# 1. Download and extract the ADE20K dataset train and val images from http://sceneparsing.csail.mit.edu/.
# DOWNLOADS -> Instance Segmentation -> download for Data: [Images(851MB)]
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/images.tar
tar -xvf images.tar

# DOWNLOADS -> Instance Segmentation -> download for Data: [Annotations(86MB)]
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xvf annotations_instance.tar
mv annotations_instance new_segment_anno_continuous

# 2. Download the annotations for panoptic segmentation from here (Google Drive).
# Download on Google Drive file ADE_download.zip(148M)
unzip ADE_download.zip
mv ADE_download/json/* /workspace/Graphonomy-Panoptic/detectron2/datasets/ADE20K_2017/

rm -rf ADE_download.zip annotations_instance.tar images.tar ADE_download

# 3. prepare the data as the following structure:
# detectron2/
#   datasets/
#    ADE20K_2017/
#      images/
#        training/
#        validation/
#      new_segment_anno_continuous/
#        training/
#        validation/
#      ade_{train,val}_things_only.json
#      panoptic_ade20k_val_iscrowd.json
```