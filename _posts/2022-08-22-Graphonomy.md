---
layout: post
title: "Graphonomy setting on ubuntu server"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.2. C++, 1.2. Artificial Intelligence, 1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker]
---

Server SPEC

|OS/Version|CPU|GPU/Version|
|----------|---|-----------|
|Ubuntu-server/20.04|Intel(R) Xeon(R) Silver 4110 CPU|NVIDIA TITAN Xp/515.65.01|

---

### [Graphonomy - Repository](https://github.com/Gaoyiminggithub/Graphonomy)

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
|   0  NVIDIA TITAN Xp     Off  | 00000000:B3:00.0 Off |                  N/A |
| 21%   37C    P0    57W / 250W |      0MiB / 12288MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

# https://hub.docker.com/r/nvidia/cuda

docker pull nvidia/9.0-cudnn7-devel-ubuntu16.04

docker run -i -t --gpus all --shm-size 16gb --name Graphonomy nvidia/9.0-cudnn7-devel-ubuntu16.04
```

### Setup in Container

```bash
# Apt Updata && Upgrade && install
apt-get update && apt-get -y dist-upgrade

apt-get install -y wget git vim build-essential zip libgl1-mesa-glx libglib2.0-0
```

### Install package

```bash
# miniconda install python3.7 version
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
sh Miniconda3-py37_4.12.0-Linux-x86_64.sh

# conda package install
conda install pytorch=0.4.1 cuda90 torchvision -c pytorch
conda install -c anaconda scipy networkx && conda install -c conda-forge tensorboardx opencv matplotlib
```

### Data Preparation

Follow the Repo [Getting Started](https://github.com/Gaoyiminggithub/Graphonomy#getting-started)