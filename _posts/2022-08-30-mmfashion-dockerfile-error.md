---
layout: post
title: "mmfashion Dockerfile GPG error"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

### [mmfashion - Issues](https://github.com/open-mmlab/mmfashion/issues/147#issuecomment-1231127145)

### Error message

```
W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease' is not signed.
```

### mmfashion/docker/Dockerfile

change like that

```Dockerfile
ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get -y dist-upgrade \
    && apt-get install -y git libglib2.0-0 libsm6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmfashion
RUN conda clean --all
RUN git clone --recursive https://github.com/open-mmlab/mmfashion.git /mmfashion
WORKDIR /mmfashion
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .
```