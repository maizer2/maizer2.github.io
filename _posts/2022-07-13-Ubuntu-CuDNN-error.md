---
layout: post
title: "The public CUDA GPG key does not appear to be installed."
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux]
---

### Error messege

```ubuntu-server

$ sudo dpkg -i cudnn-local-repo-ubuntu2004-8.4.1.50_1.0-1_amd64.deb

(Reading database ... 142484 files and directories currently installed.)
Preparing to unpack cudnn-local-repo-ubuntu2004-8.4.1.50_1.0-1_amd64.deb ...
Unpacking cudnn-local-repo-ubuntu2004-8.4.1.50 (1.0-1) over (1.0-1) ...
Setting up cudnn-local-repo-ubuntu2004-8.4.1.50 (1.0-1) ...

The public CUDA GPG key does not appear to be installed.
To install the key, run this command:
sudo cp /var/cudnn-local-repo-ubuntu2004-8.4.1.50/cudnn-local-E3EC4A60-keyring.gpg /usr/share/keyrings/
```

Error operate When install CuDNN in Ubuntu server 20.04 version

### Analyne error message

```
The public CUDA GPG key does not appear to be installed
To install the key, run this command:
sudo cp /var/cudnn-local-repo-ubuntu2004-8.4.1.50/cudnn-local-E3EC4A60-keyring.gpg /usr/share/keyrings/
```

Anticipate solution is create the CUDA GPG key in "/var/cudnn-local-repo-ubuntu.../....gpg" and copy to "/usr/share/keyrings/"

### Solution

Check Offical Nvidia CuDNN Installation Guide

[https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb)

```ubuntu-server
# Download the Debian local repository installation package. Before issuing the following commands, you must replace X.Y and 8.x.x.x with your specific CUDA and cuDNN versions.

# Procedure

# 1. Navigate to your <cudnnpath> directory containing the cuDNN Debian local installer file.

# 2. Enable the local repository.
sudo dpkg -i cudnn-local-repo-${OS}-8.x.x.x_1.0-1_amd64.deb
# Or
sudo dpkg -i cudnn-local-repo-${OS}-8.x.x.x_1.0-1_arm64.deb

# 3. Import the CUDA GPG key.
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/

# 4. Refresh the repository metadata.
sudo apt-get update

# 5. Install the runtime library.
sudo apt-get install libcudnn8=8.x.x.x-1+cudaX.Y

# 6. Install the developer library.
sudo apt-get install libcudnn8-dev=8.x.x.x-1+cudaX.Y

# 7. Install the code samples and the cuDNN library documentation.
sudo apt-get install libcudnn8-samples=8.x.x.x-1+cudaX.Y
```

### Verifying the install on linux

[https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify)

```ubuntu-server
# 2.4. Verifying the Install on Linux
# To verify that cuDNN is installed and is running properly, compile the mnistCUDNN sample located in the /usr/src/cudnn_samples_v8 directory in the Debian file.

# Procedure
# Copy the cuDNN samples to a writable path.
cp -r /usr/src/cudnn_samples_v8/ $HOME

# Go to the writable path.
cd  $HOME/cudnn_samples_v8/mnistCUDNN

# Compile the mnistCUDNN sample.
make clean && make

## 
# Run the mnistCUDNN sample.
./mnistCUDNN

# If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following:
# Test passed!
```

**If the following compilation errors are reported when “sudo make” is executed: fatal error: FreeImage. H**

[https://developpaper.com/ubuntu-20-04-system-3090-graphics-card-steps-of-installing-driver-cuda-and-cudnn/](https://developpaper.com/ubuntu-20-04-system-3090-graphics-card-steps-of-installing-driver-cuda-and-cudnn/)

```
mnistCUDNN  sudo make
CUDA_VERSION is 11010
Linking agains cublasLt = true
CUDA VERSION: 11010
TARGET ARCH: x86_64
HOST_ARCH: x86_64
TARGET OS: linux
SMS: 35 50 53 60 61 62 70 72 75 80 86
test.c:1:10: fatal error: FreeImage.h: No such file or directory
    1 | #include "FreeImage.h"
      |          ^~~~~~~~~~~~~
compilation terminated.
```
Then execute:sudo **apt-get install libfreeimage3 libfreeimage-dev**, and then revalidate.
