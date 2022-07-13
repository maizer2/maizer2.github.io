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