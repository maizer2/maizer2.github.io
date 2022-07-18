---
layout: post
title: "Ubuntu install guide"
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux]
---

|Os|Os version|CPU|GPU|
|--|----------|---|---|
|Ubuntu-server|20.04.04|AMD Ryzen Threadripper PRO 5955WX 16-Cores|Nvidia RTX A6000|

### Install Ubuntu-Server OS on Server-Computer

1. GNU GRUB
    * Install Ubuntu Server
2. Select Language
    * English(US)
3. Select Keyboard configuration
    * English(US)
4. Network connections
    * Edit IPv4
    * IPv4 Method:
        * Manual
        * Subnet : *.*.*.0/24
        * Address : *.*.*.*
        * Gateway : *.*.*.1
        * Name servers : *.*.*.*, *.*.*.*
        * search domains : empty
        * save
5. Proxy address
    * Pass(Done)
6. Mirror address
    * Pass(Done)
7. Select Disk and Disk Size
    * Use an entire disk
        * Done
    * USED DEVICES
        * ubuntu-lv ... size -> Entire size
        * Done
8. Profile setup
9. Install OpenSSH server
    * Check
    * Done
10. Featured Server Snaps
    * Pass(Done)

```ubuntu-server
apt-get update && apt-get upgrade && sudo apt-get dist-upgrade && sudo apt-get install -y nvidia-driver-515-server && apt-get update && apt-get install -y dialog language-pack-en && export LANGUAGE=en_US && export LANG=en_US.UTF-8 && export LC_ALL=en_US.UTF-8 && sudo update-locale && sudo vi /etc/default/locale && sudo apt-get update && sudo apt-get install -y \ ca-certificates \ curl \ gnupg \ lsb-release && sudo mkdir -p /etc/apt/keyrings &&  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg && echo \ "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \ $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin && distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \ && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \ && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list && sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker 

sudo vim /etc/modprobe.d/blacklist.conf

blacklist nouveau
options nouveau modeset=0

:wq

sudo update-initramfs -u

LANG="en_US.UTF-8"
LANGUAGE="en_US:en"
LC_ALL="en_US.UTF-8"

sudo reboot

lsmod |grep nouveau && sudo docker run --gpus all --name torch --ip 3979:3979 ubuntu:20.04
```

```docker
apt-get update && apt-get upgrade && apt-get install -y wget \ net-tools \ git \ vim && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# yes

source ~/.bashrc && conda create --name torch python=3.8 pytorch torchvision cudatoolkit -c pytorch -c conda-forge