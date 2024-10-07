---
layout: post
title: "Nvidia-Docker Error"
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker]
---

### When

실수로 nvidia-docker를 삭제하게 되었고 재설치하는 과정에서 apt-get update가 되지 않게 되었다.

```ubuntu-server
maizer@maizerworkstation:~$ sudo apt-get update
E: Conflicting values set for option Signed-By regarding source https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/ /: /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg !=
E: The list of sources could not be read.
```

### How to solve

```ubuntu-server
cd /etc/apt/sources.list.d
sudo rm -rf nvidia-container-toolkit.list && sudo rm -rf nvidia-docker.list
sudo apt-get update
```

and then Setup the package repository and the GPG key

[Setting-up-nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)