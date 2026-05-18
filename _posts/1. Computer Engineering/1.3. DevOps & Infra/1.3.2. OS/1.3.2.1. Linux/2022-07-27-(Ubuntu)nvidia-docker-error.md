---
layout: post
categories: [1. Computer Engineering, 1.3. DevOps & Infra, 1.3.2. OS, 1.3.2.1. Linux]
title: "Nvidia-Docker Error"
tags: [1.3.2.1. Linux, 1.3.2. OS, 1.3.3.1. Docker, 1.3.3. Container, 1.3. DevOps & Infra]
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