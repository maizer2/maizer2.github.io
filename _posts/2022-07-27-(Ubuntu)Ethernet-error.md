---
layout: post
title: "Ubuntu-server Ethernet Error"
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux]
---

### When

이더넷 포트가 3개 있는 워크스테이션에 Ubuntu-server을 설치 하였고

재부팅 시 "A start job is running for wait for network to be configured."가 발생하여 3분 이상 기다려야되는 상황이 발생하였다.

### How to solve

```ubuntu-server
sudo vi /etc/netplan/*.yaml

# In my case, I have two ethernet address
## Then I erase unused ethernet network setting.

```*.yaml
network:
  ethernets:
#    enp2s0:
#      nameservers:
#        addresses:
#        - *.*.*.*
#        - *.*.*.*
#        search: []
    enp3s0:
      addresses:
      - *.*.*.*/24
      gateway4: *.*.*.1
      nameservers:
        addresses:
        - *.*.*.*
        - *.*.*.*
        search: []
  version: 2
```

```ubuntu-server
sudo netplan apply
```