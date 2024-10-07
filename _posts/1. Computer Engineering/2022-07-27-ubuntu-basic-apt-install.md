---
layout: post
title: "Docker apt-get list"
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker]
---

```container
apt-get update && apt-get -y upgrade && apt-get install -y wget git vim build-essential net-tools

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sh Miniconda3-latest-Linux-x86_64.sh

rm Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc
```