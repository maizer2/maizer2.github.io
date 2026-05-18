---
layout: post
categories: [1. Computer Engineering, 1.3. DevOps & Infra, 1.3.2. OS, 1.3.2.1. Linux]
title: "Docker apt-get list"
tags: [1.3.2.1. Linux, 1.3.2. OS, 1.3.3.1. Docker, 1.3.3. Container, 1.3. DevOps & Infra]
---

```container
apt-get update && apt-get -y upgrade && apt-get install -y wget git vim build-essential net-tools

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sh Miniconda3-latest-Linux-x86_64.sh

rm Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc
```