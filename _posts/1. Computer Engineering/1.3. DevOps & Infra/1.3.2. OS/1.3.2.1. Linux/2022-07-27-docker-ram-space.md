---
layout: post
categories: [1. Computer Engineering, 1.3. DevOps & Infra, 1.3.2. OS, 1.3.2.1. Linux]
title: "Docker --shm-size"
tags: [1.3.2.1. Linux, 1.3.2. OS, 1.3.3.1. Docker, 1.3.3. Container, 1.3. DevOps & Infra]
---

RuntimeError: DataLoader worker (pid 13881) is killed by signal: Bus error

```ubuntu-server
docker run -i -t --gpus all --shm-size=32gb --name torch040 ubuntu:18.04
```