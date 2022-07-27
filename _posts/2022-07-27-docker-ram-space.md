---
layout: post
title: "Docker --shm-size"
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker]
---

RuntimeError: DataLoader worker (pid 13881) is killed by signal: Bus error

```ubuntu-server
docker run -i -t --gpus all --shm-size=32gb --name torch040 ubuntu:18.04
```