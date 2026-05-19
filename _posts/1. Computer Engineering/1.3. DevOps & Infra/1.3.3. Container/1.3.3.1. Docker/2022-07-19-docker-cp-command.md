---
layout: post
categories: [1. Computer Engineering, 1.3. DevOps & Infra, 1.3.3. Container, 1.3.3.1. Docker]
title: "Docker cp command"
tags: [Docker, cp]
---

### Move host file to container

docker cp [host file location] [container name]:[container location]

### Move container file to host

docker cp [container name]:[container location] [host location]