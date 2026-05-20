---
layout: post
categories: [1. Computer Engineering, 1.3. DevOps & Infra, 1.3.2. OS, 1.3.2.1. Linux]
title: "Ubuntu server Kernel panic error"
tags: [Ubuntu Server, Kernel Panic]
---

### When

1. 윈도우 -> 우분투 
    * 파일 옮기면 서버 꺼짐

2. 우분투 -> 도커 내부 컨테이너
    * 파일 옮기면 에러 발생


### Kernel panic error

```ubuntu-server
Message from syslogd@maizer-workstation at Jul  7 20:50:26 ...
 kernel:[  489.471575] Kernel panic - not syncing: Fatal exception in interrupt
```

### How to solve

미해결