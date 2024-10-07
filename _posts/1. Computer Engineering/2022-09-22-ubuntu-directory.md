---
layout: post
title: "ubuntu count directory files"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

현재 위치에서 파일의 개수 세기
```
ls -l | grep ^- | wc -l
```
현재 디렉토리의 하위 파일 개수 세기
```
find . -type f | wc -l
```

##### Reference

1. https://lee-mandu.tistory.com/420