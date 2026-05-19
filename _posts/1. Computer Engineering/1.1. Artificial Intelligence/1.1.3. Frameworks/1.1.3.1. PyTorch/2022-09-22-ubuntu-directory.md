---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.3. Frameworks, 1.1.3.1. PyTorch]
title: "ubuntu count directory files"
tags: [Ubuntu, File Count]
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