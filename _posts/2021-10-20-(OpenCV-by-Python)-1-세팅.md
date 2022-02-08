---
layout: post
title: "(OpenCV by Python)1. Setting"
categories: "OpenCV"
tags: [Python, AI, OpenCV]
---

사전 준비물 : Anaconda3, Python, pip, PyCharm



### 가상환경 생성

```cmd
conda create --name opencv python=3.7
conda activate opencv
```



### OpenCV 설치

```cmd
python -m pip install opencv-python
```



설치 확인

```Python
import cv2
print(cv2.__version__)
```

출력

```
4.5.3
```



### PyCharm 설정

![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/BookReview/2021-10-20-(OpenCV-by-Python)-1-세팅/1.PNG)

File -> Setting -> Project: "프로젝트 이름" -> Python interpreter -> 우측 톱니 선택 -> Add



![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/BookReview/2021-10-20-(OpenCV-by-Python)-1-세팅/2.PNG)

Conda Environment -> Existing environment -> "아나콘다 설치 폴더/envs/opencv/python.exe" 선택



![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/BookReview/2021-10-20-(OpenCV-by-Python)-1-세팅/3.PNG)

패키지에 opencv-python 확인후 적용



![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/BookReview/2021-10-20-(OpenCV-by-Python)-1-세팅/4.PNG)

파이참 우측 하단에 Python3.7(opencv) 적용확인



![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/BookReview/2021-10-20-(OpenCV-by-Python)-1-세팅/5.PNG)

버전 출력 확인



