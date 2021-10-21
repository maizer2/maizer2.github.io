---
layout: post
title: "(OpenCV by Python)2. Lenna 출력하기"
categories: Python
tags: [AI, OpenCV]
---

*참고 https://docs.opencv.org/master/db/deb/tutorial_display_image.html*

---



### Lenna 출력하기

---

폴더 내부에 lenna.bmp가 없으면 이미지가 불러와지지 않습니다.

![](https://raw.githubusercontent.com/maizer2/-Cpp-OpenCV/main/lenna.bmp)



디렉터리 구조

-Python-OpenCV

​	-image

​		-lenna.bmp

​	-Program

​		-__init__.py

​	-main.py

입력

```python
import cv2 as cv
import sys

path = "image/"


img = cv.imread(path + "lenna.bmp")

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)

k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("starry_night.png", img)
```

출력

![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/Python/2021-10-21-(OpenCV by Python)-2-Lenna 출력하기/1.PNG)

