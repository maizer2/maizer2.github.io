---
layout: post
title: "(OpenCV by Python)3. 동영상 다루기"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, a.a. OpenCV]
---

## [←  이전 글로](https://maizer2.github.io/1.%20computer%20engineering/2021/10/21/(OpenCV-by-Python)-2-Lenna-출력하기.html) 　 [다음 글로 →](https://maizer2.github.io/1.%20computer%20engineering/2022/00/00/(OpenCV-by-Python)-4-미정.html)

*참고 : https://docs.opencv.org/master/db/deb/tutorial_display_image.html*

---



### 실시간 동영상 처리하기

---



#### 코드

```python
#-Python-OpenCV/Program/video/video_prac

import numpy as np  # cv2 설치하면 numpy는 같이 설치됨
import cv2 as cv

def video_capture(path):
    cap = cv.VideoCapture(0)
	if not cap.isOpened():
    	print("Cannot open camera")
    	exit()
    	
	while True:
    	# Capture frame-by-frame
    	ret, frame = cap.read()
    	
		# if frame is read correctly ret is True
   		if not ret:
        	print("Can't receive frame (stream end?). Exiting ...")
        	break
    
    	# Our operations on the frame come here
    	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    	# Display the resulting frame
    	cv.imshow('frame', gray)
    	if cv.waitKey(1) == ord('q'):
        	break
        
	# When everything done, release the capture
	cap.release()
	cv.destroyAllWindows()
```

