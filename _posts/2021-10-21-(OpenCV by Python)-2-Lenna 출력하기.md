---
layout: post
title: "(OpenCV by Python)2. Lenna 출력하기"
categories: "OpenCV"
tags: [AI, OpenCV]
---

*참고 : https://docs.opencv.org/master/db/deb/tutorial_display_image.html*



### Lenna 출력하기

---

폴더 내부에 lenna.bmp가 없으면 이미지가 불러와지지 않습니다.

![](https://raw.githubusercontent.com/maizer2/-Cpp-OpenCV/main/lenna.bmp)



##### 디렉터리 구조

```Directory
-Python-OpenCV
	-image
		-after_img
			-lenna.bmp	#해당파일은 프로그램 실행 후 생성됩니다.
		-lenna.bmp
	-Program
		-__init__.py
	-main.py
```



##### 코드

```python
#-Python-OpenCV/main.py

from Program.img.img_prac import img_show

path = "image/"

if __name__ == "__main__" :
    img_show(path)
```

```python
#-Python-OpenCV/Program/img/img_prac.py

import cv2 as cv
import sys

def img_show(path):
	img = cv.imread(path + "lenna.bmp")	#영상을 가져올 위치를 지정합니다.

    if img is None:
    	sys.exit("Could not read the image.")

	cv.imshow("Display window", img)	#cv.imread를 통해 영상을 띄울 창의 이름과 영상을 지정합니다.

	k = cv.waitKey(0)	#키 입력을 무한정 기다립니다. 

	if k == ord("s"):	#s키 입력시 종료
    	cv.imwrite(path + "after_img/" + "lenna.bmp", img) #s키 입력시 해당 폴더의 이름으로 저장합니다.
```

##### 출력

![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/Python/2021-10-21-(OpenCV by Python)-2-Lenna 출력하기/1.PNG)



### img_prac코드 분석

---



* #### cv.imread()

  imread 함수는 지정된 파일에서 이미지를 로드하고 반환합니다. 

  ```python
  cv.imread(filename[,flags]) -> retval
  ```

  * 이미지를 읽을 수 없는 경우(파일 누락, 부적절한 권한, 지원되지 않거나 잘못된 형식으로 인해) 함수는 빈 행렬( [Mat::data](https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a4d33bed1c850265370d2af0ff02e1564) ==NULL )을 반환합니다 .



* #### cv.imshow()

  지정된 창에 OpenGL 2D 텍스처를 표시합니다.

  ```python
  cv.imshow(winname, mat) -> None
  ```

  | Parameters |                              |
  | ---------- | ---------------------------- |
  | winname    | 영상을 출력할 대상 창 이름   |
  | mat        | 출력할 영상 데이터(Mat 객체) |



* #### cv.waitKey()

  키입력을 기다립니다.

  ```python
  cv.imwrite(filename, img[, params]) -> retval
  ```

  | Parameters |                                                              |
  | ---------- | ------------------------------------------------------------ |
  | delay      | Delay in milliseconds. 0 is the special value that means "forever". |



* ####  cv.imwrite

  이미지를 지정된 파일에 저장합니다.

  ```python
  cv.imwrite(filename, img[, params]) -> retval
  ```

  | Parameters |                                                              |
  | ---------- | ------------------------------------------------------------ |
  | filename   | Name of the file.                                            |
  | img        | ([Mat](https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html) or vector of [Mat](https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html)) Image or Images to be saved. |
  | params     | Format-specific parameters encoded as pairs (paramId_1, paramValue_1, paramId_2, paramValue_2, ... .) see [cv::ImwriteFlags](https://docs.opencv.org/master/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac) |



* ### 참조

  | 함수         | 설명                                                         |
  | ------------ | ------------------------------------------------------------ |
  | cv.imread()  | https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56 |
  | cv.imshow()  | https://docs.opencv.org/master/df/d24/group__highgui__opengl.html#gaae7e90aa3415c68dba22a5ff2cefc25d |
  | cv.waitKey() | https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7 |
  | cv.imwrite() | https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce |

  
