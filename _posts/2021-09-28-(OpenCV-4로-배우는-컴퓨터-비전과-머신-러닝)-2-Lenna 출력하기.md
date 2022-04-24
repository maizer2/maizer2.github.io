---
layout: post
title: "(OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝)2. Lenna 출력하기"
categories: [5. BookReview]
tags: [1.1 Programming, 1.1.2. C++, OpenCV]
---



### Lenna 출력하기

---

폴더 내부에 lenna.bmp가 없으면 이미지가 불러와지지 않습니다.

![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/BookReview/2021-09-28-(OpenCV-by-C++)-2-Lenna-출력하기/lenna.bmp)

입력

```C++
#include "opencv2/opencv.hpp"
#include <iostream>

int main() {
	std::cout << "Hello OpenCV" << CV_VERSION << std::endl;
	
	cv::Mat img;
	img = cv::imread("lenna.bmp");

	if (img.empty()) {
		std::cerr << "Image load failed! " << std::endl;
		return -1;
	}

	cv::namedWindow("lenna");
	cv::imshow("lenna", img);

	cv::waitKey();
	return 0;
}
```

출력

![](https://raw.githubusercontent.com/maizer2/gitblog_img/master/img/C++/2021-09-28-(OpenCV by C++)-2-Lenna 출력하기/1.PNG)



### main코드 분석

---

위 main코드에는 기본적인 OpenCV 함수를 볼 수 있다.

imread(), nameWindow(), imshow(), waitKey()



* #### cv::imread()

  불러온 이미지 데이터를 Mat 객체로 변환하여 반환합니다.

  ```C++
  Mat imread(const String& filename, int flags = IMREAD_COLOR);
  ```

  * filename : 불러올 영상 이름

  * flags        :  영상 파일 불러오기 옵션 플래그, ImreadModes 열거형 상수를 지정한다. 

    * 기본값으로 IMREAD_COLOR가 지정되어 있다. (3채널 컬러 영상으로 반환)

      | ImreadModes 열거형 상수   | 설명                                                      |
      | ------------------------- | :-------------------------------------------------------- |
      | IMREAD_UNCHANGED          | 입력 파일에 지정된 그대로의 컬러 속성을 사용합니다.       |
      | IMREAD_CRAYSCALE          | 1채널 그레이스케일 영상으로 변환하여 불러옵니다.          |
      | IMREAD_COLOR              | 3채널 BGR 컬러 영상으로 변환하여 불러옵니다.              |
      | IMREAD_REDUCED_GRAYSCALE  | 크기를 1/2로 줄인 1채널 그레이스케일 영상으로 변환합니다. |
      | IMREAD_REDUCED_COLOR_2    | 크기를 1/2로 줄인 3채널 BGR 영상으로 변환합니다.          |
      | IMREAD_IGNORE_ORIENTATION | EXIF에 저장된 방향 정보를 사용하지 않습니다.              |

  * return     : 불러온 영상 데이터(Mat 객체), 파일이 없거나 잘못된 확장자일 경우 빈 객체를 반환

  

  * ##### Mat::empty()

    Mat 객체가 제대로 생성되었는지를 확인한다.
    
    ```c++
    bool Mat::empty() const
    ```
    
    * 반환값 : 행렬의 rows 또는 cols 멤버 변수가 0이거나, 또는 data 멤버 변수가 NULL이면 true를 반환
    
      

* #### cv::imwrite()

  Mat 객체에 저장되어 있는 영상 데이터를 파일로 저장한다.

  ``` C++
  bool imwrite(const String$ filename, InputArray img, const std::vector<int>& params = std::vector<int>());
  ```

  * filename : 저장할 영상 파일 이름

  * img          : 저장할 영상 데이터(Mat 객체)

  * params   : 저장할 영상 파일 형식에 의존적이 파라미터(플래그 & 값) 쌍 

    ​				   (paramId_1, paramValue_1, paramId_2, paramValue_2, ... )

  * 반환값     : 정사적으로 저장하면 true, 실패하며 false를 반환합니다.
  
    
  
* #### cv::namedWindow()

  ```C++
  void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);
  ```

  * winname : 영상 출력 창 상단에 출력되는 창 고유 이름, 이 문자열로 창을 구분한다.

  * flags         : 생성되는 창의 속성을 지정하는 플래그, WindowFlags 열거형 상수를 지정합니다.

    | WindowFlags 열거형 상수  | 설명                                                    |
    | ------------------------ | ------------------------------------------------------- |
    | WINDOW_NORMAL            | 출력 창의 크기에 맞춰 출력, 임의 변경 가능              |
    | WINDOW_AUTOSIZE(Defualt) | 출력 영상 크기에 맞춰 자동 변경됩니다. 임의 변경 불가능 |
    | WINDOW_OPENGL            | OpenGL을 지원합니다.                                    |

    

  * 윈도우에서 창을 구분하기 위해서는 핸들이라는 숫자 값을 사용하지만, 

    OpenCV에서는 각가의 창에 고유한 문자열을 부여하여 각각의 창을 구분합니다.



* #### cv::destroyWindow(), cv::destroyAllWindows()

  프로그램이 동작 중에 창을 닫고 싶을 때

  ```c++
  void destroyWindow(const String& winname);
  void destroyAllWindows();
  ```

  * winname : 소멸시킬 창 이름



* #### cv::moveWindow()

  ```c++
  void moveWindow(const String& winname, int x, int y);
  ```

  * winname : 위치를 이동할 창 이름
  * x                : 창이 이동할 위치의 x 좌표
  * y                : 창이 이동할 위치의 y 좌표



* #### cv::resizeWindow()

  ```c++
  void resizeWindow(const String& winname, int width, int height);
  ```

  * winname : 크기를 변경할 창 이름
  * width        : 창의 가로 크기
  * height       :  창의 세로 크기

* #### cv::imshow();

  Mat 클래스 객체에 저장된 영상 데이터를 화면에 출력하는 함수

  ```c++
  void imshow(const String& winname, InputArray mat);
  ```

  * winname : 영상을 출력할 대상 창 이름
  * mat          : 출력할 영상 데이터(Mat 객체)



* #### cv::waitKey()

  사용자로부터 키보드 입력을 받는 용도로 사용

  ```
  int waitKey(int delay = 0);
  ```

  * delay   : 키 입력을 기다릴 시간(밀리초 단위). delay <= 0 이면 무한히 기다립니다.
  * 반환값 : 입력한 키 값. 지정한 시간이 지나면 -1을 반환

  

