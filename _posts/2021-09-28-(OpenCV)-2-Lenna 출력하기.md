---
layout: post
title: "(OpenCV)2. Lenna 출력하기"
categories: C++
tags: [AI, OpenCV]
---



### Lenna 출력하기

---

폴더 내부에 lenna.bmp가 없으면 이미지가 불러와지지 않습니다.

![](https://raw.githubusercontent.com/maizer2/-Cpp-OpenCV/main/lenna.bmp)

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

![](https://raw.githubusercontent.com/maizer2/gitblog_img/img/C++/2021-09-28-(OpenCV)-2-Lenna 출력하기/1.PNG)



### main코드 분석

---

위 main코드에는 기본적인 OpenCV 함수를 볼 수 있다.

imread(), nameWindow(), imshow(), waitKey()



* #### cv::imread()

  불러온 이미지 데이터를 Mat 객체로 변환하여 반환합니다.

  ```C++
  Mat imread(const String$ filename, int flags = IMREAD_COLOR);
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

* #### cv::imwrite()

  Mat 객체에 저장되어 있는 영상 데이터를 파일로 저장한다.

  ``` C++
  bool imwrite(const String$ filename, InputArray img, const std::vector<int>$ params = std::vector<int>());
  ```

  * filename : 저장할 영상 파일 이름

  * img          : 저장할 영상 데이터(Mat 객체)

  * params   : 저장할 영상 파일 형식에 의존적이 파라미터(플래그 & 값) 쌍 

    ​				   (paramId_1, paramValue_1, paramId_2, paramValue_2, ... )

  * 반환값     : 정사적으로 저장하면 true, 실패하며 false를 반환합니다.

