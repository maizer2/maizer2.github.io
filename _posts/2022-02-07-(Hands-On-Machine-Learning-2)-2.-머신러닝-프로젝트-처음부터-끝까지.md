---
layout: post
title: "(Hands-On Machine Learning 2) 2. 머신러닝 프로젝트 처음부터 끝까지"
categories: "BookReview"
tags:  [AI, Machine Lerning, TensorFlow, Keras, Scikit-Learn]
---

## [←  이전 글로](https://maizer2.github.io/bookreview/2022/01/13/(Hands-On-Machine-Learning-2)-1.-한눈에-보는-머신러닝.html) 　 [다음 글로 →](https://maizer2.github.io/bookreview/2022/02/00/(Hands-On-Machine-Learning-2)-3.-분류.html)


### 들어가기 전에

2장은 머신러닝의 예제 프로젝트를 처음부터 끝까지 진행합니다.

내용을 자세히 이해하려니 어려움이 있었지만, 프로젝트의 큰 흐름을 이해할 수 있었습니다.

해당 장에서는 캘리포니아 인구조사 데이터를 사용하여 캘리포니아 주택 가격 모델을 만듭니다.

---

### 문제 정의

금전적 수익을 위해 제작되는 모델임으로, ***비즈니스 목적***을 정확히 파악해야한다.

#### 부동산 투자를 위한 머신러닝 파이프라인 [[2](https://tensorflow.blog/핸즈온-머신러닝-1장-2장/2-2-큰-그림-보기)]

---

![](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-31-e1848be185a9e1848ce185a5e186ab-10-42-47.png?w=625&h=248)

[파이프 라인이란](https://maizer2.github.io/용어_인공지능/2021/11/02/머신러닝에서-파이프라인이란.html)

#### 학습 방법 선택

지도, 비지도, 강화, 분류, 회귀, 배치, 온라인 등등 다양한 선택지가 있다.

이를 선택하기 위해서 현재 가지고 있는 데이터를 잘 파악하는 것이 중요하다.

#### 성능 측정 지표 선택

[회귀 문제]() 의 전형적인 성능 지표는 [평균 제곱근 오차](https://maizer2.github.io/용어_인공지능/2022/02/08/평균-제곱근-오차.html), [평균 절대 오차](https://maizer2.github.io/용어_인공지능/2022/02/11/평균-절대-오차.html) 등이 있다.


### 데이터 가져오기

[텐서 플로우 블로그](https://tensorflow.blog/핸즈온-머신러닝-1장-2장/2-3-데이터-가져오기/) 에서 데이터를 가져오는 과정을 참고 할 수 있습니다.



---

##### 참고문헌

1) 오렐리앙 제롱 (Aurelien Geron), Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow 핸즈온 머신러닝 2판, 박해선, 오라일리, 한빛미디어(주)(2021년 5판)

2) "텐서 플로우 블로그" 박해선 "https://tensorflow.blog/핸즈온-머신러닝-1장-2장/2-머신러닝-프로젝트-처음부터-끝까지/"

3) "박해선 유튜브" 박해선 "https://youtube.com/playlist?list=PLJN246lAkhQjX3LOdLVnfdFaCbGouEBeb"
