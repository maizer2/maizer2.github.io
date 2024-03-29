﻿---
layout: post
title: "(Hands-On Machine Learning 2) 1. 한눈에 보는 머신러닝"
categories: [5. BookReview]
tag: [1.2. Artificial Intelligence, 1.2.1. Machine Learning]
---
## [←  이전 글로](https://maizer2.github.io/5.%20bookreview/2022/01/13/(Hands-On-Machine-Learning-2)-0.-서론.html) 　 [다음 글로 →](https://maizer2.github.io/5.%20bookreview/2022/02/07/(Hands-On-Machine-Learning-2)-2.-머신러닝-프로젝트-처음부터-끝까지.html)


### 들어가기 전에

1부는 머신러닝의 기초 개념과 용어를 소개해주는 장입니다.

수학적 지식도 부족하고 머신러닝 지식도 부족하다보니 1부를 통해 정말 많은걸 배웠습니다.

---

### 머신러닝 시스템의 종류 [[35p](https://tensorflow.blog/핸즈온-머신러닝-1장-2장/1-3-머신러닝-시스템의-종류/)]

* [지도 학습](https://maizer2.github.io/1.%20computer%20engineering/2022/01/24/지도-학습.html), [비지도](https://maizer2.github.io/1.%20computer%20engineering/2022/02/01/비지도-학습.html), [준지도](https://maizer2.github.io/1.%20computer%20engineering/2022/02/04/준지도-학습.html), [강화 학습](https://maizer2.github.io/1.%20computer%20engineering/2022/02/04/강화-학습.html)
  * ***사람의 감독*** 유/무
* [온라인 학습](https://maizer2.github.io/1.%20computer%20engineering/2022/01/14/인공지능에서-입력-데이터-스트림이란.html), [배치 학습](https://maizer2.github.io/1.%20computer%20engineering/2022/02/04/배치-학습.html)
  * ***실시간 학습*** 유/무
* [사례 기반 학습](https://maizer2.github.io/1.%20computer%20engineering/2022/02/05/사례-기반-학습.html), [모델 기반 학습](https://maizer2.github.io/1.%20computer%20engineering/2022/02/05/모델-기반-학습.html)
  * 단순하게 ***알고 있는 데이터 포인트와 새 데이터 포인트를 비교***
  * 훈련 데이터셋에서 ***패턴을 발견하여 예측 모델을 생성***

각자 학습들은 서로 배타적이지 않고 원하는 대로 연결할 수 있다.

---

### 머신러닝의 주요 도전 과제[[53p](https://tensorflow.blog/핸즈온-머신러닝-1장-2장/1-4-머신러닝의-주요-도전-과제/)]

나쁜 알고리즘과 나쁜 데이터에 관하여

#### 나쁜 데이터

* 충분하지 않은 데이터
  * 샘플이 작을경우 ***샘플링 잡음***
  * 샘플이 많을경우 ***샘플링 편향***
* 대표성 없는 훈련 데이터
  * 정확한 예측이 어렵다
  * ***샘플링 잡음*** -> 우연에 의한 대표성 없는 데이터
  * ***샘플링 편향*** -> 큰 샘플의 표본 추출이 잘못될 경우
* 품질이 낮은 데이터
* 관련 없는 특성
* ***과대 적합***
  * 인간 일반화의 기계 버전
  * 훈련 데이터에 너무 잘 맞으면 일반성이 떨어진다.
  * 잡음이 많거나 데이터 셋이 적아으면 잡음이 섞인 패턴을 감지하게 된다.
    * 우연히 발견된 패턴을 모델은 진짜인지 잡음이 섞인지 구분할 수 없다.
  * 훈련 데이터에 있는 잡음에 비해 모델이 너무 복잡할 때 발생한다.
    * 파라미터 수가 적은 모델을 선택, 특성 수를 줄인다, 모델에 제약(규약)을 가한다.
      * 규제의 양은 [하이퍼 파라미터](https://maizer2.github.io/1.%20computer%20engineering/2022/01/15/인공지능에서-모델-파라미터란.html) 가 결정한다.
    * 훈련 데이터를 추가한다.
    * 잡음을 줄인다.
* 과소적합

### 테스트와 검증

데이터 세트를 훈련과 테스트 세트로 나눈다. (보통 8:2 의 비율로 데이터를 나눈다, 하지만 데이터가 많을수록 테스트 세트 비율이 적어진다.)

테스트 세트를 통해 일반화 오차가 낮은 [하이퍼 파라미터](https://maizer2.github.io/1.%20computer%20engineering/2022/01/15/인공지능에서-모델-파라미터란.html) 값을 찾을 때는 [홀드아웃 검증](https://maizer2.github.io/1.%20computer%20engineering/2022/02/06/홀드아웃-검증.html) 을 통해 해결 할 수 있다.

### 데이터 불일치

데이터 세트가 기대하는 데이터를 잘 대표해야한다. 그렇지 않으면 실망스러운 모델의 성능이 될 것이다.

#### 훈련-개발 세트

웹에서 얻은 대량의 데이터는 훈련 데이터로서 쓰기 힘든 경우가 대부분이다.

이를 해결하기 위해서는 데이터 라벨링을 통해 재사용할 수 있다.

---

##### 참고문헌

1) 오렐리앙 제롱 (Aurelien Geron), Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow 핸즈온 머신러닝 2판, 박해선, 오라일리, 한빛미디어(주)(2021년 5판)

2) "텐서 플로우 블로그" 박해선 "https://tensorflow.blog/핸즈온-머신러닝-1장-2장/"

3) "박해선 유튜브" 박해선 "https://youtube.com/playlist?list=PLJN246lAkhQjX3LOdLVnfdFaCbGouEBeb"
