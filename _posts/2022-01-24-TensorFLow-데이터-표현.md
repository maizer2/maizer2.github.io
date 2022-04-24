---
layout: post
title: "TensorFLow 데이터 표현"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.1. Machine Learning, 1.2.2. DeepLearning, TensorFlow]
---



### 텐서(Tensor)의 수학적 정의

여러 벡터 공간 및 그 쌍대 공간들을 일종의 '곱연산'을 사용해 복합적으로 연결시킨 구조.[1]

---

### 머신러닝에서의 텐서(Tensor)

* 데이터를 저장하는 다차원 배열(Numpy 배열)

* 텐서의 축 개수(N차원)를 랭크(rank)라고 부른다.

* 텐서는 float32, uint8, float64, char(가끔)이 될 수 있는데, 가변 길이의 문자열은 지원하지 않는다.

각 차원에 따라
* 0차원 텐서, 스칼라
* 1차원 텐서, 백터 
* 2차원 텐서, 행렬
* 3차원 텐서, 4차원 텐서
* 5차원 텐서 (동영상 데이터를 다룰 때만 쓰이고 잘 안쓰임)

---

#### 0차원 텐서, 스칼라

하나의 숫자만을 가지고 있는 텐서를 스칼라라고 한다.

스칼라는 0 rank이다.

#### 1차원 텐서, 벡터

숫자의 배열을 벡터라고 부른다.

벡터는 1 rank이다.

#### 2차원 텐서, 행렬

벡터의 배열을 행렬이라고 부른다.

행렬은 2 rank이다.

#### 3차원 텐서, 4차원 텐서, 5차원 텐서 ... N차원 텐서

N-1차원 텐서를 이어 부치면 N차원 텐서가 된다.

딥러닝에서는 주로 0 ~ 4차원 텐서까지 사용되지만 동영상을 다룰 때 5차원 텐서가 사용 되기도 한다.



---
##### 참고문헌

1) 텐서, 나무위키, 2022-01-24 방문, https://namu.wiki/w/텐서

2) 프랑소와 숄레 (Francois Chollet), 케라스 창시자에게 배우는 딥러닝, 박해선, 오라일리, (주)도서출판 길벗(2021년 초판 8쇄 발행), 61p

3) 텐서 플로우 블로그, 박해선, https://tensorflow.blog/케라스-딥러닝/2-2-신경망을-위한-데이터-표현/

4) 텐서플로우(TensorFlow) 텐서 기본 개념 - Tensor란 무엇인가? (Rank, Shapes, Types), 테크 스케치, 이즈군(complusblog), 2018-03-27 작성, 2022-01-24 방문, https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=complusblog&logNo=221237818389

5) 텐서와 상대론 (Tensor and Relativity) - 0. 텐서 (Tensor) 란?, kipid's blog, 2019-02-19 작성, 2022-01-24 방문, https://kipid.tistory.com/entry/Tensor
