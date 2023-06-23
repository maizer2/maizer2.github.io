---
layout: post 
title: "Manifold learning"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence]
---

![manifold](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.2.%20Artificial%20Intelligence/2023-06-23-Manifold.png)

<center>Manifold 시각화<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup></center><br/>

### Manifold란?

매니폴드(Manifold)는 간단히 말해서, 데이터가 분포하는 저차원의 구조를 나타냅니다.

일반적인 데이터는 고차원 공간에서 표현되지만, 데이터가 생성되거나 분포되는 과정에서 내재된 저차원의 구조가 존재할 수 있다.

이렇게 데이터가 내재된 저차원 구조에서의 데이터들을 표현한 것이 그림과 같이 manifold이다.


우리는 다양한 데이터 압축<sup>Encoder</sup>을 통해 Latent space에 매핑할 수 있다.

데이터가 Latent space에 매핑되어 있을 때, 매핑된 데이터들의 관계를 manifold라고 할 수 있다.

### Manifold learning이란?

여기서 Learning은 모델 학습의 learning이 아닌, 데이터 분석과 시각화가 가능함을 의미한다.

데이터가 저차원으로 압축될 경우 다음과 같은 장점이 있다.

1. 데이터 분포 시각화

2. 데이터 시각화를 통한 패턴 분석

3. Curse of Dimensionality(차원의 저주) 문제의 완화
    * 데이터 차원이 커질수록 데이터의 거리가 상당히 멀어져, 데이터 포인트간의 이웃성이 희박해진다. 따라서 특성 파악히 힘들고 패턴을 발견하기 어렵다.
    * Manifold learning은 이를 완화해준다.

---

##### 참고문헌


<a href="#footnote_1_2" name="footnote_1_1">1.</a> [인공지능 이론] Manifold Learning, roytravel.tistory, [https://roytravel.tistory.com/105](https://roytravel.tistory.com/105)