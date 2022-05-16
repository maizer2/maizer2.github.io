---
layout: post
title: "Entropy of Machine Learning"
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 2.2. Pobability and Statistics, 2.2. Pobability and Statistics]
---

### **Concept of Entropy**

Entropy는 노드에 서로 다른 데이터가 얼마나 섞여 있는지를 의미하는 impurity(불순도)를 측정한다.

Imputrity가 낮을수록 데이터가 섞여 있지 않다는 것을 의미한다.

> 엔트로피(entropy)는 정보 이론에서 사용하는 개념으로 확률 변수의 불확실성 정도를 측정하기 위해 사용합니다. ... 엔트로피는 하나의 분포를 대상으로 한다.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

>엔트로피<sup>entropy</sup>는 불확실성<sup>uncertainty</sup>을 설명하는 수학적 아이디어입니다. ... 동전은 던졌을 때 앞면이나 뒷면이나 같은 확률로 ... 이러한 경우 불확실성은 최대화되고, 엔트로피 역시 최대가 됩니다.<sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup>

<br/>

### **Expressiion of Entropy**

$$Entropy(P) = - \sum P(x)logP(x) = -\sum_{i=1}^{k}p(i|d)log_{2}(p(i|d)) = -E(logP(x))$$
$Entropy(P)$는  $H(P)$ 또는 $H(X)$라고 쓰기도 한다.

$$-\sum Pln(P)$$
위와 같이 간단하게 표현도 가능하다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 7.5.4 엔트로피 156p, 선형대수와 통계학으로 배우는 머신러닝 with 파이썬, 장철원, 비제이퍼블릭

<a href="#footnote_2_2" name="footnote_2_1">2.</a> 부록. BCE 손실 244p, GAN 첫걸음, 타리크라시드 지음, 고락윤 옮김, 한빛미디어(주)