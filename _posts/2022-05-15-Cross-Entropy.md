---
layout: post
title: What is Cross Entropy?
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 2.2. Pobability and Statistics, a.b. Regression Problem]
---

### Cross Entropy is?
    
> 엔트로피<sup><a href="https://maizer2.github.io/1.%20computer%20engineering/2.%20mathematics/2022/05/13/Entropy-of-machine-learning.html#footnote_1_2">[Entropy]</a></sup>는 하나의 분포를 대상으로 하는 반면, 크로스-엔트로피는 두 분포 $P(x), Q(x)$를 대상으로 엔트로피를 측정해 두 분포 간의 차이를 계산합니다. 머신러닝<sup>GAN</sup>에서 크로스-엔트로피를 사용할 때는$P(x)$를 실제 모형의 분포, $Q(x)$를 추정 모형의 분포라고 설정합니다.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

> 교차 엔트로피<sup>cross entropy</sup>는 실제 결과가 도출될 우도와 우리가 생각하는 우도의 사이의 차이에 따른 결과의 불확실성에 대한 지표입니다. ... 교차 엔트로피를 두 확률분포의 차이라고 생각할 수 있습니다. 두 분포의 차이가 없는 만큼 교차 엔트로피도 낮아집니다. 정확히 일치한다면 교차 엔트로피는 0입니다.<sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup>

<br/>

### **Expressiion of Cross-Entropy**

$$CrossEntropy(P, Q) = -\sum P(x)logQ(x)$$

<center>$P$는 실제 확률의 분포, $Q$는 추정 모형의 분포이다.</center><br/>


### Example

![Cross-Entropy-Example](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.2.%20Artificial%20Intelligence/2022-05-15-Cross-Entropy/Cross-Entropy-Example.jpg)

$$CrossEntropy(P, Q) = -\sum P(x)logQ(x)$$
$$ \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;= -(1 \times log0.92 \; + \; 0 \times log 0.05 \; + \; 0 \times log 0.01)$$
$$ \;\;\;\;\;\;\;\;\;\;= 0.08 $$

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 7.5.4 엔트로피 156p, 선형대수와 통계학으로 배우는 머신러닝 with 파이썬, 장철원, 비제이퍼블릭

<a href="#footnote_2_2" name="footnote_2_1">2.</a> 부록. BCE 손실 246p, GAN 첫걸음, 타리크라시드 지음, 고락윤 옮김, 한빛미디어(주)