---
layout: post
title: "Forward Propagation"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning]
---

### Introduction

이 과정은 신경망인 [Perceptron](https://maizer2.github.io/1.%20computer%20engineering/2022/05/18/single-layer-perseptron.html)을 참고하면 더 이해가 잘된다.

> 순전파<sup>forward propagation</sup>란 입력 데이터에 대해 신경망 구조를 따라가면서 현재의 파라미터값들을 이용해 손실 함숫값을 계산하는 과정을 말한다.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>


1. $W$값들을 통해 $\hat{y}$ 값을 얻는다.
2. Loss Function을 통해 실제 레이블 $y$와 예측 레이블 $\hat{y}$값을 비교한다.

<br/>

#### First Step of Forward propagation

<center>$n$ is number of True labels</center>

$$ \hat{y} = \sum_{i=1}^{n} W_{i} \cdot x_{i}  $$

#### Second Step of Forward propagation

대표적인 Loss Function : [MSE](https://maizer2.github.io/1.%20computer%20engineering/2022/04/08/제곱근-오차.html), [Cross Entropy](https://maizer2.github.io/1.%20computer%20engineering/2.%20mathematics/2022/05/15/Cross-Entropy.html), $\cdots$

<center>$m$ is number of Features</center>

$$MSE(\hat{y}) = \frac{1}{m}\sum_{i=1}^{m}(y - \hat{y}_{i})^{2}$$

두 label값의 차이를 구하는데, $W$가 잘 맞을 수록 두 값의 차이가 작아진다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 1.6 경사하강법과 역전파 59p, 윤덕호, 파이썬 날코딩으로 알고 짜는 딥러닝,  한빛미디어(주)