---
layout: post 
title: "Perceptron"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence]
---

### Structure of Perceptron

![perceptron](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.2.%20Artificial%20Intelligence/2022-05-18-perseptron/Perceptron.JPG)

|variable|mean|
|--------|----|
|x|Feature<sup>특징</sup>|
|b|Bias<sup>편향</sup>|
|w|Wieght<sup>가중치</sup>|
|$\sum$|$\sum^{n}w\cdot x + b$|
|$f$|$f(\sum^{n}w\cdot x + b)$
|$\hat{y}$|Predict label<sup>예측 레이블|

---

### Why use $f()$

non-linear Function<sup>비선형 함수</sup> $f()$는 Activation Fucntion<sup>활성화 함수</sup>라고도 불린다.

뉴런 세포체가 단순히 입력 전기 신호를 합한 값을 출력으로 삼지 않고 나름의 처리를 거쳐 출력값을 결정하는 것처럼 perceptron의 activation function도 $\sum$에서 구한 값에 적절한 non-linear function을 적용하여 최종 출력을 결정한다.