---
layout: post 
title: "Single Layer Perceptron"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2.1. ANN]
---

### Structure of Single Layer Perceptron

![single layer perceptron](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.2.%20Artificial%20Intelligence/2022-05-18-single-layer-perseptron/Single-Layer-Perceptron.JPG)

|variable|mean|
|--------|----|
|x|Feature<sup>특징</sup>|
|b|Bias<sup>편향</sup>|
|w|Wieght<sup>가중치</sup>|
|$\sum$|$\sum^{n}w\cdot x + b$|
|$f$|$f(\sum^{n}w\cdot x + b)$
|$\hat{y}$|Predict label<sup>예측 레이블|

---

### Characteristics of Single Layer Perceptron

Single Layer Perceptron은 Perceptron이 2개 이상이다.

Perceptron의 개수에 따라 출력값 $\hat{y}$의 개수가 결정된다.

이 구조는 Linear Algebra에서 Inear Product를 통해 계산할 수 있다.

![single layer perceptron](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.2.%20Artificial%20Intelligence/2022-05-18-single-layer-perseptron/Perceptron-Inear-Product.JPG)

$$ x \cdot W = y $$

값들을 단순한 반복문으로 계산할 수 있지만, Inear Product를 사용하는 이유는 GPU의 병렬연산을 사용하여 빠르게 계산할 수 있기 때문이다.

병렬연산은 쉽게말해 텐서라는 개념의 행렬을 순차계산 없이 통으로 계산하는 개념이다.