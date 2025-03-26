---
layout: post 
title: "Cross-Correlation Calculation"
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 2.1. Pure mathematics, 2.1.2. Linear Algebra]
---

### Introduction

Cross-Correlation Calculation은 CNN에서 쓰이는 연산이다.

Cross-Correlation은 [Convolution Calculation](https://maizer2.github.io/2.%20mathematics/2022/05/19/Convolution-Calculation.html)의 Flip 과정을 하지 않고 Filter를 그대로 Input value와 연산한다.

### Computation

$$ y(i) = (x \ast w)(i) = \sum_{k=-\infty}^{\infty} x(k)w(i+k) $$

$$ y(i, j) = (x \ast w)(i, j) = \sum_{k_{1}=-\infty}^{\infty}\sum_{k_{2}=-\infty}^{\infty} x(k_{1}, k_{2})w(i+k_{1}, j + k_{2}) $$


식을 쉽게 이해하려면 [Forward Propagation](https://maizer2.github.io/1.%20computer%20engineering/2022/05/17/Forward-Propagation.html)의 First Step of Forward propagation을 참고하자
