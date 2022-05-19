---
layout: post 
title: "Gradient Descent Algorithm"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2.1. ANN]
---

### Introduction

Gradient Descent Alorithm은 Optimization Algorithm 중 하나이다.

Neural Network는 예측값과 실제값의 차이를 계산하는 Loss Function을 통해 Loss value를 구한다. 

Loss Function은 용도에 맞는 다양한 함수가 존재하는데, 가장 단순한 Loss Function인 [MSE](https://maizer2.github.io/1.%20computer%20engineering/2022/04/08/%EC%A0%9C%EA%B3%B1%EA%B7%BC-%EC%98%A4%EC%B0%A8-MSE.html)를 사용하면 두 값의 차이의 제곱 평균인 Loss value를 구할 수 있다.

이 과정을 통해 Linear Function이 Non-Linear Function으로 변환된다.

학습은 [Forward Propagation](https://maizer2.github.io/1.%20computer%20engineering/2022/05/17/Forward-Propagation.html)을 통해 얻은 Loss value를 최소화하는 방향으로 진행되는데, 이 과정을 [Backward Propagation](https://maizer2.github.io/1.%20computer%20engineering/2022/05/17/Backward-Propagation.html)이라고하고 Optimization 중 하나인  **Gradient Descent Algorithm**이 사용된다.

> [Neural Network's Detail works](https://maizer2.github.io/1.%20computer%20engineering/2022/05/18/why-convert-to-a-non-linear-function.html)를 통해, 왜 Optimization을 써야하는지 알 수 있다.

### Gradient Descent Algorithm Sequence

다른 Optimization algorithm과 비교하면 가장 단순하다.

$$ W_{i} = W_{i-1} - lr(\frac{\delta L}{\delta W}) $$

$w$에 대한 편미분 값, 즉 w에 대한 기울기를 Forward Propagation에 사용했던 Weight에 뺴준다.

$Learning Rate$ 인 [hyperparameter]()를 사전에 정의하여 gradient의 감소 폭을 조절한다.