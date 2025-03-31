---
layout: post
title: "Softmax Function"
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning, 2.2.2. Mathematical Statistics]
---

### Background

> 복수의 후보 항목들에 대한 로짓값 벡터를 확률 분포 벡터로 변환하는 함수와 이렇게 구해진 확률 분포와 정답에 나타난 확률 분포 사이의 교차 엔트로피를 계산해주는 함수가 필요하게 되었다. ... 한 마디로 시그모이드 함수가 이진 판단 문제 해결의 주역이었다면 선택 분류 문제 해결의 주역은 소프트맥스 함수인 것이다.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

### Introduction

> Softmax it’s a function, not a loss. It squashes a vector in the range (0, 1) and all the resulting elements add up to 1. It is applied to the output scores $s$. As elements represent $a$ class, they can be interpreted as class probabilities. The Softmax function cannot be applied independently to each $s_{i}$, since it depends on all elements of $s$.<sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup>

> 소프트맥스 함수는 로짓값 벡터를 확률 분포 벡터로 변화해주는 비선형 함수다.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

각 출력에 대한 확률은 0과 1의 사이 값이며 출력값의 합은 1이 돼야 한다.

### Softmax computate using Sigmoid Function

softmax 함수는 결국 [Sigmoid Function](https://maizer2.github.io/1.%20computer%20engineering/2022/05/19/sigmoid-function.html)의 기초 개념으로 만들어진 함수이다.

> 일반적인 softmax함수가 왜 sigmoid함수와 동일한지 생각해보자면 이진 분류를 위한 sigmoid는 $\sigma(x)=\frac{1}{1+e^{-x}}$와 같이 두 확률값을 비례상수를 통해 도출한 함수이고, softmax함수는 다중 분류를 위한 N개의 확률 값을 비례상수를 통해 도출한 함수이다.

$$\mathrm{Softmax}(x)=\frac{e^{x_{i}}}{e^{x_{1}}+e^{x_{2}+\cdots+e^{x_{n}}}}$$

위 함수는 계산중 오버플로 발생 가능성이 있어 $x_{i}$ 가운데 최대값$x_{k}$을 찾은 후 다음과 같은 식을 사용한다.

$$\mathrm{Softmax}(x)=\frac{e^{x_{i}-x_{k}}}{e^{x_{1}-x_{k}}+e^{x_{2}-x_{k}}+\cdots+e^{x_{n}-x_{k}}}$$

#### why occur overflow

[Sigmoid Cross-Entropy](https://maizer2.github.io/1.%20computer%20engineering/2.%20mathematics/2022/05/15/Binary-Cross-Entropy-Sigmoid-Cross-Entropy.html) 에서 발생했던 $e^{-x}$ 의 $x$값이 너무 낮은 값을 가질 경우 값이 폭주해버리는 문제와 동일하게 위 식의 $e^{x_{i}}$값이 매우 커져 값이 폭주하는 문제, 너무 작아져 분모 분자가 0에 수렴하여 0으로 나눠지는 오류가 발생하게 된다.

#### Overflow problem solution

Overflow의 핵심은 $x$값이 커지는 문제이다.

이를 해결하는 대중적인 방법 중 하나는 $x$값 중 최대값인 $x_{k}$값을 찾아 위의 식에 분자와 분모를 최대값인 $x_{k}$로 나눠주는것이다.

이를 적용하면 임의의 $x_{j}$는 $x_{k}-x_{j}\geq{0}$ 와 같아짐으로 $0<e^{x_{k}-x_{j}}<1$ 가 되기 때문에 지나치게 커지는 문제가 해결되며, $e^{x_{k}-x_{k}}=e^{0}=1$를 통해 분모가 0이 되는 문제도 방지된다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a>  3.2 선택 분류 문제의 신경망 처리 128p, 윤덕호, 파이썬 날코딩으로 알고 짜는 딥러닝,  한빛미디어(주)

<a href="#footnote_2_2" name="footnote_2_1">2.</a> Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names - Softmax, gombru.github.io, Written May-23-2018,  Visit May-29-2022, [https://gombru.github.io/2018/05/23/cross_entropy_loss/](https://gombru.github.io/2018/05/23/cross_entropy_loss/)