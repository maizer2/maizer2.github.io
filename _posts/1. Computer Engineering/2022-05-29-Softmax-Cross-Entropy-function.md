---
layout: post
title: "Softmax Cross-Entropy Function"
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning, 2.2.2. Mathematical Statistics]
---

### Background

[Softmax Function](http://maizer2.github.io/1.%20computer%20engineering/2.%20mathematics/2022/05/29/Softmax-function.html)의 결과값을 [Cross-Entropy](https://maizer2.github.io/1.%20computer%20engineering/2.%20mathematics/2022/05/15/Cross-Entropy.html)의 확률값을 Q로 지정하게 될 때 문제가 발생한다.

### Introduction

로짓값들 중 가장 작은 로짓값 $a_{i}$값이 너무 작아 0으로 표현될 수 있다.

이런 값이 log값에 적용되면 $-\infty$로 폭주하는 계산 오류가 발생한다.

이를 해결하기 위해서 아주 작은 양수값 $\epsilon$을 도입하여 다음과 같이 고쳐 쓸 수 있다.

### Softmax Cross-Entropy computate

$$H(P, Q) = \sum_{i=1}^{n}p_{i}\;log\;q_{i}\approx{-\sum_{i=1}^{n}p_{i}\;log\;(q_{i}+\epsilon)}$$

#### why add a very small positive value

원래 cross-entropy 함수와 $q_{i}$ 값에 매우 작은 양수를 추가한 근사값은 아주 작은 차이만 가져오게 돼서 큰 문제가 되지 않지만 $q_{i}$ 값이 0에 가까운 수가 될 경우 $\epsilon$은 하한선 역할을 하면서 0이 되지 않도록 막게 된다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a>  3.5 소프트맥스 교차 엔트로피 134p, 윤덕호, 파이썬 날코딩으로 알고 짜는 딥러닝,  한빛미디어(주)
