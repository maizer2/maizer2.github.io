---
layout: post 
title: "Convolution Calculation"
categories: [2. Mathematics]
tags: [2.1. Linear Algebra]
---

### Introduction

> Convolution Calculation은 CNN에서 쓰이지 않는다. CNN에서는 [Cross-Correlation](https://maizer2.github.io/1.%20computer%20engineering/2.%20mathematics/2022/05/19/Cross-Correlation-Calculation.html) 연산을 사용한다. Kernel을 Flip하지 않는 이유는 filter를 random으로 초기화하기 때문에 flip에 의미가 없다.

> Convolution Calculation은 하나의 함수와 또 다른 함수를 반전 이동한 값을 곱한 다음, 구간에 대해 적분하여 새로운 함수를 구하는 수학 연산자이다. 합성곱 연산은 두 함수 f, g 가운데 하나의 함수를 반전(reverse), 전이(shift)시킨 다음, 다른 하나의 함수와 곱한 결과를 적분하는 것을 의미한다. 이를 수학 기호로 표시하면 다음과 같다.

이 연산을 CNN의 이미지와 필터를 합성곱한다고 생각하면 전혀 이해가 안가니까 아래 그림을 참고하면 이해하기 쉽다.

![Convolution Calculation](https://upload.wikimedia.org/wikipedia/commons/9/97/Convolution3.PNG)

### Computation

$$ (f \ast g)(t) = \int_{-\infty}^{\infty}f(\tau)g(t-\tau)d\tau $$
