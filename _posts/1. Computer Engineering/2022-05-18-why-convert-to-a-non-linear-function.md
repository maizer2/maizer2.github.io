---
layout: post 
title: "Why convert to a Non-Linear Function?"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence]
---

### Introduction

인공신경망을 훈련하기 위해서는 Weight와 Bias를 수정하는 과정이 필요하다.

Weight와 Bias를 통해 계산된 값은 단순한 1차 함수로 선형 값을 가지게된다.

선형 함수를 사용하면 최소값을 알 수 없는데 이를 해결하기 위해 비선형 함수로 변환해주는 과정이 발생한다.

왜 비선형 함수를 사용하면 최소값을 알 수 있을까? 

---

### Why convert to a Non-Linear Function?

![Quadratic Function](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/%D7%92%D7%A8%D7%A3x%5E2.svg/2560px-%D7%92%D7%A8%D7%A3x%5E2.svg.png)

$$ f(x) = wx + b $$

단순한 비선형 그래프인 2차함수<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>를 예로 들자면 그림과 같이 최소값을 가지는 지점이 발생한다.

2차 함수는 한개의 미지수 값을 입력으로 받는 함수를 의미하고 2차원 데이터로 표현된다. 


![3D Fucntion](https://jamesmccaffrey.files.wordpress.com/2013/11/doubledip.jpg)

$$ f(x, y) = ax + by + c $$

만약 두개의 미지수를 입력으로 받는 함수라면 다음과 같이 3차원<sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup> 데이터로 표현이 될 것이다.

신경망에서 지수값이 많아짐에 따라 차원이증가해서3차원, 4차원 시각적으로 표현하지못하고 사람이 인식하지 못하는 수준에 가더라도 비선형 그래프는 최소값을 가지는 지점이 존재하게된다.

이 Non-Linear Graph의 최소값을 찾아가기 위해 고안된 Algorithm이 바로 [Gradient Descent Algorithm](https://maizer2.github.io/1.%20computer%20engineering/2022/05/18/Gradient-Descent-Algorithm.html)이다.

---

### Summury

신경망은 단순히 Linear Function이므로 최소값을 찾을 수 없다.

Linear Function을 Non-Linear Function에 합성하면 다음과 같게된다.

$$ f(g(w)) $$

Linear Function인 $g(w)$가 Non-Linear Function $f(y)$를 통과하며 Non-Linear Function으로 변환된다.

Non-Linear Function의 최소값을 찾는 과정인 [Gradient Descent Algorithm](https://maizer2.github.io/1.%20computer%20engineering/2022/05/18/Gradient-Descent-Algorithm.html)을 사용하면 Function의 최소지점으로 도달할 수 있게된다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 이차 함수, wikipedia, [https://ko.wikipedia.org/wiki/이차_함수](https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%B0%A8_%ED%95%A8%EC%88%98)

<a href="#footnote_2_2" name="footnote_2_1">2.</a> Graphing the 3D Double-Dip Function, Posted on December 16, 2013, jamesdmccaffrey, [https://jamesmccaffrey.wordpress.com/2013/12/16/graphing-the-3d-double-dip-function/](https://jamesmccaffrey.wordpress.com/2013/12/16/graphing-the-3d-double-dip-function/)