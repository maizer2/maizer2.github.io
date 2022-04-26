---
layout: post
title: "Loss Function과 Cost Function의 차이점"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning]
---

> 비용 함수(cost function)는 손실 함수(loss function)의 다른 말입니다. 엄밀히 말하면 손실 함수는 샘플 하나에 대한 손실을 정의하고 비용 함수는 훈련 세트에 있는 모든 샘플에 대한 손실 함수의 합을 말합니다. 하지만 보통 이 둘을 엄격히 구분하지 않고 섞어서 사용합니다.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

[ANN의 과정](https://maizer2.github.io/1.%20computer%20engineering/2022/04/26/How-ANN-proceeds.html)을 보면 쉽게 이해할 수 있다.

First Step of Backward Propagation에서 Loss Function을 사용해 L 값을 구한다.

Second Steop of Backward Propagation에서 Cost Function을 사용해 L

---

##### 참고문헌

<a href="#footnote_1_2" name="footnote_1_1">1.</a> Chapter 04-02 손실함수, 혼자 공부하는 머신러닝 + 딥러닝, 박해선, 한빛미디어(주)