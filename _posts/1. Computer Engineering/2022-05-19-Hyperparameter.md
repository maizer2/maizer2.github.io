---
layout: post
title: "Hyper Parameter?"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.1. Machine Learning, 1.2.2. Deep Learning]
---

### 하이퍼 파리미터

* 모델링할 때 ***사용자가 직접 세팅해주는 값***, 
* 학습을 ***시작하기 전에 정해두는 값***, 모델 학습 과정에 반영된다.

학습 데이터를 한 바퀴 돌 때 1epoch이라고 하며 n개의 epoch을 정해두고 학습을 시작한다.

또한 [미니배치](https://maizer2.github.io/1.%20computer%20engineering/2022/05/18/mini-batch.html) 크기를 의미하는 Iteration 또한 정해두고 학습을 시작한다.
Epoch과 Iteration처럼 **학습 과정에서 변경되지 않으면서 신경망 구조나 학습 결과에 영향을 미치는 고려 요인들을 Hyper parameter**라고 한다.