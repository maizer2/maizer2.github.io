---
layout: post
title: "ANN의 진행방식"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning, 1.2.2.1. ANN, a.b. Regression Problem]
---

### ANN이란

ANN이란 Artifical Nureal Network의 약자로 인공 신경망을 뜻한다.

ANN의 종류는 [Perceptron](https://maizer2.github.io/1.%20computer%20engineering/2022/05/18/perseptron.html), CNN, [AE](https://maizer2.github.io/1.%20computer%20engineering/2023/06/23/AE.html), [VAE](https://maizer2.github.io/1.%20computer%20engineering/2023/06/23/VAE.html) 등등 많다.

### ANN에 사용되는 변수

* $x$ : Feature, 특징
* $W$ : Weight, 가중치
    * 특징 $x$에 대한 가중치이다.
    * If $i$ == 1: $W_{1}$ = Random_number
    * else: $W_{i} = W_{i-1} + lr\cdot \frac{\sigma L}{\sigma w}$ 
* $y$ : True label, 실제 결과(레이블) 값 
    * 레이블은 쉽게 말해 정답을 의미한다. 
    * ex) 고양이, 강아지, $\cdots$  
* $\hat{y}$ : Predicted label, 예측 결과(레이블) 값
    * $ \hat{y} = W \cdot x $
    * 가중치와 특성값을 내적(Inner Product)해준 값이다.
* $lr$ : Learning Rate, Gradient Descent에서 기울기가 이동하는 크기를 의미한다. 스텝이라고도 한다.
    * $lr$이 너무 크면 발산한다.
    * $lr$이 너무 작으면 최소값을 갖기 위해 많은 연산을 필요로한다.
    * $lr$이 적당할 경우 발산하지 않을 정도의 속도로 최소값을 가지게 된다.

### ANN에 사용되는 함수

What is defferent Loss Function and Cost Fucntion<sup><a href="">[Link]</a></sup>
* Loss Function
    * MSE, Cross Entropy, $\cdots$
* Cost Function
    * Gradient Descent, Adam $\cdots$
    
---

### [Forward propagation, 전방 전파](https://maizer2.github.io/1.%20computer%20engineering/2022/05/17/Forward-Propagation.html)

1. $W$값들을 통해 $\hat{y}$ 값을 얻는다.
2. Loss Function을 통해 실제 레이블 $y$와 예측 레이블 $\hat{y}$값을 비교한다.

<br/>

### [Backward propagation, 후방 전파](https://maizer2.github.io/1.%20computer%20engineering/2022/05/17/Backward-Propagation.html)

Optimizer(최적화) : Forward propagation에서 사용될 $W$를 업데이트 해주는 과정

1. Loss값에 대한 $W$의 gradient 계산
2. Gradient descent를 이용하여 W 갱신