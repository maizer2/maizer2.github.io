---
layout: post
title: "ANN의 진행방식"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning, 1.2.2.1. ANN, Loss Function, Cost Function]
---

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
    * Gradient Descent, $\cdots$
    
---

### Forward propagation, 전방 전파

1. $W$값들을 통해 $\hat{y}$ 값을 얻는다.
2. Loss Function을 통해 실제 레이블 $y$와 예측 레이블 $\hat{y}$값을 비교한다.

<br/>

#### First Step of Forward propagation

<center>$n$ is number of True labels</center>

$$ \hat{y} = \sum_{i=1}^{n} W_{i} \cdot x_{i}  $$

#### Second Step of Forward propagation

대표적인 Loss Function : [MSE](https://maizer2.github.io/1.%20computer%20engineering/2022/04/08/제곱근-오차.html), Cross Entropy, $\cdots$

<center>$m$ is number of Features</center>

$$MSE(\hat{y}) = \frac{1}{m}\sum_{i=1}^{m}(y - \hat{y}_{i})^{2}$$

두 label값의 차이를 구하는데, $W$가 잘 맞을 수록 크기가 작아진다.

<br/>

###  Backward propagation, (역) 전파

Optimizer(최적화) : Forward propagation에서 사용될 $W$를 업데이트 해주는 과정

1. Loss값에 대한 $W$의 gradient 계산
2. Gradient descent를 이용하여 W 갱신

#### First Step of Backward propagation

Gradient Descent, Second Step of Frontward propagation에서 구한 Loss값 $L$에 대한 각 특성의 $W$의 Gradient를 구한다.

Gradient는 순간변화량(도함수) 즉 미분을 통해 구할 수 있다.

Loss Function과 $\hat{y}$는 합성함수로서 합성함수의 미분을 통해 도함수를 구할 수 있다.

$$ Loss() = MSE() $$
$$ L = MSE(f(W)) = \frac{1}{m}\sum_{i=1}^{m}(y - f(W_{ij}))^{2} = \frac{1}{m}\sum_{i=1}^{m}(y-\sum_{j=1}^{n}W_{ij}\cdot x_{j})^{2} $$
$$MSE(f(W))' = \frac{dL}{dW} = \frac{dL}{d\hat{y}}\cdot\frac{d\hat{y}}{dW} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$
$$ = \lim_{W\to0}\frac{\delta L}{\delta W} = \lim_{W\to0}(\frac{\delta L}{\delta \hat{y}}\cdot\frac{\delta \hat{y}}{\delta W})$$
$$ = \lim_{\hat{y}\to0}\frac{\delta L}{\delta \hat{y}} \times \lim_{W\to0}\frac{\delta \hat{y}}{\delta W} \;\;\;\;\;\;\;\;\;\;\;\;\;$$
$$ = \frac{dL}{d\hat{y}}\cdot\frac{d\hat{y}}{dW} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$ 
$$ MSE(f(W))' = MSE'(f(W)) \cdot f'(W) \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$
$$ MSE'(f(W)) = \frac{2}{m}\sum_{i=1}^{m}|y-f(W_{ij})|$$
$$ f'(W_{ij}) = \sum_{j=1}^{n}W_{ij}\cdot x_{j}$$

#### Second Step of Backward propagation

미분을 통해 얻은 기울기 값과 $lr$, Learning Rate와 곱해 수정된 기울기가 0으로 수렴하도록, 다시말해

기울기가 0으로 수렴하는 의미는 $f(x)$이 최소값을 가지는 $x$값을 구하는 것이다.

따라서 

$$ W_{i} = W_{i-1} - lr(\frac{\delta L}{\delta W}) $$