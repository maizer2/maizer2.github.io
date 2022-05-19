---
layout: post
title: "Backward Propagation"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning]
---

> 역전파<sup>backward propagation, backpropagation</sup>란 순전파의 계산 과정을 역순으로 거슬러가면서 손실 함숫값에 직간접적으로 영향을 미친 모든 성분에 대하여 손실 기울기를 계산하는 과정을 말한다.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

Optimizer(최적화) : [Forward Propagation](https://maizer2.github.io/1.%20computer%20engineering/2022/05/17/Forward-Propagation.html)에서 사용될 $W$를 업데이트 해주는 과정, 쉽게말해 Forward propagation의 과정인 합성함수  $f(g(w))$를 되돌아가는 과정, 미분해주는 과정이라고 볼 수 있다.

더 쉽게 말해 $f(g(w))$ 를 Loss Function으로 볼 수 있고 이는 Non-Linear한 데이터를 가지고 있기 때문에 어떤 지점에 필연적으로 최솟값을 가지게된다.<sup>[왜 Non-Linear Function은 최소값을 가지게 되는가?](https://maizer2.github.io/1.%20computer%20engineering/2022/05/18/why-convert-to-a-non-linear-function.html)</sup>

이 최솟값을 구하기 위해 Non-Linear한 함수에 가중치에 대한 미분값을 해주면 그 가중치에 대한 기울기를 얻을 수 있게된다.

이는 그 지점이 Non-Linear Function에서 어떤 위치에 있음을 알 수 있는 지표가 되는데, 기울기가 0에 수렴하도록 weight값을 업데이트 해줌으로서 최종적으로 Non-Linear Function의 최소값에 근사할 수 있게된다.

이 과정을 다음과 같이 분류할 수 있다.

1. Loss값에 대한 $W$의 gradient 계산
2. Optimizationd을 이용하여 W 갱신

### First Step of Backward propagation

Forward propagation에서 구한 Loss값 $L$에 대한 각 특성의 $W$의 Gradient를 구한다.

Gradient는 순간변화량(도함수) 즉 미분을 통해 구할 수 있다.

Loss Function과 $\hat{y}$는 합성함수로서 합성함수의 미분을 통해 도함수를 구할 수 있다.

$$ $$


이해의 편의성을 위해 가장 쉬운 LossFunction인 [MSE](https://maizer2.github.io/1.%20computer%20engineering/2022/04/08/%EC%A0%9C%EA%B3%B1%EA%B7%BC-%EC%98%A4%EC%B0%A8-MSE.html)를 사용한다.

$$ Loss() = MSE() $$
$$ L = MSE(f(W)) = \frac{1}{m}\sum_{i=1}^{m}(y - f(W_{ij}))^{2} = \frac{1}{m}\sum_{i=1}^{m}(y-\sum_{j=1}^{n}W_{ij}\cdot x_{j})^{2} $$
$$MSE(f(W))' = \frac{dL}{dW} = \frac{dL}{d\hat{y}}\cdot\frac{d\hat{y}}{dW} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$
$$ = \lim_{W\to0}\frac{\delta L}{\delta W} = \lim_{W\to0}(\frac{\delta L}{\delta \hat{y}}\cdot\frac{\delta \hat{y}}{\delta W})$$
$$ = \lim_{\hat{y}\to0}\frac{\delta L}{\delta \hat{y}} \times \lim_{W\to0}\frac{\delta \hat{y}}{\delta W} \;\;\;\;\;\;\;\;\;\;\;\;\;$$
$$ = \frac{dL}{d\hat{y}}\cdot\frac{d\hat{y}}{dW} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$ 
$$ MSE(f(W))' = MSE'(f(W)) \cdot f'(W) \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$
$$ MSE'(f(W)) = \frac{2}{m}\sum_{i=1}^{m}|y-f(W_{ij})|$$
$$ f'(W_{ij}) = \sum_{j=1}^{n}W_{ij}\cdot x_{j}$$

### Second Step of Backward propagation(Gradient Descent Algorithm)

미분을 통해 얻은 기울기 값과 $lr$, Learning Rate와 곱해 수정된 기울기가 0으로 수렴하도록, $w$값을 수정하는 과정이다.

$$ W_{i} = W_{i-1} - lr(\frac{\delta L}{\delta W}) $$

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 1.6 경사하강법과 역전파 59p, 윤덕호, 파이썬 날코딩으로 알고 짜는 딥러닝,  한빛미디어(주)