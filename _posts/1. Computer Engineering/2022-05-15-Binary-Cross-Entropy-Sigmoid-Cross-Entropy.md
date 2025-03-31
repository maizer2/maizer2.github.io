---
layout: post
title: What is Binary Cross Entropy(Sigmoid Cross-Entropy)?
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 2.2.2. Mathematical Statistics, a.b. Regression Problem]
---

### Introduction

Refer [Sigmoid Function](https://maizer2.github.io/1.%20computer%20engineering/2022/05/19/sigmoid-function.html)., [Cross Entropy Function](https://maizer2.github.io/1.%20computer%20engineering/2.%20mathematics/2022/05/15/Cross-Entropy.html)

> Binary Cross Entropy called Sigmoid Cross-Entropy loss. It is a Sigmoid activation plus a Cross-Entropy loss. Unlike Softmax loss it is independent for each vector component (class), meaning that the loss computed for every CNN output vector component is not affected by other component values. That’s why it is used for multi-label classification, were the insight of an element belonging to a certain class should not influence the decision for another class. It’s called Binary Cross-Entropy Loss because it sets up a binary classification problem between $C′=2$ classes for every class in $C$ , as explained above.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>
>> Sigmoid Cross-Entropy라고 불리는 Binary Cross Entropy는 Sigmoid 활성화 함수와 Cross-Entropy loss function을 합친 것입니다. Softmax와는 다르게 출력값(class)이 독립적이며, 이는 모든 CNN의 결과인 loss값이 다른 결과에 영향을 미치지 않는다는 의미입니다. 클래스에 독립적이기 때문에 다중 레이블 분류에 사용되는 이유입니다.

> 이진 교차 엔트로피는 이진 분류 문제에서 쓰는 교차 엔트로피입니다. 판별기중에 참을 1.0, 거짓을 0.0으로 표시하는 신경망이 대표적으로 이 손실함수를 쓴다고 할 수 있습니다.<sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup>

필자는 Cross Entropy와 Binary Cross Entroyp와 다른 뭔가가 있을거라고 생각하고 깊게 공부를 했었다.

하지만 Binary Cross Entropy는 2개의 데이터데 대한 Cross Entropy라고 볼 수 있다 ;;

---

### Computation from Cross-Entropy

$$BinaryCrossEntropy(y, x) = - \sum_{i=1}^{2}y_{i}\cdot lnx_{i}$$

$$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\ = - y_{1}\cdot lnx_{1} - y_{2}\cdot lnx_{2}$$

두 값의 합이 1일 경우, 한 값이 $a$이면 다른 값은 $1 - a$가 된다.

이를 식에 적용하면 다음과 같다.

$$BCE(y, x) = - y_{1}\cdot lnx_{1} - (1 - y_{1})\cdot ln(1 - x_{1})$$

---

### Computation from Sigmoid

> 이진 판단 문제에 대한 정답으로 z가 주어졌다면 이 데이터의 결과가 참일 확률을 $p_{T}$ , 거짓일 확률을 $p_{F}$ 이라 할 때, $p_{T} = z, p_{F}=1-z$ 임을 나타낸다. $\cdots$ 이 데이터에 대해 신경망 회로의 출력이 로짓값 x로 계산되었다고 할 때 이에 대응하는 확률값은 $q_{T}=\sigma(x), q_{F}=1-\sigma(x)$에 해당한다. 이제 교차 엔트로피의 정의식 $H(P,Q)=-\sum{p_{i}\log{q_{i}}}$ 에 위의 확률값들을 대입하면 $H=-p_{r}\log{q_{T}-p_{F}}=-z\log{\sigma(x)-(1-z)\log(1-\sigma(x))}$ 이다.<sup><a href="#footnote_3_1" name="footnote_3_2">[3]</a></sup>

$$SigmoidCrossEntropy(x)=-z\log{\frac{1}{1+e^{-x}}}-(1-z)\log{\left(1-\frac{1}{1+e^{-x}}\right)}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$=-z\log{\frac{1}{1+e^{-x}}}-(1-z)\log{\frac{e^{-x}}{1+e^{-x}}}\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$\;\;\;\;\;\;=z\log{\left(1+e^{-x}\right)}-(1-z)\left(\log{e^{-x}}-\log{(1+e^{-x})}\right)$$

$$=(z+1-z)\log{(1+e^{-x})}-(1-z)(-x)\;\;\;\;\;\;\;\;\;\;$$

$$=x-xz+\log{\left(1+e^{-x}\right)}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$
---

만약 $z$값이 0과 1로 주어진 경우(대부분의 데이터 셋은 0과 1로 주어진다.) 다음과 같이 두 값으로 제한해 정의할 수 있다.

$$\mathrm{If}\;z=0, SigmoidCrossEntropy(x)=x+\log{(1+e^{-x})}$$

$$\mathrm{If}\;z=1, SigmoidCrossEntropy(x)=\log{(1+e^{-x})}\;\;\;\;\;\;\;$$

---

### Computation from Sigmoid Function about Cross-Entropy

$$\frac{\partial{H}}{\partial{x}}=-z+\sigma(x)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

***Proof***

$$\frac{\partial{H}}{\partial{x}}=\frac{\partial}{\partial{x}}\left(x-xz+\log{\left(1+e^{-x}\right)}\right)\;\;\;$$

$$*\;\left\{\log\left(1+e^{-x}\right)\right\}'=\left\{g(f(x))\right\}'=g'\left(f\left(x\right)\right)\times{f'\left(x\right)}=\frac{\left(1+e^{-x}\right)'}{1+e^{-x}}\;$$

$$=1-z+\frac{\left(1+e^{-x}\right)'}{1+e^{-x}}\;\;\;\;\;\;\;\;\;\;\;$$

$$=1-z+\frac{-e^{-z}}{1+e^{-x}}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$=-z+\frac{1}{1+e^{-x}}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$=-z+\sigma{\left({x}\right)}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

---

### Troubleshooting Calculation Congestion

$\sigma(x)=\frac{1}{1+e^{-x}}$을 프로그래밍으로 계산할 때, $x$값이 큰 음수가 들어갈 경우 오버플로 오류가 발생한다.

Sigmoid Cross-Entropy를 계산할 때 Sigmoid함수가 들어가기 때문에 $e^{-x}$값이 폭주하게 될 위험이 존재한다.

이를 해결하기 위해서는 $e^{-x}$의 $x$가 음수일 때 계산 방법을 달리 하는 것이다.

하지만 딥러닝에서 미니배치 처리를 일괄처리 할 수 있어야하기 때문에 음수, 양수로 나눠 처리하기보다 하나의 식으로 처리해야한다.

$$\sigma{\left(x\right)}=\frac{e^{-\max{\left(-x,0\right)}}}{1+e^{-|x|}}$$

$$H=\max{(x,0)}-xz+\log{\left(1+e^{-|x|}\right)}$$

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names - Binary Cross-Entropy Loss, gombru.github.io, Written May-23-2018,  Visit May-28-2022, [https://gombru.github.io/2018/05/23/cross_entropy_loss/](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

<a href="#footnote_2_2" name="footnote_2_1">2.</a> 부록. BCE 손실 248p, GAN 첫걸음, 타리크라시드 지음, 고락윤 옮김, 한빛미디어(주)

<a href="#footnote_3_2" name="footnote_3_1">1.</a> 2.7 시그모이드 교차 엔트로피와 편미분 - 시그모이드 교차 엔트로피 정의식 도출 과정 103p, 윤덕호, 파이썬 날코딩으로 알고 짜는 딥러닝,  한빛미디어(주)