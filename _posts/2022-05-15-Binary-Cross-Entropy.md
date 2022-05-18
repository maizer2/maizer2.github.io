---
layout: post
title: What is Binary Cross Entropy?
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 2.2. Pobability and Statistics, a.b. Regression Problem]
---

### Binary Cross Entropy is?
    
> 이진 교차 엔트로피는 이진 분류 문제에서 쓰는 교차 엔트로피입니다. 판별기중에 참을 1.0, 거짓을 0.0으로 표시하는 신경망이 대표적으로 이 손실함수를 쓴다고 할 수 있습니다.<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

필자는 Cross Entropy와 Binary Cross Entroyp와 다른 뭔가가 있을거라고 생각하고 깊게 공부를 했었다.

하지만 Binary Cross Entropy는 2개의 데이터데 대한 Cross Entropy라고 볼 수 있다 ;;

쉽게 말해 그냥 다음과 같다

$$BinaryCrossEntropy(y, x) = - \sum_{i=1}^{2}y_{i}\cdot lnx_{i}$$
$$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\ = - y_{1}\cdot lnx_{1} - y_{2}\cdot lnx_{2}$$

두 값의 합이 1일 경우, 한 값이 $a$이면 다른 값은 $1 - a$가 된다.

이를 식에 적용하면 다음과 같다.

$$BinaryCrossEntropy(y, x) = - y_{1}\cdot lnx_{1} - (1 - y_{1})\cdot ln(1 - x_{1})$$

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 부록. BCE 손실 248p, GAN 첫걸음, 타리크라시드 지음, 고락윤 옮김, 한빛미디어(주)