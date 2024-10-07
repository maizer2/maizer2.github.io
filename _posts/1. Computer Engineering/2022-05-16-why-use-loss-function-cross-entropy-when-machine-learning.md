---
layout: post
title: "Why use loss function Cross Entropy when Machine Learning"
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 2.2. Pobability and Statistics, a.b. Regression Problem]
---

$$CrossEntropy(P, Q) =  \sum -P(x)lnQ(x)$$

CrossEntropy에서 $-log$를 사용하는 이유는 다음 그래프를 보면 알 수 있다.

![minus log graph](https://t1.daumcdn.net/cfile/tistory/2603F434579AF9B52A)
<center>y = -log(x)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup></center><br/>

$-log(x)$의 그래프를 보면 x가 0으로 수렴할 때 y값이 급격히 커지는 것을 알 수 있다.

Cross Entropy를 사용하면 오답에 대한 큰 피드백을 받을 수 있다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 21. 로그 함수 정리, 파이쿵, 2016.07.29 작성, 2022.05.16 방문, [https://pythonkim.tistory.com/28](https://pythonkim.tistory.com/28)