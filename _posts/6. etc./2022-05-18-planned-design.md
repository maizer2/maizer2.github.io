---
layout: post
title: "Planned Design"
categories: [6. etc.]
tags: [6.2 삶을 살아가는 태도]
---

### 주저리 주저리

내 석사의 목표는 논문 3편쓰기다.

3편을 쓰기위해서는 국내, 국외 논문 투고 일자를 파악해야하고 어디에 낼 건지, 내는 곳의 분야와 논문 규격등 미리 알아야할 것이 많다.

오늘은 본격적으로 논문을 작성하기 전에 기초 공부와 읽어야 할 논문들을 정리해보도록 한다.

논문 작성을 시작할 날짜는 7월로 생각하고 있고 7월부터 한편 내년에 2편 써볼 것이다.

현재 GAN을 공부하고 있고 StyleGAN 2까지의 논문의 행적만 알고있지 크게 공부하거나 이해하지는 못한다.

현재 Corsera 강의를 통해 매주 GAN 수업을 듣고 있다.

저번주는 Multi layer perceptron을 사용한 Full connected 연산으로 구현된 Vanila GAN을 구현해보는 수업을 했고

이번주는 Convolutional Neural Network를 사용한 DCGAN을 구현해보는 시간이다.

사실 공부를 하면서 가장 큰 장애물이 됬던 점은 현재도 마찬가지로 이론적 지식의 한계이다.

수학적 특히 확률분포쪽 지식이 부족하다보니 근본적인 Activation Function과 Cost, Loss Function들이 이해가 안갈 때가 많다.

물론 지금은 많은 노가다를 통해서 대부분의 기초적인 내용은 이해한 상태지만, 만약 논문에서 어떤 수학적 식이 눈앞에 펼쳐졌을 때 바로 이해가 갈 수 있을지 의문이 든다.

또한 시각적 이해하지 못하는 내용은 이해가 잘 안된다. 내 뇌가 그쪽으로 발단이 안됬는지 이해가 계속 안된다.

예를 들어서 perceptron의 과정에서 가중치와 편향을 통해 계산된 값이 시각적으로 그려지지 않는다.

아마도 너무 다양한 자료들을 봐서 큰 혼란이 온 것같다. 쉽게 말해서 현재 정형화되지 않은 인공지능이라는 분야에서 사람들은 한 개념을 너무 다양하게 설명하고 다르게 이해하고있다.

나는 이 과정이 너무 힘들고 이해가 안가는데, 특히 gradient Discent algorithm을 적용하기 위해 gradient를 표현한 그래프를 2차원 식으로 처음 접했을 때 그리고 다차원으로 표현된 그래프를 접했을 때 너무 큰 혼돈이 와서 생각이 멈췄었다.

이런 과정이 너무 생략되고 설명이 부실해서 물어볼 사람도 없고 너무 힘들지만 해결해 나가고 있다.

하지만 아직도 왜 Activation Function은 선형함수를 비선형함수로 바꿔준다는지 이해가 안간다.

공부하면 그만이야 ~ 

아무튼 DCGAN 구현을 하게 되면 DCGAN 논문을 읽어봐야지 않겠는가? 다음주 발표도 있고 ..

### 7월 전까지 계획

오늘 5월 18일, 7월 전까지 약 6주 남았다.

목표는 남은 6주동안 StyleGAN 2까지 읽고 이해하는 것이다.

GAN의 주요 논문인 GAN, DCGAN, InfoGAN, WGAN, WGAN2, LSGAN, ENERGY-GAN, BEGAN, Conditional-GAN, Image2Image-GAN, unpair-GAN, semantic-GAN, star-GAN, super-resolution-GAN, spectral-normalization-GAN, self-attention-GAN, progressive-GAN, Large-scare-GAN, A-Style-based-GAN, Style-GAN 

... 실환가?

6주동안 다 읽기는 사실 불가능이다. 한주에 하나 잡아서 styleGan까지 읽는걸로 줄여서

DCGAN, InfoGAN, WGAN, BEGAN, Conditional-GAN, Image2Image-GAN, .... 

사실 너무 좋은 논문들이라 걸러읽기가 불가능한 정도이다..

일단 이번주 DCGAN 논문을 읽고 판단하도록 하자..

그리고 Coursera GAN 수업을 6월 6일까지 들어야한다.

실제 구현은 수업에서 하면 될 것 같고

기초 공부는 Coursera에 나오는 부분과 논문에 나오는 부분을 하도록 한다.

매주 논문 한편 + coursera 수업 하는 걸로