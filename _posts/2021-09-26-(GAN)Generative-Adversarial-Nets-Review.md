---
layout: post
title: "(GAN)Generative Adversarial Nets Review"
categories: 논문리뷰
tags: [AI, 논문]
---
<p align="center">Abstract</p>

> We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model $ G $ that captures the data distribution, and a discriminative model $ D $ that estimates the probability that a sample came from the training data rather than $ G $.
>> 우리는 적대적 프로세스를 통해 생성 모델을 추정하는 새로운 프레임워크를 제시한다. 우리는 두 모델을 동시에 훈련시키는데: 데이터 분포를 감지하는 생성모델 $ G $와 샘플은 $ G $ 가 아닌 훈련 데이터에서 오는 확률을 추정하는 식별모델 $ D $ 이다.

>The training procedure for $ G $ is to maximize the probability of $ D $ making a mistake.
>> $ G $ 의 훈련과정은 $ D $ 가 실수를 야기할 확률을 최대로 한다.

> This framework corresonds to a minimax two-player game.
>> 이 프레임워크는 두명에서하는 minimax 게임이다.

> In the space of arbitrary functions $ G $ and $ D $ , a unique solution exists, with $ G $ recovering the training data distribution and $ D $ equal to $ \frac{1}{2} $ everywhere.
>> 임의의 함수 $ G $ 와 $ D $ 의 공간에는 훈련 데이터 분포를 회복시키는 $ G $ 와 항상 $ \frac{1}{2} $ 이 되는 $ D $ 로써 고유한 해결책이 존재한다. 

> In the case where $ G $ and $ D $ are defined by multilayer perceptrons, the entire system can be trained with backpropagation.
>> $ G $ 와 $ D $ 가 다층 퍼셉트론으로 정의되는 경우, 전체 시스템에서 역전파로 훈련된다.

> There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples.
>> 훈련 또는 샘플 생성 중 Markov 연쇄 혹은 근사 추론 네트워크를 전개할 필요가 없다.

>  Experiments demonstrate the potential of the framework through qualitative and  quantitative evaluation of the generated samples.
>> 실험을 증거로 생성 샘플에 대한 양질적인 평가를 통해 프레임워크의 잠재력을 보여줄 수 있다.

what is sample
