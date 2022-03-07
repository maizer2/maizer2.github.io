---
layout: post
title: "(GAN)Generative Adversarial Nets Review"
categories: 논문리뷰
tags: [AI, 논문]
---

> We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.
>> 우리는 적대적 프로세스를 통해 생성 모델을 추정하는 새로운 프레임워크를 제시한다. 우리는 두 모델을 동시에 훈련시키는데: 데이터 분포를 감지하는 생성모델 G와 샘플은 G가 아닌 훈련 데이터에서 오는 확률을 추정하는 식별모델 D이다.

>The training procedure for G is to maximize the probability of D making a mistake.
>> G의 훈련과정은 D가 실수를 야기할 확률을 최대로 한다.



what is sample
