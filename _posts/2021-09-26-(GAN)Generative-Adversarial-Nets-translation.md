---
layout: post
title: "(GAN)Generative Adversarial Nets Translation"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning, 1.2.2.5. GAN, 1.7. Literature Review]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

$$Generative\;Adversarial\;Nets$$

<h3><p align="center">Abstract</p></h3>

> We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model $ G $ that captures the data distribution, and a discriminative model $ D $ that estimates the probability that a sample came from the training data rather than $ G $.
>> 우리는 데이터 분포를 측적하는 생성 모델 $ G $ 와 $ G $ 가 아닌 샘플에서 얻게되는 훈련 데이터의 확률을 추정하는 식별 모델 $ D $를 동시에 훈련함으로서 적대적 과정을 통해 생성 모델을 추정하는 새로운 프레임워크를 제시한다.

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

---

<h3>1 Introduction</h3>

> The promise of deep learning is to discover rich, hierarchical models that represent probability distributions over the kinds of data encountered in artificial intelligence applications, such as natural images, audio waveforms containing speech, and symbols in natural language corpora.
>> 딥러닝의 약속은 자연 이미지, 음성을 포함한 오디오 파형, 자연어 말뭉치의 기호와 같은 인공지능 어플리케이션이 접하고 있는 데이터의 종류에 대한 확률 분포를 나타내는 풍부하고 계층적 모델을 발경하는 것이다.

> So far, the most striking successes in deep learning have involved discriminative models, usally those that map a high-dimensional, rich sensory input to a class label.
>> 지금까지 딥러닝의 두드러지는 성공은 차별적 모델로서, 보통 고차원적이고, 풍부한 감각의 입력을 클래스 레이블에 매핑하는 모델이었다.

> These striking successes have primarily been based on the backpropagation and dropout algorithms, using piecewise linear units which have a particularly well-behaved gradient.
>> 이 두드러지는 성공은 대체로 역전파와 dropout 알고리즘이 기반이 되며, 특히 잘 작동하는 기울기를 가지는 조각별 선형(piecewise linear) 단위를 사용한다.

> Deep generative models have had less of an impact, due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context.
>> 심층 생성 모델은 최대 가능성 추정 및 관련 전략에서 발생하는 많은 다루기 어려운 확률적 계산을 근사화하는 어려움과 생성 맥락에서 부분 선형 단위의 이점을 활용하는 어려움으로 인해 영향을 덜 받았다.
