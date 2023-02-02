---
layout: post 
title: "Batch Normalization: Accelerating Deep Network Training b y Reducing Internal Covariate Shift"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review]
---

## Abstract

> Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. 
>> 심층 신경망(Deep Neural Networks) 훈련은 이전 계층의 매개 변수가 변경됨에 따라 훈련 중에 각 계층의 입력 분포(distribution)가 변경된다는 사실로 인해 복잡하다(is complicated). 

> This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. 
>> 이는 낮은 학습률(learning rates)과 신중한 매개 변수 초기화(careful parameter initialization)를 요구하여 훈련 속도를 늦추고 포화 비선형성(saturating nonlinearities)을 가진 모델을 훈련시키는 것을 악명높도록 어렵게(notoriously hard) 만든다. 

> We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. 
>> 우리는 이 현상(phenomenon)을 내부 공변량 이동(internal covariate shift)이라고 하며, 레이어 입력을 정규화(normalizing)하여 문제를 해결한다. 

> Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. 
>> 배치 정규화(Batch Normalization)를 사용하면 훨씬 더 높은 학습 속도(learning rates)를 사용하고 초기화(initialization)에 덜 주의할 수 있다. 

> Batch Normalization allows us to use much higher learning rates and be less careful about initialization. 
>> 또한 정규화 기능(regularizer)을 수행하여 경우에 따라 드롭아웃(Dropout)이 필요하지 않습니다. 

> It also acts as a regularizer, in some cases eliminating the need for Dropout. 
>> 또한 정규화 기능(regularizer)을 수행하여 경우에 따라 드롭아웃(Dropout)이 필요하지 않습니다. 

> Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. 
>> 최첨단 이미지 분류 모델에 적용된 배치 정규화(Batch Normalization)는 14배 적은 훈련 단계(14 times fewer training steps)로 동일한 정확도를 달성하고 원래 모델을 상당한 차이(significant margin)로 능가한다(beats). 

> Using an ensemble of batchnormalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.
>> 배치 정규화된(batchnormalized) 네트워크 앙상블(ensemble)을 사용하여 ImageNet 분류에 대한 가장 잘 발표된 결과를 개선한다(improve upon). 4.9%의 상위 5개 검증 오류(및 4.8%의 테스트 오류)에 도달하여 인간 평가자의 정확도를 초과한다.

## 1. Introduction

> Deep learning has dramatically advanced the state of the art in vision, speech, and many other areas. 
>> 딥 러닝은 비전, 스피치 및 많은 다른 영역에서 기술 수준을 극적(dramatically)으로 발전시켰다(advanced). 

> Stochastic gradient descent (SGD) has proved to be an effective way of training deep networks, and SGD variants such as momentum (Sutskever et al., 2013) and Adagrad (Duchi et al., 2011) have been used to achieve state of the art performance. 
>> 확률적 경사 하강법(Stochastic gradient descent)(SGD)은 심층 네트워크(deep networks)를 훈련하는 효과적인 방법으로 입증되었으며, momentum(Sutskever et al., 2013) 및 Adagrad(Duchi et al., 2011)과 같은 SGD 변형이 최첨단 성능을 달성하는 데 사용되었다. 

> SGD optimizes the parameters Θ of the network, so as to minimize the loss
>> SGD는 손실을 최소화(minimize the loss)하기 위해 네트워크의 매개변수 (parameters) $Φ$를 최적화(optimizes)합니다

$$ Θ = \arg\underset{Θ}{\min{}}\frac{1}{N}\sum^{N}_{i=1}ℓ(x_{i}, Θ)$$

> where $x_{1...N}$ is the training data set. 
>> 여기서 x_{1...N}은(는) 교육 데이터 집합(training data set)입니다.

> With SGD, the training proceeds in steps, and at each step we consider a minibatch $x_{1...m}$ of size $m$.
>> SGD를 사용하며, 훈련은 단계별로 진행되고(proceeds), 각 단계에서 $m$ 크기의 미니 배치(minibatch) $x_{1...m}$를 고려한다.

> The mini-batch is used to approximate the gradient of the loss function with respect to the parameters, by computing 
>> 미니 배치(minibatch)는 계산을 통해 매개 변수에 대한 손실 함수의 기울기(loss function)를 근사화(approximate)하는 데 사용됩니다

$$\frac{1}{m}=\frac{∂ℓ(x_{i}, Θ)}{∂Θ}.$$

> Using mini-batches of examples, as opposed to one example at a time, is helpful in several ways.
>> 한번에 하나의 example만 사용하는 것과 대조적으로(as opposed to), 여러개의 example을 mini-batch에 사용하는 것이 여러가지 측면에서 유용하다.

>