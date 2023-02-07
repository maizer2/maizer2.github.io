---
layout: post 
title: "Layer Normalization"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review]
---

## Abstract

> Training state-of-the-art, deep neural networks is computationally expensive. 
>> 최첨단 심층 신경망을 훈련하는 것은 계산 비용이 많이 든다.

> One way to reduce the training time is to normalize the activities of the neurons. 
>> 훈련 시간을 줄이는 한 가지 방법은 뉴런의 활동을 정상화하는 것이다.

> A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. 
>> 배치 정규화라고 불리는 최근 도입된 기술은 훈련 사례의 미니 배치에 대한 뉴런에 대한 합산 입력의 분포를 사용하여 평균과 분산을 계산한 다음 각 훈련 사례에서 해당 뉴런에 대한 합산 입력을 정규화하는 데 사용된다.

> This significantly reduces the training time in feed forward neural networks. 
>> 이는 피드포워드 신경망에서 훈련 시간을 크게 단축시킨다.

> However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. 
>> 그러나 배치 정규화의 효과는 미니 배치 크기에 따라 달라지며 이를 반복 신경망에 적용하는 방법은 명확하지 않다.

> In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. 
>> 본 논문에서, 우리는 단일 훈련 사례의 레이어에서 뉴런으로 합계된 모든 입력에서 정규화에 사용되는 평균과 분산을 계산하여 배치 정규화를 레이어 정규화로 전환한다.

> Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. 
>> 배치 정규화와 마찬가지로, 우리는 또한 정규화 후에 비선형성 이전에 적용되는 각 뉴런에 자체 적응 편향과 이득을 제공한다.

> Unlike batch normalization, layer normalization performs exactly the same computation at training and test times.
>> 배치 정규화와 달리 계층 정규화는 훈련 및 테스트 시간에 정확히 동일한 계산을 수행한다.

> It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step.
>> 각 시간 단계에서 정규화 통계를 별도로 계산하여 반복 신경망에 적용하는 것도 간단하다.

> Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. 
>> 계층 정규화는 반복 네트워크에서 숨겨진 상태 역학을 안정화하는 데 매우 효과적이다.

> Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.
>> 경험적으로, 우리는 계층 정규화가 이전에 발표된 기술에 비해 훈련 시간을 크게 줄일 수 있음을 보여준다.

# 1. Introduction

> Deep neural networks trained with some version of Stochastic Gradient Descent have been shown to substantially outperform previous approaches on various supervised learning tasks in computer vision [Krizhevsky et al., 2012] and speech processing [Hinton et al., 2012]. 
>> 일부 버전의 확률적 경사 하강법으로 훈련된 심층 신경망은 컴퓨터 비전과 음성 처리에서 다양한 지도 학습 작업에 대한 이전 접근 방식을 크게 능가하는 것으로 나타났다[Krizhevsky et al., 2012].

> But state-of-the-art deep neural networks often require many days of training. 
>> 그러나 최첨단 심층 신경망은 종종 많은 날의 훈련을 필요로 한다.

> It is possible to speed-up the learning by computing gradients for different subsets of the training cases on different machines or splitting the neural network itself over many machines [Dean et al., 2012], but this can require a lot of communication and complex software. 
>> 서로 다른 기계에서 훈련 사례의 서로 다른 하위 집합에 대한 그레이디언트를 계산하거나 신경망 자체를 많은 기계로 분할하여 학습 속도를 높일 수 있지만, 이는 많은 통신과 복잡한 소프트웨어를 필요로 할 수 있다[Dean et al., 2012].

> It also tends to lead to rapidly diminishing returns as the degree of parallelization increases. An orthogonal approach is to modify the computations performed in the forward pass of the neural net to make learning easier. 
>> 또한 병렬화의 정도가 증가함에 따라 수익률이 급격히 감소하는 경향이 있다. 직교 접근법은 신경망의 전진 패스에서 수행되는 계산을 수정하여 학습을 더 쉽게 만드는 것이다.

> Recently, batch normalization [Ioffe and Szegedy, 2015] has been proposed to reduce training time by including additional normalization stages in deep neural networks. 
>> 최근, 심층 신경망에 추가적인 정규화 단계를 포함하여 훈련 시간을 줄이기 위해 배치 정규화[Ioffe and Szegedy, 2015]가 제안되었다.

> The normalization standardizes each summed input using its mean and its standard deviation across the training data. 
>> 정규화는 훈련 데이터에 대한 평균과 표준 편차를 사용하여 각 합계 입력을 표준화한다.

> Feedforward neural networks trained using batch normalization converge faster even with simple SGD. 
>> 배치 정규화를 사용하여 훈련된 피드포워드 신경망은 간단한 SGD로도 더 빠르게 수렴한다.

> In addition to training time improvement, the stochasticity from the batch statistics serves as a regularizer during training.
>> 훈련 시간 개선 외에도 배치 통계의 확률성은 훈련 중 정규화기 역할을 한다.

> Despite its simplicity, batch normalization requires running averages of the summed input statistics. 
>> 단순함에도 불구하고, 배치 정규화는 합산된 입력 통계의 실행 평균을 필요로 한다.

> In feed-forward networks with fixed depth, it is straightforward to store the statistics separately for each hidden layer. 
>> 깊이가 고정된 피드포워드 네트워크에서는 숨겨진 각 계층에 대해 통계를 별도로 저장하는 것이 간단하다.

> However, the summed inputs to the recurrent neurons in a recurrent neural network (RNN) often vary with the length of the sequence so applying batch normalization to RNNs appears to require different statistics for different time-steps.
>> 그러나 반복 신경망(RNN)에서 반복 뉴런에 대한 합계 입력은 시퀀스의 길이에 따라 달라지는 경우가 많기 때문에 RNN에 배치 정규화를 적용하려면 다양한 시간 단계에 대해 다른 통계가 필요한 것으로 보인다.

> Furthermore, batch normalization cannot be applied to online learning tasks or to extremely large distributed models where the minibatches have to be small.
>> 또한 배치 정규화는 온라인 학습 작업이나 미니 배치가 작아야 하는 극도로 큰 분산 모델에 적용할 수 없다.
 
> This paper introduces layer normalization, a simple  normalization method to improve the training speed for various neural network models. 
>> 본 논문에서는 다양한 신경망 모델에 대한 훈련 속도를 향상시키기 위한 간단한 정규화 방법인 계층 정규화를 소개한다.

> Unlike batch normalization, the proposed method directly estimates the normalization statistics from the summed  inputs to the neurons within a hidden layer so the normalization does not introduce any new dependencies between training cases. 
>> 배치 정규화와 달리 제안된 방법은 숨겨진 계층 내 뉴런에 대한 합계 입력에서 정규화 통계를 직접 추정하여 정규화가 훈련 사례 간에 새로운 종속성을 도입하지 않도록 한다.

> We show that layer normalization works well for RNNs and improves both the training time and the generalization performance of several existing RNN models.
>> 우리는 계층 정규화가 RNN에 잘 작동하고 여러 기존 RNN 모델의 훈련 시간과 일반화 성능을 모두 향상시킨다는 것을 보여준다.

# 3. Layer normalization

> We now consider the layer normalization method which is designed to overcome the drawbacks of batch normalization.
>> 우리는 이제 배치 정규화의 단점을 극복하기 위해 설계된 계층 정규화 방법을 고려한다.

> Notice that changes in the output of one layer will tend to cause highly correlated changes in the summed inputs to the next layer, especially with ReLU units whose outputs can change by a lot.
>> 특히 출력이 많이 변경될 수 있는 ReLU 장치에서 한 계층의 출력 변화는 다음 계층으로 합산된 입력에 높은 상관관계가 있는 변화를 유발하는 경향이 있다는 점에 유의하십시오.

> This suggests the “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer. 
>> 이는 각 레이어 내에서 합산된 입력의 평균과 분산을 고정함으로써 "공변량 이동" 문제를 줄일 수 있음을 시사한다.

> We, thus, compute the layer normalization statistics over all the hidden units in the same layer as follows:
>> 따라서 우리는 다음과 같이 동일한 계층에 있는 모든 숨겨진 단위에 대한 계층 정규화 통계를 계산한다:

$$\mu{}^{l}=\frac{1}{H}\sum^{G}_{i=1}a^{l}_{i}\;\;\;\;\;\;\;\sigma{}^{l}=\sqrt{\frac{1}{H}\sum^{H}_{i=1}(a^{l}_{i}-\mu{}^{l})^{2}}$$

> where H denotes the number of hidden units in a layer. 
>> 여기서 H는 레이어의 숨겨진 단위의 수를 나타낸다.

> The difference between Eq. (2) and Eq. (3) is that under layer normalization, all the hidden units in a layer share the same normalization terms µ and σ, but different training cases have different normalization terms. 
>> Eq. (2)와 Eq. (3)의 차이점은 계층 정규화 하에서, 한 계층의 모든 숨겨진 단위는 동일한 정규화 항 µ와 σ을 공유하지만, 서로 다른 훈련 사례는 다른 정규화 항을 가지고 있다는 것이다.

> Unlike batch normalization, layer normaliztion does not impose any constraint on the size of a mini-batch and it can be used in the pure online regime with batch size 1.
>> 배치 정규화와 달리 계층 정규화는 미니 배치의 크기에 아무런 제약을 가하지 않으며 배치 크기가 1인 순수 온라인 체제에서 사용할 수 있다.
