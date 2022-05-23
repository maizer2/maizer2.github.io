---
layout: post 
title: "(GAN)DCGAN Translation"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review, 1.2.2.5. GAN, 1.2.2.2. CNN]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

$UNSUPERVISED\;REPRESENTATION\;LEARNING$  
$WITH\;DEEP\;CONVOLUTIONAL$  
$GENERATIVE\;ADVERSARIAL\;NETWORKS$  

$ABSTRACT$
> In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications.
>> 최근 몇년간, 합성곱망(CNNs)을 사용한 지도학습은 computer vision applications에서 크게 채택되었다.

> Comparatively, unsupervised learning with CNNs has received less attention.
>> 비교적이게도, CNNs를 사용한 비지도학습은 더 적은 관심을 받았다.

> In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning
>> 이번 일로 우리는 지도 학습과 비지도 학습에 대한 합성곱의 성공의 차이를 해소하는데 도움이 되기를 바란다.
>>> CNN을 적용한 비지도 학습을 성공시킬 것이다.

> We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning.
>> 우리는 깊은 합성곱 생성 적대적 네트워크(DCGANs)라고 불리는 CNNs의 계층을 소개한다. 이는 특정 아키텍처의 제약 조건을 가지며, 비지도 학습의 강력한 후보자임을 입증한다.

> Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator.
>> 다양한 이미지 데이터세트에 대한 학습을 통해, 우리의 심층 합성곱 적대적인 두쌍은 생성기와 판별기가 장면의 객체 요소에서 표현 계층 구조를  학습한다는 것에 설득력있는 증거를 보여준다.

> Additionally, we use the learned features for novel tasks - demonstrating their applicability as general images representations.
>> 또한, 우리는 학습된 특징을 새로운 작업에 사용하여 일반적인 이미지에 적용가능성을 입증한다.

$1. INTRODUCTION$

> Learning reusable feature representations from large unlabeled datasets has been an area of active research.
>> 큰 라벨링되지 않은 데이터 셋으로부터 재사용 가능한 특징 표현을 학습하는 것은 활발한 연구 분야가 되었다.

> In the context of computer vision, one can leverage the practically unlimited amount of unlabeled images and videos to learn good intermediate representations, which can then be used on a variety of supervised learning tasks such as image classification.
>> 컴퓨터 비전의 맥락에서, 라벨링이 되지않은 이미지와 비디오를 실질적으로 무제한 활용하여 좋은 중간 표현을 학습할 수 있으며, 이는 이미지 분류와 같은 다양한 지도 학습에 사용될 수 있다.

> We propose that one way to build good image representations is by training Generative Adversarial Networks(GANs) (Goodfellowet al., 2014), and later reusing parts of the generator and discriminator networks as feature extractors for supervised tasks.
>> 우리는 좋은 이미지표현을 만들어내는 한가지 방법이 GAN을 통해 훈련시키는 것이라고 제안한다, 그 이후 생성기와 판별기 네트워크의 일부를 supervised task을 위한 특징 추출기로 재사용한다.

> GANs provide an attractive alternative to maximum likelihood techniques.
>> GAN은 MLE(Maximum Likelihood) 기술에 대한 매력적인 대안을 제공한다.

> One can additionally argue that their learning process and the lack of a heuristic cost function (such as pixel-wise independent mean-square error) are attractive to representation learning.
>> 학습 프로세스와 heuristic 비용함수(pixel-wise independent MSE 등)의 부재가 표현 학습에 매력적이라고 일반적으로 주장할 수 있다.

> GANs have been known to be unstable to train, often resulting in generators that produce nonsensical outpus.
>> GAN은 훈련이 불안정하다고 알려져 있고, 생성기는 터무니없는 출력을 자주 보여준다.

> There has been very limited published research in trying to understand and visualize what GANs learn, and the intermediate representations of multi-layer GANs.
>> GAN이 학습하는 내용과 다중 계층 GAN의 중간 표현을 이해하고 시각화하기 위해 발표된 연구는 매우 제한적이다.

> In this paper, we make the following contributions
>> 이 논문에서 우리는 다음과 같은 기여를 한다.

* > We propose and evaluate a set of constraints on the architectural topology of Convolutional GANs that make them stable to train in most settings. We name this class of architectures Deep Convolutional GANs(DCGAN)
    >>  우리는 합성곱 GAN의 architectural topology에 대한 제약을 제안하고 평가하여 대부분의 환경에서 안정적으로 훈련한다. 우리는 이 class의 아키텍처를 DCGAN이라고 한다.

* > We used the trained discriminators for image classification tasks, showing competitive performance with other unsupervised algorithms.
    >> 우리는 훈련된 판별기를 이미지 분류에 사용하여 다른 비지도 학습 알고리즘과 경쟁적인 성능을 보여주었다.

* > We visualize the filters learnt by GANs and empirically show that specific filters have learned to draw specific objects.
    >> 우리는 GANS에 의해 학습된 필터를 시각화하고 특정 필터가 특정 객체를 그리는 방법을 학습했음을 경험적으로 보여준다.

* > We show that the generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples.
    >> 우리는 생성자가 생성된 샘플의 많은 의미적 특성을 쉽게 조작할 수 있는 흥미로운 벡터 산술 특성을 가지고 있음을 보여준다.

$2. RELATED\;WORK$
$2.1\;REPRESENTATION\;LEARNING\;FROM\;UNLABELED\;DATA$

> Unsupervised representation learning is a fairly well studied problem in general computer vision research, as well as in the context of images.
>> 비지도 묘사 학습은 일반적인 컴퓨터 비전 연구 문제뿐만 아니라 이미지의 맥락에서 꽤 잘 학습된다.

> A classic approach to unsupervised representation learning is to do clustring on the data (for example using K-means), and leverage the clusters for improved classification scores.
>> 보편적인 비지도 표현 학습의 접근 방법은 데이터를 군집화(예를 들어 k-means를 사용하는 등)하고 영향력 있는 항상된 분류 점수를 개선하기 위해 군집을 활용하는 것이다.

> In the context of images, one can do hierarchical clustering of image patches to learn powerful image representations.
>> 이미지의 문맥에서, 이는 강력한 이미지 표현을 학습하기 위해 이미지 패치의 계층적 군집화를 할 수 있다.

> Another popular method is to train auto-encoders separating the what and where components of the code, ladder structures that encode an image into a compact code, and decode the code to reconstruct the image as accurately as possible.
>> 다른 인기있는 방법은 코드의 what과 where 구성 요소를 분리하는 auto-encoder, 이미지를 컴팩트 코드로 인코딩하는 래더 구조, 코드를 디코딩하여 가능한 정확하게 이미지를 재구성하는 것이다.
> These methods have also been shown to learn good feature representations from image pixels. Deep belief networks (Lee et al., 2009) have also been shown to work well in learning hierarchical representations.
>> 이러한 방법은 또한 이미지 픽셀에서 좋은 특징 표현을 학습하는 것으로 나타났다. 심층 신념 네트워크는 계층적 표현을 학습하는 데도 잘 작동하는 것으로 나타났다.