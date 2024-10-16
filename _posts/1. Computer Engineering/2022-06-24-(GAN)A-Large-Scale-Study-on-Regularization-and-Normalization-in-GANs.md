---
layout: post 
title: "(GAN)A Large-Scale Study on Regularization and Normalization in GANs Translation"
categories: [1. Computer Engineering]
tags: [1.7. Paper Review, 1.2.2.1. Computer Vision]
---

### [CV Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/paper-of-GAN.html)

### [$$\mathbf{A\;Large-Scale\;Study\;on\;Regularization\;and\;Normalization\;in\;GANs}$$](https://arxiv.org/pdf/1807.04720.pdf)

#### $$\mathbf{Karol\;Kurach,\;Mario\;Lucic\;,\;Xiaohua\;Zhai,\;Marcin\;Michalski,\;Sylvain\;Gelly}$$

### $\mathbf{Abstract}$

> Generative adversarial networks (GANs) are a class of deep generative models which aim to learn a target distribution in an unsupervised fashion. While they were successfully applied to many problems, training a GAN is a notoriously challenging task and requires a significant number of hyperparameter tuning, neural architecture engineering, and a non-trivial amount of “tricks”. The success in many practical applications coupled with the lack of a measure to quantify the failure modes of GANs resulted in a plethora of proposed losses, regularization and normalization schemes, as well as neural architectures. In this work we take a sober view of the current state of GANs from a practical perspective. We discuss and evaluate common pitfalls and reproducibility issues, open-source our code on Github, and provide pre-trained models on TensorFlow Hub.
>> 생성적 적대 네트워크(GAN)는 비지도 방식으로 목표 분포를 학습하는 것을 목표로 하는 심층 생성 모델의 클래스입니다. GAN 훈련은 많은 문제에 성공적으로 적용되었지만, GAN 훈련은 악명 높은 어려운 작업이며 상당한 수의 초 매개 변수 조정, 신경 아키텍처 엔지니어링 및 적지 않은 양의 "꼼수"가 필요합니다. GAN의 고장 모드를 정량화하는 척도의 부족과 결합된 많은 실제 응용 프로그램의 성공은 신경 아키텍처뿐만 아니라 과도한 제안된 손실, 정규화 및 정규화 체계를 초래했습니다. 이 작업에서는 실제 관점에서 GAN의 현재 상태를 냉정하게 살펴봅니다. 우리는 일반적인 함정과 재현성 문제에 대해 논의하고 평가하고, 깃허브에서 코드를 공개하며, 텐서플로 허브에서 사전 훈련된 모델을 제공합니다.

### $\mathbf{1.\;Introduction}$

> Deep generative models are a powerful class of (mostly) unsupervised machine learning models. These models were recently applied to great effect in a variety of applications, including image generation, learned compression, and domain adaptation (Brock et al., 2019; Menick & Kalchbrenner, 2019; Karras et al., 2019; Lucic et al., 2019; Isola et al., 2017; Tschannen et al., 2018).
>> 심층 생성 모델은 (대부분) 감독되지 않은 기계 학습 모델의 강력한 클래스입니다. 이러한 모델은 최근 이미지 생성, 학습된 압축 및 도메인 적응을 포함한 다양한 응용 프로그램에 크게 적용되었습니다(Brock et al., 2019; Menick & Kalchbrenner, 2019; Karras et al., 2019; Lucic et al., 2019; Isola et al., 2017; Channen et al., 2018).

> Generative adversarial networks (GANs) (Goodfellow et al., 2014) are one of the main approaches to learning such models in a fully unsupervised fashion. The GAN framework can be viewed as a two-player game where the first player, the generator, is learning to transform some simple input distribution to a complex high-dimensional distribution (e.g. over natural images), such that the second player, the discriminator, cannot tell whether the samples were drawn from the true distribution or were synthesized by the generator. The solution to the classic minimax formulation (Goodfellow et al., 2014) is the Nash equilibrium where neither player can improve unilaterally. As the generator and discriminator are usually parameterized as deep neural networks, this minimax problem is notoriously hard to solve. 
>> 생성적 적대 네트워크(GANs)(Goodfellow et al., 2014)는 이러한 모델을 완전히 감독되지 않은 방식으로 학습하는 주요 접근 방식 중 하나입니다. GAN 프레임워크는 2인용 게임으로 볼 수 있습니다. 첫 번째 플레이어인 생성기는 두 번째 플레이어인 판별자가 샘플이 실제 분포에서 추출되었는지 또는 구문인지 구별할 수 없는 복잡한 고차원 분포(예: 자연 이미지에 대한)로 간단한 입력 분포를 변환하는 방법을 학습합니다.발전기 옆에 있었어요 고전적인 미니맥스 공식(Goodfellow et al., 2014)의 해결책은 두 플레이어 모두 일방적으로 개선할 수 없는 내시 균형입니다. 생성기와 판별기는 일반적으로 심층 신경망으로 매개 변수화되기 때문에, 이 미니맥스 문제는 해결하기 어려운 것으로 악명이 높습니다.

> In practice, the training is performed using stochastic gradient-based optimization methods. Apart from inheriting the optimization challenges associated with training deep neural networks, GAN training is also sensitive to the choice of the loss function optimized by each player, neural network architectures, and the specifics of regularization and normalization schemes applied. This has resulted in a flurry of research focused on addressing these challenges (Goodfellow et al., 2014; Salimans et al., 2016; Miyato et al., 2018; Gulrajani et al., 2017; Arjovsky et al., 2017; Mao et al., 2017).
>> 실제로 훈련은 확률적 그레이디언트 기반 최적화 방법을 사용하여 수행됩니다. GAN 훈련은 심층 신경망 훈련과 관련된 최적화 과제를 상속하는 것 외에도 각 플레이어에 의해 최적화된 손실 함수, 신경망 아키텍처, 적용되는 정규화 및 정규화 체계의 세부 사항의 선택에도 민감합니다. 이로 인해 이러한 과제를 해결하는 데 초점을 맞춘 연구가 쇄도했습니다(Goodfellow et al., 2014; Salimans et al., 2016; Miyato et al., 2018; Gulrajani et al., 2017; Arjovsky et al., 2017; Mao et al., 2017).

> **Our Contributions** In this work we provide a thorough empirical analysis of these competing approaches, and help the researchers and practitioners navigate this space. We first define the GAN landscape – the set of loss functions, normalization and regularization schemes, and the most commonly used architectures. We explore this search space on several modern large-scale datasets by means of hyperparameter optimization, considering both “good” sets of hyperparameters reported in the literature, as well as those obtained by sequential Bayesian optimization.
>> 우리의 기여 이 연구에서 우리는 이러한 경쟁적 접근법에 대한 철저한 경험적 분석을 제공하고 연구자와 실무자가 이 공간을 탐색할 수 있도록 도와줍니다. 먼저 손실 함수의 집합, 정규화 및 정규화 체계 및 가장 일반적으로 사용되는 아키텍처인 GAN 환경을 정의합니다. 우리는 문헌에 보고된 "좋은" 하이퍼 파라미터 세트와 순차 베이지안 최적화를 통해 얻은 하이퍼 파라미터 세트를 모두 고려하여 하이퍼 파라미터 최적화를 통해 여러 현대 대규모 데이터 세트에서 이 검색 공간을 탐색합니다.

> We first decompose the effect of various normalization and regularization schemes. We show that both gradient penalty (Gulrajani et al., 2017) as well as spectral normalization (Miyato et al., 2018) are useful in the context of high-capacity architectures. Then, by analyzing the impact of the loss function, we conclude that the non-saturating loss (Goodfellow et al., 2014) is sufficiently stable across datasets and hyperparameters. Finally, show that similar conclusions hold for both popular types of neural architectures used in state-of-the-art models. We then discuss some common pitfalls, reproducibility issues, and practical considerations. We provide reference implementations, including training and evaluation code on Github<sup><a href="https://github.com/google/compare_gan">1</a></sup> , and provide pre-trained models on TensorFlow Hub<sup><a href="https://www.tensorflow.org/hub">2</a></sup>
>> 우리는 먼저 다양한 정규화 및 정규화 계획의 효과를 분해합니다. 우리는 그레이디언트 패널티(Gulrajani et al., 2017)와 스펙트럼 정규화(Miyato et al., 2018)가 고용량 아키텍처의 맥락에서 유용함을 보여준다. 그런 다음 손실 함수의 영향을 분석하여 비포화 손실(Goodfellow et al., 2014)이 데이터 세트와 하이퍼 파라미터에 걸쳐 충분히 안정적이라는 결론을 내립니다. 마지막으로, 유사한 결론이 최첨단 모델에 사용되는 인기 있는 유형의 신경 아키텍처 두 가지 모두에 대해 유지된다는 것을 보여 줍니다. 그런 다음 몇 가지 일반적인 함정, 재현성 문제 및 실제 고려 사항에 대해 논의합니다. 우리는 Github<sup><ahref="https://github.com/google/compare_gan">1</a></sup>에 대한 교육 및 평가 코드를 포함한 참조 구현을 제공하고 TensorFlow Hub<sup><ahref="https://www.tensorflow.org/hub">2</a>에 대한 사전 훈련된 모델을 제공합니다.

### $\mathbf{2.\;The\;GAN\;Landscape}$

> The main design choices in GANs are the loss function, regularization and/or normalization approaches, and the neural architectures. At this point GANs are extremely sensitive to these design choices. This fact coupled with optimization issues and hyperparameter sensitivity makes GANs hard to apply to new datasets. Here we detail the main design choices which are investigated in this work.
>> GAN의 주요 설계 선택은 손실 함수, 정규화 및/또는 정규화 접근법 및 신경 아키텍처입니다. 이 시점에서 GAN은 이러한 설계 선택에 매우 민감합니다. 이러한 사실은 최적화 문제 및 초 매개 변수 민감도와 결합되어 GAN을 새로운 데이터 세트에 적용하기가 어렵습니다. 여기서는 이 작업에서 조사되는 주요 설계 선택에 대해 자세히 설명합니다.

#### $\mathbf{2.1.\;Loss\;Functions}$

> Let $P$ denote the target (true) distribution and $Q$ the model distribution. Goodfellow et al.(2014) suggest two loss functions: the minimax GAN and the non-saturating (NS) GAN. In the former the discriminator minimizes the negative loglikelihood for the binary classification task. In the latter the generator maximizes the probability of generated samples being real. In this work we consider the non-saturating loss as it is known to outperform the minimax variant empirically. The corresponding discriminator and generator loss functions are
>> $P$는 목표값(참) 분포를 나타내고 $Q$는 모형 분포를 나타냅니다. Goodfellow 외 연구진(2014)은 미니맥스 GAN과 비포화(NS) GAN의 두 가지 손실 함수를 제안합니다. 전자의 경우 판별기는 이진 분류 작업에 대한 음의 로그 가능성을 최소화합니다. 후자에서 생성자는 생성된 샘플이 실제일 확률을 최대화합니다. 이 연구에서 우리는 미미맥스 변형을 경험적으로 능가하는 것으로 알려져 있기 때문에 비포화 손실을 고려합니다. 해당 판별기 및 제너레이터 손실 함수는 다음과 같습니다.

$$L_{D}=-E_{x\sim{P}}[\log{(D(x))}]-E_{\hat{x}\sim{Q}}[\log{(1-D(\hat{x}))}],$$

$$L_{G}=-E_{\hat{x}\sim{Q}}[\log{(D(\hat{x}))}],\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

> where $D(x)$ denotes the probability of $x$ being sampled from $P$. In Wasserstein GAN (WGAN) (Arjovsky et al., 2017) the authors propose to consider the Wasserstein distance instead of the Jensen-Shannon (JS) divergence. The corresponding loss functions are
>> 여기서 $D(x)$ 는 $x$ 가 $P$에서 샘플링될 확률을 나타냅니다. WGAN(Wasserstein GAN)(Arjovsky et al., 2017)에서 저자들은 Jensen-Shannon(JS) 발산 대신 Wasserstein 거리를 고려할 것을 제안합니다. 해당 손실 함수는 다음과 같습니다.

$$L_{D}=-E_{x\sim{P}}[D(x)]+E_{\hat{x}\sim{Q}}[D(\hat{x})],$$

$$L_{G}=-E_{\hat{x}\sim{Q}}[D(\hat{x})],\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

> where the discriminator output $D(x)\in{}R$ and $D$ is required to be 1-Lipschitz. Under the optimal discriminator, minimizing the proposed loss function with respect to the generator minimizes the Wasserstein distance between $P$ and $Q$. A key challenge is ensure the Lipschitzness of $D$. Finally, we consider the least-squares loss (LS) which corresponds to minimizing the Pearson $X^{2}$ divergence between $P$and $Q$(Mao et al., 2017). The corresponding loss functions are
>> 여기서 판별기 출력 $D(x)\in{}R$ 및 $D$는 1-Lipschitz 여야 합니다. 최적 판별기에서 생성기와 관련하여 제안된 손실 함수를 최소화하면 $P$와 $Q$ 사이의 와서스테인 거리를 최소화할 수 있습니다. 핵심 과제는 $D$의 Lipschitz을 보장하는 것입니다. 마지막으로, $P$와 $Q$ 사이의 피어슨 $X^{2}$ 분산을 최소화하는 것에 해당하는 최소 제곱 손실(LS)을 고려한다(Mao et al., 2017). 해당 손실 함수는 다음과 같습니다.

$$L_{D}=-E_{x\sim{P}}[(D(x)-1)^{2}]+E_{\hat{x}\sim{Q}}[D(\hat{x})^{2}],$$

$$L_{G}=-E_{\hat{x}\sim{Q}}[(D(\hat{x})-1)^{2}],\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

> where $D(x)$\in{}R$ is the output of the discriminator. Intuitively, this loss smooth loss function saturates slower than the cross-entropy loss.
>> 여기서 $D(x)$\in{}R$는 판별기의 출력입니다. 직관적으로 이 손실 평활 손실 함수는 교차 엔트로피 손실보다 포화 속도가 느립니다.

#### $\mathbf{2.2.\;Regularization\;and\;Normalization}$

> **Gradient Norm Penalty** The idea is to regularize $D$ by constraining the norms of its gradients (e.g. $L^{2}$). In the context of Wasserstein GANs and optimal transport this regularizer arises naturally and the gradient norm is evaluated on the points from the optimal coupling between samples from $P$and $Q$(GP) (Gulrajani et al., 2017). Computing this coupling during GAN training is computationally intensive, and a linear interpolation between these samples is used instead. The gradient norm can also be penalized close to the data manifold which encourages the discriminator to be piece-wise linear in that region (Dragan) (Kodali et al., 2017). A drawback of gradient penalty (GP) regularization scheme is that it can depend on the model distribution $Q$which changes during training. For Dragan it is unclear to which extent the Gaussian assumption for the manifold holds. In both cases, computing the gradient norms implies a non-trivial running time overhead.
>> **그라디언트 노름 패널티** 이 아이디어는 그레이디언트의 규범(예: $L^{2}$)을 제한하여 $D$를 정규화하는 것입니다. Wasserstein GANs 및 최적 전송의 맥락에서 이 정규화기는 자연스럽게 발생하며 $P$와 $Q$(GP)의 샘플 사이의 최적 결합에서 나온 지점에서 그레이디언트 규범을 평가합니다(Gulrajani et al., 2017). GAN 훈련 중에 이 커플링을 계산하는 것은 계산 집약적이며, 대신 이러한 샘플 간의 선형 보간이 사용됩니다. 그레이디언트 규범은 또한 데이터 매니폴드에 가깝게 처벌될 수 있으며, 이는 판별자가 해당 영역에서 부분적인 선형적이 되도록 장려합니다(Dragan). (Kodali et al., 2017) 그레이디언트 패널티(GP) 정규화 체계의 단점은 훈련 중에 변경되는 모델 분포 $Q$에 의존할 수 있다는 것입니다. Dragan의 경우 매니폴드에 대한 가우스 가정이 어느 정도까지 유지되는지 알 수 없습니다. 두 경우 모두 그레이디언트 규범을 계산한다는 것은 사소한 실행 시간 오버헤드가 아니라는 것을 의미합니다.

> Notwithstanding these natural interpretations for specific losses, one may also consider the gradient norm penalty as a classic regularizer for the complexity of the discriminator (Fedus et al., 2018). To this end we also investigate the impact of a $L^{2}$ regularization on $D$ which is ubiquitous in supervised learning.
>> 특정 손실에 대한 이러한 자연스러운 해석에도 불구하고, 판별자의 복잡성에 대한 고전적인 정규화로서 기울기 규범 페널티를 고려할 수 있습니다(Fedus et al., 2018). 이를 위해 우리는 또한 지도 학습에서 어디서나 볼 수 있는 $L^{2}$ 정규화가 $D$에 미치는 영향을 조사한다.

> **Discriminator Normalization** Normalizing the discriminator can be useful from both the optimization perspective (more efficient gradient flow, more stable optimization), as well as from the representation perspective – the representation richness of the layers in a neural network depends on the spectral structure of the corresponding weight matrices (Miyato et al., 2018).
>> **판별기 정규화** 판별기를 정규화하는 것은 최적화 관점(더 효율적인 그레이디언트 흐름, 더 안정적인 최적화)뿐만 아니라 표현 관점에서도 유용할 수 있습니다. 신경 네트워크에서 계층의 풍부한 표현은 해당 가중치 행렬의 스펙트럼 구조에 따라 달라진다(Miyato et al., 2018).

> From the optimization point of view, several normalization techniques commonly applied to deep neural network training have been applied to GANs, namely batch normalization (BN) (Ioffe & Szegedy, 2015) and layer normalization (LN) (Ba et al., 2016). The former was explored in Denton et al.(2015) and further popularized by Radford et al.(2016), while the latter was investigated in Gulrajani et al.(2017). These techniques are used to normalize the activations, either across the batch (BN), or across features (LN), both of which were observed to improve the empirical performance. From the representation point of view, one may consider the neural network as a composition of (possibly non-linear) mappings and analyze their spectral properties. In particular, for the discriminator to be a bounded operator it suffices to control the operator norm of each mapping. This approach is followed in Miyato et al.(2018) where the authors suggest dividing each weight matrix, including the matrices representing convolutional kernels, by their spectral norm. It is argued that spectral normalization results in discriminators of higher rank with respect to the competing approaches.
>> 최적화 관점에서, 심층 신경망 훈련에 일반적으로 적용되는 몇 가지 정규화 기법, 즉 배치 정규화(BN)(Ioffe & Szegdy, 2015)와 계층 정규화(LN)(Ba et al., 2016)가 GAN에 적용되었습니다. 전자는 Denton 등(2015)에서 탐색되었고, Radford 등(2016)에 의해 더욱 대중화되었으며, 후자는 Gulrajani 등(2017)에서 조사되었습니다. 이러한 기술은 일괄 처리(BN) 또는 기능(LN)을 통해 활성화를 정규화하는 데 사용되며, 두 가지 모두 경험적 성능을 향상시키는 것으로 관찰되었습니다. 표현의 관점에서, 신경망은 (비선형일 가능성이 있는) 매핑의 구성으로 간주하고 스펙트럼 특성을 분석할 수 있습니다. 특히 판별기가 유계 연산자일 경우 각 매핑의 연산자 규범을 제어하기에 충분합니다. 이 접근 방식은 저자들이 컨볼루션 커널을 나타내는 행렬을 포함한 각 가중치 행렬을 스펙트럼 규범으로 나눌 것을 제안하는 미야토 외(2018)에서 따랐습니다. 스펙트럼 정규화는 경쟁 접근 방식과 관련하여 더 높은 등급의 판별자를 발생시킨다는 주장이 있습니다.

#### $\mathbf{2.3.\;Generator\;and\;Discriminator\;Architecture}$

> We explore two classes of architectures in this study: deep convolutional generative adversarial networks (DCGAN) (Radford et al., 2016) and residual networks (ResNet) (He et al., 2016), both of which are ubiquitous in GAN research. Recently, Miyato et al.(2018) defined a variation of DCGAN, so called SNDCGAN. Apart from minor updates (cf. Section 4) the main difference to DCGAN is the use of an eight-layer discriminator network. The details of both networks are summarized in Table 4. The other architecture, ResNet19, is an architecture with five ResNet blocks in the generator and six ResNet blocks in the discriminator, that can operate on 128 × 128 images. We follow the ResNet setup from Miyato et al.(2018), with the small difference that we simplified the design of the discriminator.
>> 우리는 이 연구에서 심층 컨볼루션 생성 적대적 네트워크(DCGAN) (Radford et al., 2016)와 잔류 네트워크(ResNet)라는 두 가지 종류의 아키텍처를 탐구하는데, 이 두 가지 유형은 모두 GAN 연구에서 유비쿼터스하다. 최근, 미야토 외 연구진(2018)은 사소한 업데이트(cf)와는 별도로 DCGAN의 변형, 이른바 SNDCGAN을 정의했습니다. 섹션 4) DCGAN과의 주요 차이점은 8계층 판별기 네트워크의 사용입니다. 두 네트워크의 세부 사항은 표 4에 요약되어 있습니다. 다른 아키텍처인 ResNet19는 128 × 128 이미지에서 작동할 수 있는 생성기에 5개의 ResNet 블록과 판별기에 6개의 ResNet 블록이 있는 아키텍처입니다. 우리는 미야토 외(2018)의 ResNet 설정을 따르는데, 판별기의 설계를 단순화했다는 작은 차이가 있습니다.

> The architecture details are summarized in Table 5a and Table 5b. With this setup we were able to reproduce the results in Miyato et al.(2018). An ablation study on various ResNet modifications is available in the Appendix.
>> 아키텍처에 대한 자세한 내용은 표 5a 및 표 5b에 요약되어 있습니다. 이 설정을 통해 Miyato et al.(2018)에서 결과를 재현할 수 있었습니다. 다양한 ResNet 수정에 대한 절제 스터디는 부록에서 사용할 수 있습니다.

#### $\mathbf{2.4.\;Evaluation\;Metrics}$

> We focus on several recently proposed metrics well suited to the image domain. For an in-depth overview of quantitative metrics we refer the reader to Borji (2019).
>> 우리는 이미지 도메인에 잘 맞는 최근 제안된 몇 가지 메트릭에 중점을 둡니다. 정량적 지표에 대한 심층적인 개요를 보려면 독자에게 Borji(2019)를 참조하라고 합니다.

> **Inception Score ($IS$)** Proposed by Salimans et al.(2016), the $IS$ offers a way to quantitatively evaluate the quality of generated samples. Intuitively, the conditional label distribution of samples containing meaningful objects should have low entropy, and the variability of the samples should be high. which can be expressed as $IS=exp_{Ex\sim{}Q}[d_{KL}(p(y\vert{}x),p(y))])$. The authors found that this score is well-correlated with scores from human annotators. Drawbacks include insensitivity to the prior distribution over labels and not being a proper distance.
>> **Salimans 등이 제안한 인셉션 점수($IS$)** $IS$는 생성된 샘플의 품질을 정량적으로 평가하는 방법을 제공합니다. 직관적으로 유의한 개체를 포함하는 표본의 조건부 레이블 분포는 엔트로피가 낮아야 하며 표본의 변동성이 높아야 합니다. $IS=exp_{Ex\sim{}Q}[d_{KL}(p(y\vert{}x),p(y))])$로 표현할 수 있습니다. 저자들은 이 점수가 인간 주석자의 점수와 잘 연관되어 있다는 것을 발견했습니다. 단점으로는 레이블에 대한 이전 분포에 대한 무감각하고 적절한 거리가 되지 않는 것이 있습니다.

> **Frechet Inception Distance (FID)** In this approach proposed by Heusel et al.(2017) samples from $P$and $Q$ are first embedded into a feature space (a specific layer of InceptionNet). Then, assuming that the embedded data follows a multivariate Gaussian distribution, the mean and covariance are estimated. Finally, the Frechet distance between these two Gaussians is computed, i.e.
>> **Frechet 인셉션 거리(FID)** 휴젤 외 연구진(2017)이 제안한 이 접근 방식에서 $P$ 및 $Q$의 샘플은 먼저 특징 공간(InceptionNet의 특정 계층)에 내장됩니다. 그런 다음 내장된 데이터가 다변량 가우스 분포를 따른다고 가정하면 평균과 공분산이 추정됩니다. 마지막으로, 이 두 가우스 사이의 Frechet 거리를 계산합니다.

![](https://latex.codecogs.com/svg.image?FID=%7C%7C%5Cmu_%7Bx%7D-%5Cmu_%7By%7D%7C%7C_%7B2%7D%5E%7B2%7D&plus;Tr(%5Csum_%7Bx%7D&plus;%5Csum_%7By%7D-2(%5Csum_%7B2%7D%5Csum_%7By%7D)%5E%7B%5Cfrac%7B1%7D%7B2%7D%7D))

> where ($\mu_{x},\sum_{x}$), and ($\mu_{y},\sum_{y}$) are the mean and covariance of the embedded samples from $P$ and $Q$, respectively. The authors argue that FID is consistent with human judgment and more robust to noise than $IS$. Furthermore, the score is sensitive to the visual quality of generated samples – introducing noise or artifacts in the generated samples will reduce the FID. In contrast to $IS$, FID can detect intra-class mode dropping – a model that generates only one image per class will have a good $IS$, but a bad FID (Lucic et al., 2018).
>> 여기서 ($\mu_{x},\sum_{x}$) 및 ($\mu_{y},\sum_{y}$)는 각각 $P$ 및 $Q$의 포함된 샘플의 평균 및 공분산입니다. 저자들은 FID가 인간의 판단과 일치하고 $IS$보다 소음에 더 강하다고 주장합니다. 또한 점수는 생성된 샘플의 시각적 품질에 민감합니다. 생성된 샘플에 노이즈 또는 아티팩트를 도입하면 FID가 감소합니다. $IS$와 대조적으로, FID는 클래스 내 모드 드롭을 감지할 수 있습니다. 클래스당 하나의 이미지만 생성하는 모델은 $IS$는 좋지만 FID는 좋지 않습니다(Lucic et al., 2018).

> **Kernel Inception Distance (KID)** Binkowski et al.(2018) argue that FID has no unbiased estimator and suggest KID as an unbiased alternative. In Appendix B we empirically compare KID to FID and observe that both metrics are very strongly correlated (Spearman rank-order correlation coefficient of 0.994 for LSUN-BEDROOM and 0.995 for CELEBA-HQ-128 datasets). As a result we focus on FID as it is likely to result in the same ranking.
>> **Kernel Inception Distance(KID)** Binkowski et al.(2018)는 FID에 편향되지 않은 추정기가 없다고 주장하며 KID를 편향되지 않은 대안으로 제안합니다. 부록 B에서 우리는 KID와 FID를 경험적으로 비교하고 두 메트릭이 매우 강한 상관 관계를 관찰합니다(LSUN-BROME의 경우 스피어맨 순위 상관 계수 0.994, CELEBA-HQ-128 데이터 세트의 경우 0.995). 결과적으로 동일한 순위가 될 가능성이 높기 때문에 우리는 FID에 초점을 맞춥니다.

#### $\mathbf{2.5.\;Datasets}$

> We consider three datasets, namely CIFAR10, CELEBA-HQ128, and LSUN-BEDROOM. The LSUN-BEDROOM dataset contains slightly more than 3 million images (Yu et al., 2015).<sup>The images are preprocessed to 128 ×128 × 3 using TensorFlow resize image with crop or pad.
</sup> We randomly partition the images into a train and test set whereby we use 30588 images as the test set. Secondly, we use the CELEBA-HQ dataset of 30K images (Karras et al., 2018). We use the 128×128×3 version obtained by running the code provided by the authors.<sup><a href="https://github.com/tkarras/progressive_growing_of_gans">4</a></sup> We use 3K examples as the test set and the remaining examples as the training set. Finally, we also include the CIFAR10 dataset which contains 70K images (32×32×3), partitioned into 60K training instances and 10K testing instances. The baseline FID scores are 12.6 for CELEBA-HQ-128, 3.8 for LSUN-BEDROOM, and 5.19 for CIFAR10. Details on FID computation are presented in Section 4.
>> 우리는 세 가지 데이터 세트, 즉 CIFAR10, CELEBA-HQ128 및 LSUN-BROCHEDION을 고려합니다. LSUN-BRODE 데이터 세트에는 300만 개 이상의 이미지가 포함되어 있습니다(Yu et al., 2015). <sup>이미지는 자르기 또는 패드가 있는 TensorFlow 크기 조정 이미지를 사용하여 128 × 128 × 3으로 사전 처리됩니다. </sup> 이미지를 무작위로 열차와 테스트 세트로 분할하여 30588개의 이미지를 테스트 세트로 사용합니다. 둘째, 30K 이미지의 CELEBA-HQ 데이터 세트를 사용합니다(Karas et al., 2018). 우리는 작성자가 제공한 코드를 실행한 128×128×3 버전을 사용합니다.<sup><a href="https://github.com/tkarras/progressive_growing_of_gans">4</a></sup> 우리는 3K 예제를 테스트 세트로 사용하고 나머지 예제를 교육 세트로 사용합니다. 마지막으로, 60K 교육 인스턴스와 10K 테스트 인스턴스로 분할된 70K 이미지(32x32x3)를 포함하는 CIFAR10 데이터 세트도 포함합니다. 기본 FID 점수는 CELEBA-HQ-128의 경우 12.6점, LSUN-BHOME의 경우 3.8점, CIFAR10의 경우 5.19점입니다. FID 연산에 대한 자세한 내용은 섹션 4에 나와 있습니다.

#### $\mathbf{2.6.\;Exploring\;the\;GAN\;Landscape}$

> The search space for GANs is prohibitively large: exploring all combinations of all losses, normalization and regularization schemes, and architectures is outside of the practical realm. Instead, in this study we analyze several slices of this search space for each dataset. In particular, to ensure that we can reproduce existing results, we perform a study over the subset of this search space on CIFAR10. We then proceed to analyze the performance of these models across CELEBA-HQ-128 and LSUN-BEDROOM. In Section 3.1 we fix everything but the regularization and normalization scheme. In Section 3.2 we fix everything but the loss. Finally, in Section 3.3 we fix everything but the architecture. This allows us to decouple some of these design choices and provide some insight on what matters most in practice.
>> 모든 손실, 정규화 및 정규화 체계 및 아키텍처의 모든 조합을 탐색하는 GAN의 검색 공간은 매우 큽니다. 대신, 이 연구에서는 각 데이터 세트에 대해 이 검색 공간의 여러 조각을 분석합니다. 특히, 기존 결과를 재현할 수 있도록 CIFAR10에서 이 검색 공간의 하위 집합에 대한 연구를 수행합니다. 그런 다음 CELEBA-HQ-128과 LSUN-BHOME에서 이러한 모델의 성능을 분석합니다. 3.1절에서는 정규화 및 정규화 체계를 제외한 모든 것을 수정합니다. 섹션 3.2에서는 손실을 제외한 모든 것을 수정합니다. 마지막으로, 섹션 3.3에서는 아키텍처를 제외한 모든 것을 수정합니다. 이를 통해 이러한 설계 선택 사항 중 일부를 분리하고 실제로 가장 중요한 사항에 대한 통찰력을 얻을 수 있습니다.

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)A-Large-Scale-Study-on-Regularization-and-Normalization-in-GANs/Figure-1.JPG)

> Figure 1. Plots in the first row show the FID distribution for top 5% models (lower is better). We observe that both gradient penalty (GP) and spectral normalization (SN) outperform the non-regularized/normalized baseline (W/O). Unfortunately, none fully address the stability issues. The second row shows the estimate of the minimum FID achievable for a given computational budget. For example, to otain an FID below 100 using non-saturating loss with gradient penalty, we need to try at least 6 hyperparameter settings. At the same time, we could achieve a better result (lower FID) with spectral normalization and 2 hyperparameter settings. These results suggest that spectral norm is a better practical choice.
>> 그림 1입니다. 첫 번째 행의 그림은 상위 5% 모형에 대한 FID 분포를 보여 줍니다(낮을수록 더 좋습니다). 우리는 그레이디언트 패널티(GP)와 스펙트럼 정규화(SN)가 모두 비정규화/정규화 기준선(W/O)을 능가한다는 것을 관찰합니다. 불행하게도, 어느 것도 안정성 문제를 완전히 해결하지는 못합니다. 두 번째 행은 주어진 계산 예산에 대해 달성할 수 있는 최소 FID의 추정치를 보여줍니다. 예를 들어, 그레이디언트 패널티가 있는 불포화 손실을 사용하여 FID를 100 미만으로 얻으려면 최소 6개의 하이퍼 파라미터 설정을 시도해야 합니다. 동시에 스펙트럼 정규화와 2개의 하이퍼 파라미터 설정을 통해 더 나은 결과(FID가 낮음)를 달성할 수 있었습니다. 이러한 결과는 스펙트럼 노름이 더 나은 실제 선택임을 나타냅니다.

![Table 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)A-Large-Scale-Study-on-Regularization-and-Normalization-in-GANs/Table-1.JPG)

> Table 1. Hyperparameter ranges used in this study. The Cartesian product of the fixed values suffices to uncover most of the recent results from the literature.
>> 표 1입니다. 이 스터디에 사용된 하이퍼 파라미터 범위입니다. 고정 값의 데카르트 곱은 문헌에서 최근 결과의 대부분을 확인하기에 충분합니다.

![Table 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)A-Large-Scale-Study-on-Regularization-and-Normalization-in-GANs/Table-2.JPG)

> Table 2. We use sequential Bayesian optimization (Srinivas et al., 2010) to explore the hyperparameter settings from the specified ranges. We explore 120 hyperparameter settings in 12 rounds of optimization.
>> 표 2입니다. 순차 베이지안 최적화(Srinivas et al., 2010)를 사용하여 지정된 범위의 하이퍼 파라미터 설정을 탐색합니다. 우리는 12번의 최적화에서 120개의 하이퍼 파라미터 설정을 탐구합니다.

> As noted by Lucic et al.(2018), one major issue preventing further progress is the hyperparameter tuning – currently, the community has converged to a small set of parameter values which work on some datasets, and may completely fail on others. In this study we combine the best hyperparameter settings found in the literature (Miyato et al., 2018), and perform sequential Bayesian optimization (Srinivas et al., 2010) to possibly uncover better hyperparameter settings. In a nutshell, in sequential Bayesian optimization one starts by evaluating a set of hyperparameter settings (possibly chosen randomly). Then, based on the obtained scores for these hyperparameters the next set of hyperparameter combinations is chosen such to balance the exploration (finding new hyperparameter settings which might perform well) and exploitation (selecting settings close to the best-performing settings). We then consider the top performing models and discuss the impact of the computational budget.
>> Lucic 외 연구진(2018)이 지적한 바와 같이, 추가 진행을 방해하는 한 가지 주요 문제는 하이퍼 파라미터 조정입니다. 현재 커뮤니티는 일부 데이터 세트에서 작동하며 다른 데이터 세트에서 완전히 실패할 수 있는 작은 매개 변수 값으로 수렴되었습니다. 본 연구에서는 문헌에서 발견된 최상의 하이퍼 파라미터 설정을 결합하고(Miyato et al., 2018), 순차 베이지안 최적화(Srinivas et al., 2010)를 수행하여 더 나은 하이퍼 파라미터 설정을 발견할 수 있다. 간단히 말해서, 순차적 베이지안 최적화에서는 하이퍼 파라미터 설정 집합을 평가하는 것으로 시작합니다(아마도 무작위로 선택). 그런 다음, 이러한 하이퍼 파라미터에 대해 얻은 점수를 기반으로 다음 하이퍼 파라미터 조합 집합을 선택하여 탐색(잘 수행될 수 있는 새 하이퍼 파라미터 설정 찾기)과 착취(가장 성능이 좋은 설정에 가까운 설정 선택)의 균형을 조정합니다. 그런 다음 최고 성능 모델을 고려하고 계산 예산의 영향에 대해 논의합니다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)A-Large-Scale-Study-on-Regularization-and-Normalization-in-GANs/Figure-2.JPG)

> Figure 2. The first row shows the FID distribution for top 5% models. We compare the non-saturating (NS) loss, the Wasserstein loss (WGAN), and the least-squares loss (LS), combined with the most prominent regularization and normalization strategies, namely spectral norm (SN) and gradient penalty (GP). We observe that spectral norm consistently improves the sample quality. In some cases the gradient penalty can help, but there is no clear trend. From the computational budget perspective one can attain lower levels of FID with fewer hyperparameter optimization settings which demonstrates the practical advantages of spectral normalization over competing method. 
>> 그림 2입니다. 첫 번째 행은 상위 5% 모형에 대한 FID 분포를 보여 줍니다. 우리는 가장 두드러진 정규화 및 정규화 전략, 즉 스펙트럼 노름(SN) 및 그레이디언트 패널티(GP)와 결합된 비포화성(NS), 와서스테인 손실(WGAN) 및 최소 제곱 손실(LS)을 비교합니다. 우리는 스펙트럼 노름이 샘플 품질을 지속적으로 향상시킨다는 것을 관찰합니다. 어떤 경우에는 그라데이션 패널티가 도움이 될 수 있지만 뚜렷한 추세는 없습니다. 계산 예산 관점에서, 경쟁 방법에 비해 스펙트럼 정규화의 실질적인 이점을 보여주는 더 적은 수의 하이퍼 파라미터 최적화 설정으로 더 낮은 수준의 FID를 달성할 수 있습니다.

> We summarize the fixed hyperparameter settings in Table 1 which contains the “good” parameters reported in recent publications (Fedus et al., 2018; Miyato et al., 2018; Gulrajani et al., 2017). In particular, we consider the Cartesian product of these parameters to obtain 24 hyperparameter settings to reduce the survivorship bias. Finally, to provide a fair comparison, we perform sequential Bayesian optimization (Srinivas et al., 2010) on the parameter ranges provided in Table 2. We run 12 rounds (i.e. we communicate with the oracle 12 times) of sequential optimization, each with a batch of 10 hyperparameter sets selected based on the FID scores from the results of the previous iterations. As we explore the number of discriminator updates per generator update (1 or 5), this leads to an additional 240 hyperparameter settings which in some cases outperform the previously known hyperparameter settings. The batch size is set to 64 for all the experiments. We use a fixed the number of discriminator update steps of 100K for LSUN-BEDROOM dataset and CELEBA-HQ-128 dataset, and 200K for CIFAR10 dataset. We apply the Adam optimizer (Kingma & Ba, 2015).
>> 우리는 최근 출판물에서 보고된 "좋은" 매개 변수를 포함하는 표 1의 고정 하이퍼 매개 변수 설정을 요약합니다(Fedus et al., 2018; Miyato et al., 2018; Gulrajani et al., 2017). 특히, 우리는 생존 편향을 줄이기 위해 24개의 초 매개 변수 설정을 얻기 위해 이러한 매개 변수의 데카르트 곱을 고려합니다. 마지막으로, 공정한 비교를 제공하기 위해, 우리는 표 2에 제공된 매개 변수 범위에 대해 순차적 베이지안 최적화(Srinivase et al., 2010)를 수행합니다. 우리는 순차 최적화의 12라운드(즉, 오라클과 12번 통신)를 실행하며, 각 라운드는 이전 반복의 결과에서 FID 점수를 기반으로 선택된 10개의 하이퍼 파라미터 세트 배치를 가지고 있다. 제너레이터 업데이트당 판별기 업데이트 수(1 또는 5)를 조사할 때, 이는 이전에 알려진 하이퍼 파라미터 설정보다 성능이 우수한 240개의 하이퍼 파라미터 설정으로 이어집니다. 배치 크기는 모든 실험에 대해 64로 설정됩니다. LSUN-BEADOM 데이터 세트와 CELEBA-HQ-128 데이터 세트에 대해 100K, CIFAR10 데이터 세트에 대해 200K의 고정 판별기 업데이트 단계를 사용합니다. Adam optimizer를 적용합니다(Kingma & Ba, 2015).

### $\mathbf{3.\;Experimental\;Results\;and\;Discussion}$

> Given that there are 4 major components (loss, architecture, regularization, normalization) to analyze for each dataset, it is infeasible to explore the whole landscape. Hence, we opt for a more pragmatic solution – we keep some dimensions fixed, and vary the others. We highlight two aspects:
>> 각 데이터셋에 대해 분석할 4가지 주요 구성 요소(손실, 아키텍처, 정규화, 정규화)가 있으므로 전체 환경을 탐색하는 것은 불가능합니다. 따라서, 우리는 보다 실용적인 솔루션을 선택합니다. 즉, 일부 차원을 고정하고 다른 차원을 다양화합니다. 두 가지 측면을 강조합니다.

1. > We train the models using various hyperparameter settings, both predefined and ones obtained by sequential Bayesian optimization. Then we compute the FID distribution of the top 5% of the trained models. The lower the median FID, the better the model. The lower the variance, the more stable the model is from the optimization point of view.
    >> 사전 정의된 설정과 순차 베이지안 최적화를 통해 얻은 설정 모두 다양한 하이퍼 매개 변수 설정을 사용하여 모델을 훈련합니다. 그런 다음 훈련된 모델의 상위 5%에 대한 FID 분포를 계산한다. 중위수 FID가 낮을수록 모형이 더 우수합니다. 분산이 낮을수록 최적화 관점에서 모형이 더 안정적입니다.

2. > The tradeoff between the computational budget (for training) and model quality in terms of FID. Intuitively, given a limited computational budget (being able to train only $k$ different models), which model should one choose? Clearly, models which achieve better performance using the same computational budget should be preferred in practice. To compute the minimum attainable FID for a fixed budget $k$ we simulate a practitioner attempting to find a good hyperparameter setting for their model: we spend a part of the budget on the “ood” hyperparameter settings reported in recent publications, followed by exploring new settings (i.e. using Bayesian optimization). As this is a random process, we repeat it 1000 times and report the average of the minimum attainable FID.
    >> FID 측면에서 (교육을 위한) 계산 예산과 모델 품질 사이의 균형입니다. 직관적으로 제한된 계산 예산($k$개의 다른 모델만 훈련할 수 있음)을 고려할 때 어떤 모델을 선택해야 합니까? 실제로 동일한 계산 예산을 사용하여 더 나은 성능을 달성하는 모델이 선호되어야 합니다. 고정 예산 $k$에 대해 달성 가능한 최소 FID를 계산하기 위해 우리는 모델에 대한 좋은 하이퍼 매개 변수 설정을 찾으려는 실무자를 시뮬레이션한다. 우리는 최근 출판물에 보고된 “ood” 하이퍼 매개 변수 설정에 예산의 일부를 쓴 다음 새로운 설정(예: 베이지안 최적화 사용)을 탐구한다. 랜덤 프로세스이므로 1000회 반복하고 최소 달성 가능한 FID의 평균을 보고합니다.

> Due to the fact that the training is sensitive to the initial weights, we train the models 5 times, each time with a different random initialization, and report the median FID. The variance in FID for models obtained by sequential Bayesian optimization is handled implicitly by the applied exploration-exploitation strategy.
>> 훈련은 초기 가중치에 민감하기 때문에, 우리는 매번 다른 무작위 초기화로 모델을 5번 훈련시키고 중앙값 FID를 보고합니다. 순차 베이지안 최적화를 통해 얻은 모델에 대한 FID의 분산은 적용된 탐색-탐색 전략에 의해 암묵적으로 처리됩니다.

#### $\mathbf{3.1.\;Regularization\;and\;Normalization}$

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)A-Large-Scale-Study-on-Regularization-and-Normalization-in-GANs/Figure-3.JPG)

> Figure 3. The first row show s the FID distribution for top 5% models. We compare the ResNet-based neural architecture with the SNDCGAN architecture. We use the non-saturating (NS) loss in all experiments, and apply either spectral normalization (SN) or the gradient penalty (GP). We observe that spectral norm consistently improves the sample quality. In some cases the gradient penalty can help, but the need to tune one additional hyperparameter leads to a lower computational efficiency.
>> 그림 3입니다. 첫 번째 행은 상위 5% 모델에 대한 FID 분포를 보여 줍니다. 우리는 ResNet 기반 신경 아키텍처를 SNDCGAN 아키텍처와 비교합니다. 우리는 모든 실험에서 비포화성(NS) 손실을 사용하고 스펙트럼 정규화(SN) 또는 그레이디언트 패널티(GP)를 적용합니다. 우리는 스펙트럼 노름이 샘플 품질을 지속적으로 향상시킨다는 것을 관찰합니다. 경우에 따라 그레이디언트 패널티가 도움이 될 수 있지만 하이퍼 파라미터 하나를 추가로 조정해야 하므로 계산 효율성이 떨어집니다.

> The goal of this study is to compare the relative performance of various regularization and normalization methods presented in the literature, namely: batch normalization (BN) (Ioffe & Szegedy, 2015), layer normalization (LN) (Ba et al., 2016), spectral normalization (SN), gradient penalty (GP) (Gulrajani et al., 2017), Dragan penalty (DR) (Kodali et al., 2017), or $L^{2}$ regularization. We fix the loss to nonsaturating loss (Goodfellow et al., 2014) and the ResNet19 with generator and discriminator architectures described in Table 5a. We analyze the impact of the loss function in Section 3.2 and of the architecture in Section 3.3. We consider both CELEBA-HQ-128 and LSUN-BEDROOM with the hyperparameter settings shown in Tables 1 and 2.
>> 본 연구의 목표는 문헌에 제시된 다양한 정규화 및 정규화 방법의 상대적인 성능을 비교하는 것입니다. 즉, 배치 정규화(BN)(Ioffe & Szegdy, 2015), 계층 정규화(LN)(Ba et al., 2016), 스펙트럼 정규화(SN), 그레이디언트 패널티(GP)(Gulrajani et al., 2017), Dragan 패널티(Dali)이다. 등, 2017) 또는 $L^{2}$ 정규화를 참조하십시오. 우리는 표 5a에 설명된 발전기 및 판별기 아키텍처를 통해 비포화성 손실(Goodfellow et al., 2014)과 ResNet19로 손실을 해결합니다. 우리는 3.2절의 손실 함수와 3.3절의 아키텍처의 영향을 분석합니다. 표 1과 2에 표시된 하이퍼 파라미터 설정을 사용하여 CELEBA-HQ-128과 LSUN-BRODE를 모두 고려합니다.

> The results are presented in Figure 1. We observe that adding batch norm to the discriminator hurts the performance. Secondly, gradient penalty can help, but it doesn’t stabilize the training. In fact, it is non-trivial to strike a balance of the loss and regularization strength. Spectral normalization helps improve the model quality and is more computationally efficient than gradient penalty. This is consistent with recent results in Zhang et al.(2019). Similarly to the loss study, models using $GP$ penalty may benefit from 5:1 ratio of discriminator to generator updates. Furthermore, in a separate ablation study we observed that running the optimization procedure for an additional 100K steps is likely to increase the performance of the models with $GP$ penalty
>> 결과는 그림 1에 나와 있습니다. 판별기에 배치 규범을 추가하면 성능이 저하된다는 것을 관찰합니다. 두 번째로, 그레이디언트 벌칙은 도움이 될 수 있지만, 훈련을 안정시키지는 못합니다. 사실, 손실과 정규화 강도의 균형을 맞추는 것은 쉬운 일이 아닙니다. 스펙트럼 정규화는 모델 품질을 향상시키는 데 도움이 되며 그레이디언트 패널티보다 계산 효율성이 높습니다. 이는 Zhang 등(2019)의 최근 결과와 일치합니다. 손실 연구와 유사하게 $GP$ 페널티를 사용하는 모델은 판별기 대 발생기 업데이트의 5:1 비율로 혜택을 받을 수 있습니다. 또한 별도의 절제 연구에서 추가 10K 단계에 대해 최적화 절차를 실행하면 $GP$ 페널티가 있는 모델의 성능이 향상될 가능성이 있음을 관찰했습니다.

#### $\mathbf{3.2.\;Impact\;of\;the\;Loss\;Function}$

> Here we investigate whether the above findings also hold when the loss functions are varied. In addition to the non-saturating loss (NS), we also consider the the leastsquares loss (LS) (Mao et al., 2017), or the Wasserstein loss (WGAN) (Arjovsky et al., 2017). We use the ResNet19 with generator and discriminator architectures detailed in Table 5a. We consider the most prominent normalization and regularization approaches: gradient penalty (Gulrajani et al., 2017), and spectral normalization (Miyato et al., 2018). Other parameters are detailed in Table 1. We also performed a study on the recently popularized hinge loss (Lim & Ye, 2017; Miyato et al., 2018; Brock et al., 2019) and present it in the Appendix.
>> 여기서는 손실 함수가 다양한 경우에도 위의 결과가 유지되는지 여부를 조사합니다. 비포화 손실(NS) 외에도 최소 제곱 손실(LS)(Mao et al., 2017) 또는 WGAN(Arjovsky et al., 2017)도 고려합니다. 우리는 표 5a에 자세히 설명된 생성기 및 판별기 아키텍처와 함께 ResNet19를 사용합니다. 우리는 가장 두드러진 정규화 및 정규화 접근 방식인 그레이디언트 패널티(Gulrajani et al., 2017)와 스펙트럼 정규화(Miyato et al., 2018)를 고려합니다. 다른 파라미터는 표 1에 자세히 설명되어 있습니다. 또한 최근 대중화된 힌지 손실(Lim & Ye, 2017; Miyato et al., 2018; Brock et al., 2019)에 대한 연구를 수행하고 부록에 이를 제시하였다.

> The results are presented in Figure 2. Spectral normalization improves the model quality on both datasets. Similarly, the gradient penalty can help, but finding a good regularization tradeoff is non-trivial and requires a large computational budget. Models using the $GP$ penalty benefit from 5:1 ratio of discriminator to generator updates (Gulrajani et al., 2017).
>> 결과는 그림 2에 나와 있습니다. 스펙트럼 정규화는 두 데이터 세트 모두에서 모델 품질을 향상시킵니다. 마찬가지로, 그레이디언트 패널티는 도움이 될 수 있지만, 좋은 정규화 트레이드오프를 찾는 것은 사소하지 않으며 많은 계산 예산이 필요합니다. $GP$ 페널티를 사용하는 모델은 판별기 대 발전기 업데이트의 5:1 비율의 혜택을 받는다(Gulrajani et al., 2017).

#### $\mathbf{3.3.\;Impact\;of\;the\;Neural\;Architectures}$

> An interesting practical question is whether our findings also hold for different neural architectures. To this end, we also perform a study on SNDCGAN from Miyato et al.(2018). We consider the non-saturating GAN loss, gradient penalty and spectral normalization. While for smaller architectures regularization is not essential (Lucic et al., 2018), the regularization and normalization effects might become more relevant due to deeper architectures and optimization considerations.
>> 흥미로운 실용적인 질문은 우리의 연구 결과가 다른 신경 구조에도 적용되는지 여부입니다. 이를 위해 미야토 외(2018)의 SNDCGAN에 대한 연구도 수행합니다. 우리는 비포화 GAN 손실, 그레이디언트 패널티 및 스펙트럼 정규화를 고려합니다. 소규모 아키텍처의 경우 정규화가 필수적이지는 않지만(Lucic 등, 2018), 정규화 및 정규화 효과는 아키텍처와 최적화 고려사항이 더 깊어지기 때문에 더욱 관련될 수 있습니다.

> The results are presented in Figure 3. We observe that both architectures achieve comparable results and benefit from regularization and normalization. Spectral normalization strongly outperforms the baseline for both architectures.
>>

> **Simultaneous Regularization and Normalization** A common observation is that the Lipschitz constant of the discriminator is critical for the performance, one may expect simultaneous regularization and normalization could improve model quality. To quantify this effect, we fix the loss to non-saturating loss (Goodfellow et al., 2014), use the Resnet19 architecture (as above), and combine several normalization and regularization schemes, with hyperparameter settings shown in Table 1 coupled with 24 randomly selected parameters. The results are presented in Figure 4. We observe that one may benefit from additional regularization and normalization. However, a lot of computational effort has to be invested for somewhat marginal gains in FID. Nevertheless, given enough computational budget we advocate simultaneous regularization and normalization – spectral normalization and layer normalization seem to perform well in practice.
>> **동시 정규화 및 정규화** 판별기의 Lipschitz 상수가 성능에 매우 중요하므로 동시 정규화와 정규화가 모델 품질을 향상시킬 수 있다고 예상할 수 있습니다. 이 효과를 정량화하기 위해, 우리는 손실을 불포화 손실로 수정하고(Goodfellow et al., 2014), Resnet19 아키텍처를 사용하고(위처럼), 몇 가지 정규화 및 정규화 체계를 표 1에 표시된 하이퍼 파라미터 설정과 임의로 선택한 24개의 매개 변수를 결합합니다. 결과는 그림 4에 나와 있습니다. 우리는 추가적인 정규화 및 정규화를 통해 이익을 얻을 수 있다는 것을 관찰합니다. 그러나 FID에서 다소 미미한 이득을 얻으려면 많은 계산 노력이 투자되어야 합니다. 그럼에도 불구하고, 충분한 계산 예산이 주어지면 우리는 동시 정규화와 정규화를 옹호합니다. 스펙트럼 정규화와 계층 정규화는 실제로 잘 수행되는 것처럼 보입니다.

### $\mathbf{4.\;Challenges\;of\;a\;Large-Scale\;Study}$

> In this section we focus on several pitfalls we encountered while trying to reproduce existing results and provide a fair and accurate comparison.
>> 이 섹션에서는 기존 결과를 재현하고 공정하고 정확한 비교를 시도하는 동안 마주친 몇 가지 함정에 초점을 맞춥니다.

> **Metrics** There already seems to be a divergence in how the FID score is computed: (1) Some authors report the score on training data, yielding a FID between 50K training and 50K generated samples (Unterthiner et al., 2018). Some opt to report the FID based on 10K test samples and 5K generated samples and use a custom implementation (Miyato et al., 2018). Finally, Lucic et al.(2018) report the score with respect to the test data, in particular FID between 10K test samples, and 10K generated samples. The subtle differences will result in a mismatch between the reported FIDs, in some cases of more than 10%. We argue that FID should be computed with respect to the test dataset. Furthermore, whenever possible, one should use the same number of instances as previously reported results. Towards this end we use 10K test samples and 10K generated samples on CIFAR10 and LSUN-BEDROOM, and 3K vs 3K on CELEBA-HQ-128 as in in Lucic et al.(2018).
>> **Metrics** FID 점수 계산 방법에는 이미 차이가 있는 것 같습니다. (1) 일부 저자는 훈련 데이터에 대한 점수를 보고하여 50K 훈련과 50K 생성된 샘플 사이에서 FID를 산출합니다(Unterthiner et al., 2018). 일부는 10K 테스트 샘플과 5K 생성된 샘플을 기반으로 FID를 보고하고 맞춤형 구현을 사용합니다(Miyato et al., 2018). 마지막으로, Lucic 외 연구진(2018)은 테스트 데이터, 특히 10K 테스트 샘플과 10K 생성된 샘플 사이의 FID와 관련된 점수를 보고합니다. 미묘한 차이로 인해 보고된 FID 간에 불일치가 발생하며, 경우에 따라 10%가 넘는 경우도 있습니다. 우리는 FID가 테스트 데이터 세트와 관련하여 계산되어야 한다고 주장합니다. 또한 가능하면 이전에 보고된 결과와 동일한 수의 인스턴스를 사용해야 합니다. 이를 위해 우리는 Lucic et al. (2018)에서와 같이 CIFAR10 및 LSUN-BROME에서 10K 테스트 샘플과 10K 생성 샘플을 사용하고 CELEBA-HQ-128에서 3K 대 3K를 사용합니다.

> **Details of Neural Architectures** Even in popular architectures, like ResNet, there is still a number of design decisions one needs to make, that are often omitted from the reported results. Those include the exact design of the ResNet block (order of layers, when is ReLu applied, when to upsample and downsample, how many filters to use). Some of these differences might lead to potentially unfair comparison. As a result, we suggest to use the architectures presented within this work as a solid baseline. An ablation study on various ResNet modifications is available in the Appendix.
>> **신경 아키텍처의 세부 정보** ResNet과 같은 인기 있는 아키텍처에서도 보고 결과에서 종종 생략되는 설계 결정이 여전히 많이 있습니다. 여기에는 ResNet 블록의 정확한 설계(계층 순서, ReLu 적용 시기, 업샘플 및 다운샘플링 시기, 사용할 필터 수)가 포함됩니다. 이러한 차이 중 일부는 잠재적으로 불공평한 비교를 초래할 수 있습니다. 따라서 이 작업 내에 제시된 아키텍처를 견고한 기준으로 사용할 것을 권장합니다. 다양한 ResNet 수정에 대한 절제 스터디는 부록에서 사용할 수 있습니다.

> **Datasets** A common issue is related to dataset processing – does LSUN-BEDROOM always correspond to the same dataset? In most cases the precise algorithm for upscaling or cropping is not clear which introduces inconsistencies between results on the “same” dataset.
>> **Datasets** 데이터셋 처리와 관련된 일반적인 문제 - LSUN-BHOME이 항상 동일한 데이터셋에 대응합니까? 대부분의 경우 업스케일링 또는 자르기 위한 정확한 알고리즘이 명확하지 않아 "같은" 데이터셋의 결과 간에 불일치가 발생합니다.

> **Implementation Details and Non-Determinism** One major issue is the mismatch between the algorithm presented in a paper and the code provided online. We are aware that there is an embarrassingly large gap between a good implementation and a bad implementation of a given model. Hence, when no code is available, one is forced to guess which modifications were done. Another particularly tricky issue is removing randomness from the training process. After one fixes the data ordering and the initial weights, obtaining the same score by training the same model twice is non-trivial due to randomness present in certain GPU operations (Chetlur et al., 2014). Disabling the optimizations causing the non-determinism often results in an order of magnitude running time penalty.
>> **구현 세부 사항 및 비결정론** 한 가지 주요 문제는 논문에 제시된 알고리즘과 온라인으로 제공되는 코드 간의 불일치입니다. 우리는 주어진 모델의 좋은 구현과 나쁜 구현 사이에 당혹스러울 정도로 큰 차이가 있다는 것을 알고 있습니다. 따라서 사용할 수 있는 코드가 없을 때는 어떤 수정이 이루어졌는지 추측해야 합니다. 특히 까다로운 또 다른 문제는 교육 프로세스에서 무작위성을 제거하는 것입니다. 데이터 순서와 초기 가중치를 수정한 후 동일한 모델을 두 번 훈련하여 동일한 점수를 얻는 것은 특정 GPU 작업에 존재하는 무작위성으로 인해 중요하지 않습니다(Chetur 등, 2014). 최적화를 사용하지 않도록 설정하면 종종 상당한 실행 시간 페널티가 발생합니다.

> While each of these issues taken in isolation seems minor, they compound to create a mist which introduces friction in practical applications and the research process (Sculley et al., 2018).
>> 격리된 이러한 각 문제는 사소한 것처럼 보이지만, 실제 적용 및 연구 과정에서 마찰을 일으키는 안개를 생성하기 위해 복합적으로 작용합니다(Sculley et al., 2018).

### $\mathbf{5.\;Related\;Work}$

> A recent large-scale study on GANs and Variational Autoencoders was presented in Lucic et al.(2018). The authors consider several loss functions and regularizers, and study the effect of the loss function on the FID score, with low-to-medium complexity datasets (MNIST, CIFAR10, CELEBA), and a single neural network architecture. In this limited setting, the authors found that there is no statistically significant difference between recently introduced models and the original non-saturating GAN. A study of the effects of gradient-norm regularization in GANs was recently presented in Fedus et al.(2018). The authors posit that the gradient penalty can also be applied to the non-saturating GAN, and that, to a limited extent, it reduces the sensitivity to hyperparameter selection. In a recent work on spectral normalization, the authors perform a small study of the competing regularization and normalization approaches (Miyato et al., 2018). We are happy to report that we could reproduce these results and we present them in the Appendix.
>> 최근 GAN 및 Variational Autoencoder에 대한 대규모 연구가 Lucic 외(2018)에 제시되었습니다. 저자는 몇 가지 손실 함수와 정규화를 고려하고, 중저 복잡도 데이터 세트(MNIST, CIFAR10, CELEBA)와 단일 신경망 아키텍처를 사용하여 손실 함수가 FID 점수에 미치는 영향을 연구합니다. 이 제한된 설정에서, 저자들은 최근에 도입된 모델과 원래의 비포화 GAN 사이에 통계적으로 유의한 차이가 없다는 것을 발견했습니다. GAN의 그레이디언트-노름 정규화의 영향에 대한 연구가 최근 Fedus et al. (2018)에 제시되었습니다. 저자들은 그레이디언트 패널티가 비포화 GAN에도 적용될 수 있으며, 제한된 범위 내에서 초 매개 변수 선택에 대한 민감도를 감소시킨다고 가정합니다. 스펙트럼 정규화에 대한 최근 연구에서, 저자들은 경쟁적인 정규화 및 정규화 접근법에 대한 작은 연구를 수행합니다(Miyato et al., 2018). 이러한 결과를 재현할 수 있다는 소식을 전하게 되어 기쁘게 생각하며, 이를 부록에 제시합니다.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)A-Large-Scale-Study-on-Regularization-and-Normalization-in-GANs/Figure-4.JPG)

> Figure 4. Can one benefit from simultaneous regularization and normalization? The plots show the FID distribution for top 5% models where we compare various combinations of regularization and normalization strategies. Gradient penalty coupled with spectral normalization (SN) or layer normalization (LN) strongly improves the performance over the baseline. This can be partially explained by the fact that SN doesn’t ensure that the discriminator is 1-Lipschitz due to the way convolutional layers are normalized.
>> 그림 4입니다. 정규화와 정규화를 동시에 통해 이익을 얻을 수 있습니까? 그래프에는 정규화 및 정규화 전략의 다양한 조합을 비교하는 상위 5% 모델에 대한 FID 분포가 나와 있습니다. 스펙트럼 정규화(SN) 또는 계층 정규화(LN)와 결합된 그레이디언트 패널티는 기준선에 비해 성능을 크게 향상시킵니다. 이는 SN이 컨볼루션 레이어가 정규화된 방식으로 인해 판별기가 1-Lipschitz임을 보장하지 않는다는 사실에 의해 부분적으로 설명될 수 있습니다.

> Inspired by these works and building on the available opensource code from Lucic et al.(2018), we take one additional step in all dimensions considered therein: more complex neural architectures, more complex datasets, and more involved regularization and normalization schemes.
>> 이러한 작업에서 영감을 받아 Lucic 등의 사용 가능한 오픈 소스 코드를 기반으로 구축되었으며(2018년) 더 복잡한 신경 아키텍처, 더 복잡한 데이터 세트, 더 많은 관련 정규화 및 정규화 체계 등 여기서 고려된 모든 차원에 대해 한 단계 더 나아갔다.

### $\mathbf{6.\;Conclusions\;and\;Future\;Work}$

> In this work we study the impact of regularization and normalization schemes on GAN training. We consider the state-of-the-art approaches and vary the loss functions and neural architectures. We study the impact of these design choices on the quality of generated samples which we assess by recently introduced quantitative metrics.
>> 본 연구에서는 정규화 및 정규화 체계가 GAN 교육에 미치는 영향을 연구합니다. 우리는 최첨단 접근 방식을 고려하고 손실 기능과 신경 구조를 다양화합니다. 우리는 이러한 설계 선택이 생성된 샘플의 품질에 미치는 영향을 연구하며, 최근에 도입된 정량적 메트릭을 통해 평가합니다.

> Our fair and thorough empirical evaluation suggests that when the computational budget is limited one should consider non-saturating GAN loss and spectral normalization as default choices when applying GANs to a new dataset. Given additional computational budget, we suggest adding the gradient penalty from Gulrajani et al.(2017) and training the model until convergence. Furthermore, we observe that both classes of popular neural architectures can perform well across the considered datasets. A separate ablation study uncovered that most of the variations applied in the ResNet style architectures lead to marginal improvements in the sample quality.
>> 우리의 공정하고 철저한 경험적 평가는 계산 예산이 제한될 때 새로운 데이터 세트에 GAN을 적용할 때 비포화 GAN 손실과 스펙트럼 정규화를 기본 선택으로 고려해야 함을 시사합니다. 추가 계산 예산이 주어지면 Gulrajani 등(2017)의 그레이디언트 페널티를 추가하고 수렴할 때까지 모델을 훈련시킬 것을 제안한다. 또한, 우리는 인기 있는 신경 아키텍처의 두 가지 클래스가 고려된 데이터 세트에서 잘 수행될 수 있다는 것을 관찰합니다. 별도의 절제 연구에서 ResNet 스타일 아키텍처에 적용된 대부분의 변형은 표본 품질의 한계 향상으로 이어진다는 것을 발견했습니다.

> As a result of this large-scale study we identify the common pitfalls standing in the way of accurate and fair comparison and propose concrete actions to demystify the future results – issues with metrics, dataset preprocessing, non-determinism, and missing implementation details are particularly striking. We hope that this work, together with the open-sourced reference implementations and trained models, will serve as a solid baseline for future GAN research.
>> 이 대규모 연구의 결과로 정확하고 공정한 비교에 방해가 되는 일반적인 함정을 식별하고 향후 결과를 입증하기 위한 구체적인 조치를 제안합니다. 즉, 메트릭, 데이터 세트 사전 처리, 비결정론 및 누락된 구현 세부 정보가 특히 두드러집니다. 우리는 이 작업이 오픈 소스 참조 구현 및 훈련된 모델과 함께 향후 GAN 연구를 위한 견고한 기준선으로 작용하기를 바랍니다.

> Future work should carefully evaluate models which necessitate large-scale training such as BigGAN (Brock et al., 2019), models with custom architectures (Chen et al., 2019; Karras et al., 2019; Zhang et al., 2019), recently proposed regularization techniques (Roth et al., 2017; Mescheder et al., 2018), and other proposals for stabilizing the training (Chen et al., 2018). In addition, given the popularity of conditional GANs, one should explore whether these insights transfer to the conditional settings. Finally, given the drawbacks of FID and $IS$, additional quantitative evaluation using recently proposed metrics could bring novel insights (Sajjadi et al., 2018; Kynka¨anniemi et al., 2019).
>> 향후 작업은 BigGAN(Brock et al., 2019), 맞춤형 아키텍처를 가진 모델(Chen et al., 2019; Karras et al., 2019; Zhang et al., 2019), 최근에 제안된 정규화 기술(Roth et al., 2017; Mescheder et al., 2018) 및 열차 안정화를 위한 다른 제안과 같은 대규모 교육이 필요한 모델을 신중하게 평가해야 합니다.ing (Chen et al., 2018). 또한 조건부 GAN의 인기를 감안할 때 이러한 통찰력이 조건부 설정으로 전달되는지 여부를 탐구해야 합니다. 마지막으로, FID와 $IS$의 단점을 고려할 때, 최근에 제안된 메트릭을 사용한 추가 정량적 평가는 새로운 통찰력을 가져올 수 있습니다(Sajjadi et al., 2018; Kynkaannanniemi et al., 2019).

#### $\mathbf{Acknowledgments}$

> We are grateful to Michael Tschannen for detailed comments on this manuscript.
>> 이 원고에 대한 자세한 코멘트에 대해 Michael Tchannen에게 감사드립니다.

---

### $\mathbf{References}$

<a href="#footnote_1_2" name="footnote_1_1">[1]</a> Arjovsky, M., Chintala, S., and Bottou, L. Wasserstein Generative Adversarial Networks. In International Conference on Machine Learning, 2017.

<a href="#footnote_2_2" name="footnote_2_1">[2]</a> Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

<a href="#footnote_3_2" name="footnote_3_1">[3]</a> Binkowski, M., Sutherland, $D$. J., Arbel, M., and Gretton, A. Demystifying MMD GANs. In International Conference on Learning Representations, 2018.

<a href="#footnote_4_2" name="footnote_4_1">[4]</a> Borji, A. Pros and cons of GAN evaluation measures. Computer Vision and Image Understanding, 2019.

<a href="#footnote_5_2" name="footnote_5_1">[5]</a> Brock, A., Donahue, J., and Simonyan, K. Large scale GAN training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.

<a href="#footnote_6_2" name="footnote_6_1">[6]</a> Chen, T., Zhai, X., Ritter, M., Lucic, M., and Houlsby, N. Self-Supervised Generative Adversarial Networks. In Computer Vision and Pattern Recognition, 2018.

<a href="#footnote_7_2" name="footnote_7_1">[7]</a> Chen, T., Lucic, M., Houlsby, N., and Gelly, S. On Self Modulation for Generative Adversarial Networks. In International Conference on Learning Representations, 2019.

<a href="#footnote_8_2" name="footnote_8_1">[8]</a> Chetlur, S., Woolley, C., Vandermersch, $P$., Cohen, J., Tran, J., Catanzaro, B., and Shelhamer, E. cuDNN: Efficient primitives for deep learning. arXiv preprint arXiv:1410.0759, 2014.

<a href="#footnote_9_2" name="footnote_9_1">[9]</a> Denton, E. L., Chintala, S., Szlam, A., and Fergus, R. Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks. In Advances in Neural Information Processing Systems, 2015.

<a href="#footnote_10_2" name="footnote_10_1">[10]</a> Fedus, W., Rosca, M., Lakshminarayanan, B., Dai, A. M., Mohamed, S., and Goodfellow, I. Many paths to equilibrium: GANs do not need to decrease a divergence at every step. In International Conference on Learning Representations, 2018.

<a href="#footnote_11_2" name="footnote_11_1">[11]</a> Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, $D$., Ozair, S., Courville, A., and Bengio, Y. Generative Adversarial Nets. In Advances in Neural Information Processing Systems, 2014.

<a href="#footnote_12_2" name="footnote_12_1">[12]</a> Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., and Courville, A. Improved training of Wasserstein GANs. In Advances in Neural Information Processing Systems, 2017.

<a href="#footnote_13_2" name="footnote_13_1">[13]</a> He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Computer Vision and Pattern Recognition, 2016. 

<a href="#footnote_14_2" name="footnote_14_1">[14]</a> Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Klambauer, G., and Hochreiter, S. GANs trained by a two time-scale update rule converge to a Nash equilibrium. In Advances in Neural Information Processing Systems, 2017.

<a href="#footnote_15_2" name="footnote_15_1">[15]</a> Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

<a href="#footnote_16_2" name="footnote_16_1">[16]</a> Isola, $P$., Zhu, J.-Y., Zhou, T., and Efros, A. A. Image-toimage translation with conditional adversarial networks. In Computer Vision and Pattern Recognition, 2017.

<a href="#footnote_17_2" name="footnote_17_1">[17]</a> Karras, T., Aila, T., Laine, S., and Lehtinen, J. Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations, 2018.

<a href="#footnote_18_2" name="footnote_18_1">[18]</a> Karras, T., Laine, S., and Aila, T. A style-based generator architecture for generative adversarial networks. In Computer Vision and Pattern Recognition, 2019.

<a href="#footnote_19_2" name="footnote_19_1">[19]</a> Kingma, $D$. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

<a href="#footnote_20_2" name="footnote_20_1">[20]</a> Kodali, N., Abernethy, J., Hays, J., and Kira, Z. On convergence and stability of GANs. arXiv preprint arXiv:1705.07215, 2017.

<a href="#footnote_21_2" name="footnote_21_1">[21]</a> Kynka¨anniemi, T., Karras, T., Laine, S., Lehtinen, J., and Aila, T. Improved precision and recall metric for assessing generative models. arXiv preprint arXiv:1904.06991, 2019.

<a href="#footnote_22_2" name="footnote_22_1">[22]</a> Lim, J. H. and Ye, J. C. Geometric GAN. arXiv preprint arXiv:1705.02894, 2017. 

<a href="#footnote_23_2" name="footnote_23_1">[23]</a> Lucic, M., Kurach, K., Michalski, M., Gelly, S., and Bousquet, O. Are GANs Created Equal? A Large-Scale Study. In Advances in Neural Information Processing Systems, 2018.

<a href="#footnote_24_2" name="footnote_24_1">[24]</a> Lucic, M., Tschannen, M., Ritter, M., Zhai, X., Bachem, O., and Gelly, S. High-Fidelity Image Generation With Fewer Labels. In International Conference on Machine Learning, 2019.

<a href="#footnote_25_2" name="footnote_25_1">[25]</a> Mao, X., Li, $Q$., Xie, H., Lau, R. Y., Wang, Z., and Smolley, S. $P$. Least squares generative adversarial networks. In International Conference on Computer Vision, 2017.

<a href="#footnote_26_2" name="footnote_26_1">[26]</a> Menick, J. and Kalchbrenner, N. Generating high fidelity images with subscale pixel networks and multidimensional upscaling. In International Conference on Learning Representations, 2019.

<a href="#footnote_27_2" name="footnote_27_1">[27]</a> Mescheder, L., Geiger, A., and Nowozin, S. Which training methods for GANs do actually Converge? arXiv preprint arXiv:1801.04406, 2018.

<a href="#footnote_28_2" name="footnote_28_1">[28]</a> Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. Spectral normalization for generative adversarial networks. International Conference on Learning Representations, 2018.

<a href="#footnote_29_2" name="footnote_29_1">[29]</a> Radford, A., Metz, L., and Chintala, S. Unsupervised representation learning with deep convolutional generative adversarial networks. International Conference on Learning Representations, 2016.

<a href="#footnote_30_2" name="footnote_30_1">[30]</a> Roth, K., Lucchi, A., Nowozin, S., and Hofmann, T. Stabilizing training of generative adversarial networks through regularization. In Advances in Neural Information Processing Systems, 2017.

<a href="#footnote_31_2" name="footnote_31_1">[31]</a> Sajjadi, M. S., Bachem, O., Lucic, M., Bousquet, O., and Gelly, S. Assessing generative models via precision and recall. In Advances in Neural Information Processing Systems, 2018.

<a href="#footnote_32_2" name="footnote_32_1">[32]</a> Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., and Chen, X. Improved techniques for training GANs. In Advances in Neural Information Processing Systems, 2016.

<a href="#footnote_33_2" name="footnote_33_1">[33]</a> Sculley, $D$., Snoek, J., Wiltschko, A., and Rahimi, A. Winner’s Curse? On Pace, Progress, and Empirical Rigor, 2018.

<a href="#footnote_34_2" name="footnote_34_1">[34]</a> Srinivas, N., Krause, A., Kakade, S., and Seeger, M. W. Gaussian process optimization in the bandit setting: No regret and experimental design. In International Conference on Machine Learning, 2010.

<a href="#footnote_35_2" name="footnote_35_1">[35]</a> Tschannen, M., Agustsson, E., and Lucic, M. Deep generative models for distribution-preserving lossy compression. Advances in Neural Information Processing Systems, 2018. 

<a href="#footnote_36_2" name="footnote_36_1">[36]</a> Unterthiner, T., Nessler, B., Seward, C., Klambauer, G., Heusel, M., Ramsauer, H., and Hochreiter, S. Coulomb GANs: Provably Optimal Nash Equilibria via Potential Fields. In International Conference on Learning Representations, 2018.

<a href="#footnote_37_2" name="footnote_37_1">[37]</a> Yu, F., Zhang, Y., Song, S., Seff, A., and Xiao, J. LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop. arXiv preprint arXiv:1506.03365, 2015.

<a href="#footnote_38_2" name="footnote_38_1">[38]</a> Zhang, H., Goodfellow, I., Metaxas, $D$., and Odena, A. Self-Attention Generative Adversarial Networks. In International Conference on Machine Learning, 2019.