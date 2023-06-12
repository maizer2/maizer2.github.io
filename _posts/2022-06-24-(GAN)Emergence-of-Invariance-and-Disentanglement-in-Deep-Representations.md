---
layout: post 
title: "(GAN)Emergence of Invariance and Disentanglement in Deep Representation Translation"
categories: [1. Computer Engineering]
tags: [1.7. Paper Review, 1.2.2.5. GAN]
---

### [GAN Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Paper-of-GAN.html)

### [$$\mathbf{Emergence\;of\;Invariance\;and\;Disentanglement\;in\;Deep\;Representations}$$](https://arxiv.org/pdf/1706.01350.pdf)

#### $$\mathbf{Alessandro\;chille,\;Stefano\;Soatto}$$

### $\mathbf{Abstract}$

> Using established principles from Statistics and Information Theory, we show that invariance to nuisance factors in a deep neural network is equivalent to information minimality of the learned representation, and that stacking layers and injecting noise during training naturally bias the network towards learning invariant representations. We then decompose the cross-entropy loss used during training and highlight the presence of an inherent overfitting term. We propose regularizing the loss by bounding such a term in two equivalent ways: One with a Kullbach-Leibler term, which relates to a PAC-Bayes perspective; the other using the information in the weights as a measure of complexity of a learned model, yielding a novel Information Bottleneck for the weights. Finally, we show that invariance and independence of the components of the representation learned by the network are bounded above and below by the information in the weights, and therefore are implicitly optimized during training. The theory enables us to quantify and predict sharp phase transitions between underfitting and overfitting of random labels when using our regularized loss, which we verify in experiments, and sheds light on the relation between the geometry of the loss function, invariance properties of the learned representation, and generalization error.
Keywords: Representation learning; PAC-Bayes; information bottleneck; flat minima; generalization; invariance; independence;
>> 통계 및 정보 이론의 확립된 원칙을 사용하여, 우리는 심층 신경망의 성가신 요인에 대한 불변성이 학습된 표현의 정보 최소화와 동등하며, 훈련 중 레이어를 쌓고 노이즈를 주입하는 것이 자연스럽게 네트워크가 불변 표현을 학습하는 쪽으로 편향된다는 것을 보여준다. 그런 다음 훈련 중에 사용된 교차 엔트로피 손실을 분해하고 고유한 과적합 용어의 존재를 강조한다. 우리는 이러한 항을 다음과 같은 두 가지 동등한 방법으로 제한하여 손실을 정규화할 것을 제안한다. 하나는 PAC-Bayes 관점과 관련된 Kullbach-Leibler 용어를 사용하는 것이고, 다른 하나는 학습된 모델의 복잡성의 척도로 가중치의 정보를 사용하여 가중치에 대한 새로운 정보 병목 현상을 산출한다. 마지막으로, 우리는 네트워크에 의해 학습된 표현 구성 요소의 불변성과 독립성이 가중치의 정보에 의해 위와 아래에 제한되므로 훈련 중에 암묵적으로 최적화된다는 것을 보여준다. 이 이론은 실험에서 검증하는 정규화된 손실을 사용할 때 무작위 레이블의 과소적합과 과적합 사이의 날카로운 위상 전환을 정량화하고 예측할 수 있게 하며, 손실 함수의 기하학, 학습된 표현의 불변성 특성 및 일반화 오류 사이의 관계를 조명한다.
키워드: 표현 학습; PAC-베이즈; 정보 병목; 플랫 미니마; 일반화; 불변성; 독립성;

### $\mathbf{1.\;Introduction}$

> Efforts to understand the empirical success of deep learning have followed two main lines: Representation learning and optimization. In optimization, a deep network is treated as a black-box family of functions for which we want to find parameters (weights) that yield good generalization. Aside from the difficulties due to the non-convexity of the loss function, the fact that deep networks are heavily over-parametrized presents a theoretical challenge: The bias-variance trade-off suggests they may severely overfit; yet, even without explicit regularization, they perform remarkably well in practice. Recent work suggests that this is related to properties of the loss landscape and to the implicit regularization performed by stochastic gradient descent (SGD), but the overall picture is still hazy (Zhang et al., 2017).
>> 딥 러닝의 경험적 성공을 이해하려는 노력은 두 가지 주요 노선을 따라왔다. 표현 학습 및 최적화. 최적화에서, 심층 네트워크는 우리가 좋은 일반화를 산출하는 매개 변수(가중치)를 찾기를 원하는 블랙박스 함수 계열로 취급된다. 손실 함수의 비볼록성으로 인한 어려움은 차치하고, 심층 네트워크가 심하게 과도하게 매개 변수화되어 있다는 사실은 다음과 같은 이론적 과제를 제시한다. 편향-분산 트레이드오프는 그것들이 심각하게 과적합할 수 있음을 시사하지만, 명시적인 정규화 없이도, 그것들은 실제로 매우 잘 수행된다. 최근의 연구는 이것이 손실 경관의 특성과 확률적 경사 하강(SGD)에 의해 수행되는 암묵적 정규화와 관련이 있다고 제안하지만, 전체 그림은 여전히 모호하다(Zhang et al., 2017).

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)Emergence-of-Invariance-and-Disentanglement-in-Deep-Representations/Figure-1.JPG)

> Figure 1: (Left) The AlexNet model of Zhang et al. (2017) achieves high accuracy (red) even when trained with random labels on CIFAR-10. Using the IB Lagrangian to limit information in the weights leads to a sharp transition to underfitting (blue) predicted by the theory (dashed line). To overfit, the network needs to memorize the dataset, and the information needed grows linearly. (Right) For real labels, the information sufficient to fit the data without overfitting saturates to a value that depends on the dataset, but somewhat independent of the number of samples. Test accuracy shows a uniform blue plot for random labels, while for real labels it increases with the number of training samples, and is higher near the critical regularizer value $\beta{}=1$.
>> 그림 1: (왼쪽) Zhang 등의 AlexNet 모델(2017)은 CIFAR-10에서 무작위 레이블로 훈련해도 높은 정확도(빨간색)를 달성한다. IB 라그랑지안을 사용하여 가중치의 정보를 제한하면 이론에 의해 예측된 언더피팅(파란색)으로 급격히 전환됩니다. 오버핏을 하려면 네트워크가 데이터셋을 기억해야 하고 필요한 정보는 선형적으로 증가한다. (오른쪽) 실제 레이블의 경우 오버핏 없이 데이터를 적합하기에 충분한 정보는 데이터셋에 따라 다르지만 샘플 수와 다소 독립적이다. 테스트 정확도는 랜덤 레이블에 대해 균일한 파란색 플롯을 보여주지만 실제 레이블의 경우 훈련 샘플 수에 따라 증가하며 임계 정규화 값 $\beta{}=1$ 근처에서 더 높다.

> Representation learning, on the other hand, focuses on the properties of the representation learned by the layers of the network (the activations) while remaining largely agnostic to the particular optimization process used. In fact, the effectiveness of deep learning is often ascribed to the ability of deep networks to learn representations that are insensitive (invariant) to nuisances such as translations, rotations, occlusions, and also “disentangled,” that is, separating factors in the high-dimensional space of data (Bengio, 2009). Careful engineering of the architecture plays an important role in achieving insensitivity to simple geometric nuisance transformations, like translations and small deformations; however, more complex and dataset-specific nuisances still need to be learned. This poses a riddle: If neither the architecture nor the loss function explicitly enforce invariance and disentangling, how can these properties emerge consistently in deep networks trained by simple generic optimization?
>> 반면에 표현 학습은 사용되는 특정 최적화 프로세스에 크게 의존하지 않으면서 네트워크의 계층(활성화)에 의해 학습된 표현의 속성에 초점을 맞춘다. 사실, 딥 러닝의 효과는 종종 번역, 회전, 폐색 및 "분리"와 같은 불편함에 둔감(불변)하는 표현, 즉 데이터의 고차원 공간에서 요인을 분리하는 딥 네트워크의 학습 능력에 기인한다(Bengio, 2009). 아키텍처의 신중한 엔지니어링은 번역 및 작은 변형과 같은 단순한 기하학적 방해 변환에 대한 무감각을 달성하는 데 중요한 역할을 한다. 그러나 더 복잡하고 데이터 세트별 성가신 문제는 여전히 학습해야 한다. 이것은 수수께끼를 제기한다: 아키텍처나 손실 함수가 불변성과 분리를 명시적으로 시행하지 않는다면, 이러한 특성이 간단한 일반 최적화에 의해 훈련된 심층 네트워크에서 어떻게 일관되게 나타날 수 있는가?

> In this work, we address these questions by establishing information theoretic connections between these concepts. In particular, we show that: (a) a sufficient representation of the data is invariant if and only if it is minimal, i.e., it contains the smallest amount of information, although may not have small dimension; (b) the information in the representation, along with its total correlation (a measure of disentanglement) are tightly bounded by the information that the weights contain about the dataset; (c) the information in the weights, which is related to overfitting (Hinton and Van Camp, 1993), flat minima (Hochreiter and Schmidhuber, 1997), and a PAC-Bayes upper-bound on the test error (Section 6), can be controlled by implicit or explicit regularization. Moreover, we show that adding noise during the training is a simple and natural way of biasing the network towards invariant representations.
>> 본 연구에서는 이러한 개념들 사이에 정보 이론적 연결을 설정하여 이러한 질문을 해결한다. 특히, 우리는 다음과 같은 것을 보여준다. (a) 데이터의 충분한 표현은 그것이 최소인 경우에만 불변한다. 즉, 그것은 작은 차원을 갖지 않을 수 있지만 가장 작은 양의 정보를 포함할 수 있다. (b) 표현의 정보는 총 상관관계(분리의 척도)와 함께 정보에 의해 엄격하게 제한된다.가중치가 데이터 세트에 대해 포함하는 상태. (c) 과적합(힌튼 및 밴 캠프, 1993), 플랫 미니마(호크라이터 및 슈미트허버, 1997), 테스트 오류에 대한 PAC-베이즈 상한(섹션 6)과 관련된 가중치의 정보는 암시적 또는 명시적 정규화에 의해 제어될 수 있다. 또한, 우리는 훈련 중에 노이즈를 추가하는 것이 네트워크를 불변 표현으로 편향시키는 간단하고 자연스러운 방법임을 보여준다.

> Finally, we perform several experiments with realistic architectures and datasets to validate the assumptions underlying our claims. In particular, we show that using the information in the weights to measure the complexity of a deep neural network (DNN), rather than the number of its parameters, leads to a sharp and theoretically predicted transition between overfitting and underfitting regimes for random labels, shedding light on the questions of Zhang et al. (2017).
>> 마지막으로, 우리는 우리의 주장의 기초가 되는 가정을 검증하기 위해 현실적인 아키텍처와 데이터 세트로 몇 가지 실험을 수행한다. 특히, 가중치의 정보를 사용하여 심층 신경망(DNN)의 매개 변수의 수가 아닌 복잡성을 측정하면 랜덤 레이블에 대한 과적합과 과소적합 체제 간의 날카롭고 이론적으로 예측된 전환이 이루어짐을 보여줌으로써 장 외 연구진(2017)의 질문을 조명한다.

#### $\mathbf{1.1\;Related\;work}$

> The Information Bottleneck (IB) was introduced by Tishby et al. (1999) as a generalization of minimal sufficient statistics that allows trading off fidelity (sufficiency) and complexity of a representation. In particular, the IB Lagrangian reduces finding a minimal sufficient representation to a variational optimization problem. Later, Tishby and Zaslavsky (2015) and Shwartz-Ziv and Tishby (2017) advocated using the IB between the test data and the activations of a deep neural network, to study the sufficiency and minimality of the resulting representation. In parallel developments, the IB Lagrangian was used as a regularized loss function for learning representation, leading to new information theoretic regularizers (Achille and Soatto, 2018; Alemi et al., 2017a; Alemi et al., 2017b).
>> 정보 병목 현상(IB)은 충실도(충분성)와 표현의 복잡성을 절충할 수 있는 최소한의 충분한 통계의 일반화로서 티쉬비 외 연구진(1999)에 의해 도입되었다. 특히, IB 라그랑지안은 변동 최적화 문제에 대한 최소한의 충분한 표현을 찾는 것을 줄인다. 나중에 티슈비와 자슬라프스키(2015)와 슈워츠-지브와 티슈비(2017)는 결과 표현의 충분성과 최소성을 연구하기 위해 테스트 데이터와 심층 신경망의 활성화 사이에 IB를 사용할 것을 주장했다. 병렬 개발에서 IB 라그랑지안은 학습 표현을 위한 정규화 손실 함수로 사용되어 새로운 정보 이론적 정규화기로 이어졌다(Achille and Soatto, 2018; Alemi et al., 2017a; Alemi et al., 2017b).

> In this paper, we introduce an IB Lagrangian between the weights of a network and the training data, as opposed to the traditional one between the activations and the test datum. We show that the former can be seen both as a generalization of Variational Inference, related to Hinton and Van Camp (1993), and as a special case of the more general PAC-Bayes framework (McAllester, 2013), that can be used to compute high-probability upper-bounds on the test error of the network. One of our main contributions is then to show that, due to a particular duality induced by the architecture of deep networks, minimality of the weights (a function of the training dataset) and of the learned representation (a function of the test input) are connected: in particular we show that networks regularized either explicitly, or implicitly by SGD, are biased toward learning invariant and disentangled representations. The theory we develop could be used to explain the phenomena described in small-scale experiments in Shwartz-Ziv and Tishby (2017), whereby the initial fast convergence of SGD is related to sufficiency of the representation, while the later asymptotic phase is related to compression of the activations: While SGD is seemingly agnostic to the property of the learned representation, we show that it does minimize the information in the weights, from which the compression of the activations follows as a corollary of our bounds. Practical implementation of this theory on real large scale problems is made possible by advances in Stochastic Gradient Variational Bayes (Kingma and Welling, 2014; Kingma et al., 2015).
>> 본 논문에서는 활성화와 테스트 데이터 사이의 기존 방식과 달리 네트워크의 가중치와 훈련 데이터 사이의 IB 라그랑지안을 소개한다. 우리는 전자가 힌튼과 밴 캠프(1993)와 관련된 변형 추론(Variational Inference)의 일반화와 네트워크의 테스트 오류에 대한 높은 확률 상한을 계산하는 데 사용될 수 있는 보다 일반적인 PAC-Bayes 프레임워크의 특별한 경우(McAllester, 2013)로 모두 볼 수 있음을 보여준다. 우리의 주요 기여 중 하나는 심층 네트워크의 아키텍처에 의해 유도된 특정 이중성 때문에 가중치(훈련 데이터 세트의 기능)와 학습된 표현(테스트 입력의 기능)의 최소성이 연결된다는 것을 보여주는 것이다. 특히 네트워크가 명시적으로 또는 암묵적으로 정규화되었음을 보여준다. SGD에 의해, 불변하고 얽혀 있지 않은 표현을 학습하는 데 편향되었다. 우리가 개발한 이론은 Shwartz-Ziv와 Tishby(2017)의 소규모 실험에서 설명된 현상을 설명하는 데 사용될 수 있다. 따라서 SGD의 초기 빠른 수렴은 표현의 충분성과 관련이 있고, 이후 점근 단계는 활성화의 압축과 관련이 있다. SGD는 학습된 표현의 속성에 대해 겉보기에는 불가지론적이지만, 우리는 활성화의 압축이 우리 경계의 결과로 따르는 가중치의 정보를 최소화한다는 것을 보여준다. 실제 대규모 문제에 대한 이 이론의 실질적인 구현은 확률적 그레이디언트 변형 베이즈의 발전으로 가능하다(Kingma and Welling, 2014; Kingma et al., 2015).

> Representations learned by deep networks are observed to be insensitive to complex nuisance transformations of the data. To a certain extent, this can be attributed to the architecture. For instance, the use of convolutional layers and max-pooling can be shown to yield insensitivity to local group transformations (Bruna and Mallat, 2011; Anselmiet al., 2016; Soatto and Chiuso, 2016). But for more complex, dataset-specific, and in particular non-local, non-group transformations, such insensitivity must be acquired as part of the learning process, rather than being coded in the architecture. We show that a sufficient representation is maximally insensitive to nuisances if and only if it is minimal, allowing us to prove that a regularized network is naturally biased toward learning invariant representations of the data.
>> 심층 네트워크에 의해 학습된 표현은 데이터의 복잡한 성가신 변환에 민감하지 않은 것으로 관찰된다. 이는 어느 정도 아키텍처에 기인할 수 있습니다. 예를 들어, 컨볼루션 레이어와 최대 풀링의 사용은 국소 그룹 변환에 대한 무감각을 산출하는 것으로 보일 수 있다(Bruna and Mallat, 2011; Anselmiet al., 2016; Sotto and Chiuso, 2016). 그러나 더 복잡하고 데이터 세트별, 특히 비 로컬, 비그룹 변환의 경우 이러한 무감각성은 아키텍처에서 코딩되는 것이 아니라 학습 프로세스의 일부로 획득되어야 한다. 우리는 충분한 표현이 최소인 경우에만 소음에 최대 무감각하다는 것을 보여주며, 정규화된 네트워크가 데이터의 불변 표현을 학습하는 데 자연스럽게 치우쳐 있다는 것을 증명할 수 있다.

> Efforts to develop a theoretical framework for representation learning include Tishby and Zaslavsky (2015) and Shwartz-Ziv and Tishby (2017), who consider representations as stochastic functions that approximate minimal sufficient statistics, different from Bruna and Mallat (2011) who construct representations as (deterministic) operators that are invertible in the limit, while exhibiting reduced sensitivity (“stability”) to small perturbations of the data. Some of the deterministic constructions are based on the assumption that the underlying data is spatially stationary, and therefore work best on textures and other visual data that are not subject to occlusions and scaling nuisances. Anselmi et al. (2016) develop a theory of invariance to locally compact groups, and aim to construct maximal (“distinctive”) invariants, like Sundaramoorthi et al. (2009) that, however, assume nuisances to be infinite-dimensional groups (Grenander, 1993). These efforts are limited by the assumption that nuisances have a group structure. Such assumptions were relaxed by Soatto and Chiuso (2016) who advocate seeking for sufficient invariants, rather than maximal ones. We further advance this approach, but unlike prior work on sufficient dimensionality reduction, we do not seek to minimize the dimension of the representation, but rather its information content, as prescribed by our theory. Recent advances in Deep Learning provide us with computationally viable methods to train high-dimensional models and predict and quantify observed phenomena such as convergence to flat minima and transitions from overfitting to underfitting random labels, thus bringing the theory to fruition. Other theoretical efforts focus on complexity considerations, and explain the success of deep networks by ways of statistical or computational efficiency (Lee et al., 2017; Bengio, 2009; LeCun, 2012). “Disentanglement” is an often-cited property of deep networks (Bengio, 2009), but seldom formalized and studied analytically, although Ver Steeg and Galstyan (2015) has suggested studying it using the Total Correlation of the representation, also known as multi-variate mutual information, which we also use.
>> 표현 학습을 위한 이론적 프레임워크를 개발하기 위한 노력에는 표현을 (결정론적) 연산자로 구성하는 브루나 및 말라트(2011)와 달리 최소한의 충분한 통계를 근사하는 확률적 함수로 고려하는 티슈비와 자슬라프스키(2015)와 슈워츠-지브와 티슈비(2017)가 포함된다.데이터의 작은 섭동에 대해 감소된 민감도("파급")를 나타내면서 한계 내에서 가역적이다. 결정론적 구성 중 일부는 기본 데이터가 공간적으로 정지되어 있기 때문에 폐색 및 스케일링 노이즈에 영향을 받지 않는 텍스처 및 기타 시각적 데이터에 가장 잘 작동한다는 가정에 기초한다. 안셀미 외 연구진(2016)은 국소적으로 밀집된 그룹에 대한 불변성 이론을 개발하고, 순다라모르티 외 연구진처럼 최대("특이적") 불변성을 구성하는 것을 목표로 한다. (2009) 단, 잡음이 무한 차원 그룹이라고 가정한다(Grenander, 1993). 이러한 노력은 성가신 일이 집단 구조를 가지고 있다는 가정에 의해 제한된다. 그러한 가정은 최대가 아닌 충분한 불변량을 찾는 것을 옹호하는 Satto와 Chiuso(2016)에 의해 완화되었다. 우리는 이 접근법을 더욱 발전시키지만, 충분한 차원 축소에 대한 이전 연구와 달리, 우리는 우리의 이론에 의해 규정된 대로 표현의 차원을 최소화하는 것이 아니라 오히려 그것의 정보 내용을 최소화하려고 한다. 최근 딥 러닝의 발전은 고차원 모델을 훈련시키고 플랫 미니마로의 수렴과 과적합에서 과소적합 랜덤 레이블로의 전환과 같은 관찰된 현상을 예측하고 정량화하는 계산 가능한 방법을 제공하여 이론을 결실을 맺게 한다. 다른 이론적 노력은 복잡성 고려사항에 초점을 맞추고, 통계적 또는 계산적 효율성의 방법으로 심층 네트워크의 성공을 설명한다(Lee et al., 2017; Bengio, 2009; Le Cun, 2012). Ver Steeg와 Galstyan(2015)은 표현의 총 상관 관계(다변량 상호 정보라고도 알려진)를 사용하여 연구할 것을 제안했지만, "분리"는 종종 인용되는 심층 네트워크의 속성이지만(Bengio, 2009), 공식화되고 분석적으로 연구되는 경우는 드물다.

> We connect invariance properties of the representation to the geometry of the optimization residual, and to the phenomenon of flat minima (Dinh et al., 2017).
>> 표현의 불변 속성을 최적화 잔차의 기하학 및 플랫 미니마 현상에 연결한다(Dinhe et al., 2017).

> Following (McAllester, 2013), we have also explored relations between our theory and the PAC-Bayes framework (Dziugaite and Roy, 2017). As we show, our theory can also be derived in the PAC-Bayes framework, without resorting to information quantities and the Information Bottleneck, thus providing both an independent and alternative derivation, and a theoretically rigorous way to upper-bound the optimal loss function. The use of PACBayes theory to study the generalization properties of deep networks has been championed by Dziugaite and Roy (2017), who point out that minima that are flat in the sense of having a large volume, toward which stochastic gradient descent algorithms are implicitly or explicitly biased (Chaudhari and Soatto, 2018), naturally relates to the PAC-Bayes loss for the choice of a normal prior and posterior on the weights. This has been leveraged by Dziugaite and Roy (2017) to compute non-vacuous PAC-Bayes error bounds, even for deep networks.
>> (McAllester, 2013)에 이어, 우리는 우리의 이론과 PAC-Bayes 프레임워크(Dziugaite and Roy, 2017) 사이의 관계를 탐구했다. 우리가 보여주듯이, 우리의 이론은 정보 양과 정보 병목 현상에 의존하지 않고 PAC-Bayes 프레임워크에서 도출될 수 있으므로, 독립적이고 대안적인 파생과 최적의 손실 함수의 상한을 설정하는 이론적으로 엄격한 방법을 제공한다. 딥 네트워크의 일반화 특성을 연구하기 위한 PACBayes 이론의 사용은 Dziugaite와 Roy(2017)에 의해 옹호되었다. 이들은 큰 부피를 가진다는 의미에서 평평한 미니마가 확률적 경사 하강 알고리듬이 암묵적으로 또는 명시적으로 편향되어 있다고 지적한다(Chaudhari와 Sotto, 2018). 가중치에 대한 정상 사전 및 사후 선택을 위한 PAC-Bayes 손실. 이는 Dziugaite와 Roy(2017)가 딥 네트워크에 대해서도 비진공 PAC-Bayes 오류 경계를 계산하기 위해 활용했다.

### $\mathbf{2.\;Preliminaries}$

> A training set $D=(x,y)$, where $x=x_{i=1}^{N}$ and $y=y_{i=1}^{N}$, is a collection of N randomly sampled data points $x_{i}^{i}$ and their associated (usually discrete) labels. The samples are assumed to come from an unknown, possibly complex, distribution $p_{\theta}(x,y)$, parametrized by a parameter $\theta$. Following a Bayesian approach, we also consider $\theta$ to be a random variable, sampled from some unknown prior distribution $p(\theta)$, but this requirement is not necessary (see Section 6). A test datum $x$ is also a random variable. Given a test sample, our goal is to infer the random variable $y$  which is therefore referred to as our task.
>> 훈련 세트 $D=(x,y)$ 여기서 $x=x_{i=1}^{N}$ 및 $y=y_{i=1}^{N}$는 무작위로 샘플링된 N개의 데이터 포인트 $x_{i}^{i}$와 관련(일반적으로 이산) 레이블의 모음이다. 샘플은 매개 변수 $\theta$에 의해 매개 변수화된 알려지지 않은 복잡한 분포 $p_{\theta}(x,y)$에서 나온 것으로 가정한다. 베이지안 접근 방식에 따라, 우리는 $\theta$를 알 수 없는 일부 이전 분포 $p(\theta)$에서 샘플링한 무작위 변수로 간주하지만, 이 요구 사항은 필요하지 않다(섹션 6 참조). 검정 기준 $x$도 랜덤 변수입니다. 테스트 샘플이 주어지면, 우리의 목표는 무작위 변수 $y$를 추론하는 것이며, 따라서 우리의 과제라고 한다.

> We will make frequent use of the following standard information theoretic quantities (Cover and Thomas, 2012): Shannon entropy $H(x)=E_{p}[−\log{p(x)}]$, conditional entropy $H(x\vert{}y):=E_{\hat{y}}[H(x\vert{}y=\hat{y})]=H(x,y)-H(y)$, (conditional) mutual information $I(x;y\vert{}z)=H(x\vert{}z)-H(x\vert{}y,z)$, Kullbach-Leibler (KL) divergence $KL(p(x)\vert{}\vert{}q(x))=E_{p}[\log{}p/q]$, crossentropy $H_{p,q}(x)=E_{p}[−\log{}q(x)]$, and total correlation $TC(z)$, which is also known as multi-variate mutual information and defined as
>> 우리는 다음과 같은 표준 정보 이론적 양을 자주 사용할 것이다(Cover and Thomas, 2012). 섀넌 엔트로피 $H(x)=E_{p}[−\log{p(x)}]$, 조건부 엔트로피 $H(x\vert{}y):=E_{\hat{y}}[H(x\vert{}y=\hat{y})]=H(x,y)-H(y)$, (조건부) 상호 정보 $I(x;y\vert{}z)=H(x\vert{}z)-H(x\vert{}y,z)$, 쿨바흐-라이블러 (KL) 발산 $KL(p(x)\vert{}\vert{}q(x))=E_{p}[\log{}p/q]$, 교차 엔트로피 $H_{p,q}(x)=E_{p}[−\log{}q(x)]$, 그리고 총 상관 $TC(z)$는 다변량 상호 정보로도 알려져 있고 다음과 같이 정의된다.

$$TC(z)=KL(p(z)\vert{}\vert{}\prod_{i}p(z_{i})),$$

> where $p(z_{i})$ are the marginal distributions of the components of $z$. Recall that the KL divergence between two distributions is always non-negative and zero if and only if they are equal. In particular $TC(z)$ is zero if and only if the components of $z$ are independent, in which case we say that $z$ is disentangled. We often use of the following identity:
>> 여기서 $p(z_{i})$는 $z$ 성분의 주변 분포이다. 두 분포 사이의 KL 분기는 항상 음이 아니며 같은 경우에만 0이라는 점을 기억하십시오. 특히 $z$의 구성 요소가 독립적인 경우에만 $TC(z)$가 0이다. 이 경우 $z$가 분리되었다고 말한다. 우리는 종종 다음과 같은 아이덴티티를 사용한다.

$$I(z;x)=E_{x\sim{p(x)}}KL(p(z\vert{}x)\vert{}\vert{}p(z)).$$

> We say that $x$, $z$, $y$ form a Markov chain, indicated with $x\to{}z\to{}y$, if $p(y\vert{}x,z)=p(y\vert{}z)$. The Data Processing Inequality (DPI) for a Markov chain $x\to{}z\to{}y$ ensures that $I(x;z)\geq{}I(x;y)$: If $z$ is a (deterministic or stochastic) function of $x$, it cannot contain more information about $y$ than $x$ itself (we cannot create new information by simply applying a function to the data we already have).
>> 우리는 $x$, $z$, $y$가 $x\to{}z\to{}y$, if $p(y\vert{}x,z)=p(y\vert{}z)$로 표시된 마르코프 체인을 형성한다고 말합니다. 마르코프 체인 $x\to{}z\to{}y$에 대한 데이터 처리 불평등(DPI)은 $I(x;z)\geq{}I(x;y)$: $z$가 $x$의 (결정론적 또는 확률적) 함수라면 $x$ 자체보다 $y$에 대한 더 많은 정보를 포함할 수 없습니다(이미 보유한 데이터에 함수를 단순히 적용하기만 하면 새로운 정보를 생성할 수 없습니다).

#### $\mathbf{2.1\;General\;definitions\;and\;the\;Information\;Bottleneck\;Lagrangian}$

> We say that $z$ is a representation of $x$ if $z$ is a stochastic function of $x$, or equivalently if the distribution of $z$ is fully described by the conditional $p(z\vert{}x)$. In particular we have the Markov chain $y\to{}x\to{}z$. We say that a representation $z$ of $x$ is sufficient for $y$ if $y\perp{}x\vert{}z$, or equivalently if $I(z;y)=I(x;y)$; it is minimal when $I(x;z)$ is smallest among sufficient representations. To study the trade-off between sufficiency and minimality, Tishby et al. (1999) introduces the Information Bottleneck Lagrangian
>> 우리는 $z$가 $x$의 확률 함수인 경우, 또는 $z$의 분포가 조건 $p(z\vert{}x)$에 의해 완전히 설명되는 경우 $z$는 $x$의 표현이라고 말합니다. 특히 우리는 마르코프 연쇄 $y\to{}x\to{}z$를 가지고 있습니다. 우리는 $x$의 표현 $z$가 $y\perp{}x\vert{}z$인 경우 $y$에 충분하거나 $I(z;y)=I(x;y)$인 경우 동등하게 충분하다고 말합니다. $I(x;z)$가 충분한 표현 중 가장 작을 때 최소이다. 충분성과 최소성의 균형을 연구하기 위해, Tishby 외 연구진(1999)은 정보 병목 현상 라그랑지안을 소개합니다.

$$L(p(z\vert{}x))=H(y\vert{}z)+\beta{}I(z;x),$$

> where $\beta{}$ trades off sufficiency (first term) and minimality (second term); in the limit $\beta{}\to{}0$, the IB Lagrangian is minimized when $z$ is minimal and sufficient. It does not impose any restriction on disentanglement nor invariance, which we introduce next.
>> 여기서 $\beta{}$는 충분성(첫 번째 항)과 최소성(두 번째 항)을 절충한다. $\beta{}\to{}0$ 제한에서 IB 라그랑지안은 $z$가 최소이고 충분할 때 최소화된다. 그것은 우리가 다음에 소개하는 분리나 불변성에 어떠한 제한도 부과하지 않는다.

#### $\mathbf{2.2\;Nuisances\;for\;a\;task}$

> A nuisance is any random variable that affects the observed data $x$, but is not informative to the task we are trying to solve. More formally, a random variable $n$ is a nuisance for the task $y$ if $y\perp{}n$, or equivalently $I(y;n)=0$. Similarly, we say that the representation $z$ is invariant to the nuisance $n$ if $z\perp{}n$, or $I(z;n)=0$. When $z$ is not strictly invariant but it minimizes $I(z;n)$ among all sufficient representations, we say that the representation $z$ is **maximally insensitive** to $n$.
>> 방해는 관찰된 데이터 $x$에 영향을 미치지만 우리가 해결하려는 작업에 도움이 되지 않는 임의의 변수이다. 보다 공식적으로, 무작위 변수 $n$은 $y\perp{}n$인 경우 작업 $y$에 대한 성가신 요소이다. 또는 동등한 $I(y;n)=0$이다. 마찬가지로, $z\perp{}n$ 또는 $I(z;n)=0$인 경우 표현 $z$는 성가신 $n$에 대해 불변이라고 말한다. $z$가 엄격하게 불변하지는 않지만 모든 충분한 표현 중 $I(z;n)$를 최소화할 때, 우리는 $z$ 표현이 **n$에 대해 최대 무감각하다고 말한다.

> One typical example of nuisance is a group $G$. such as translation or rotation, acting on the data. In this case, a deterministic representation $f$ is invariant to the nuisances if and only if for all $g\in{}G$ we have $f(g·x)=f(x)$. Our definition however is more general in that it is not restricted to deterministic functions, nor to group nuisances. An important consequence of this generality is that the observed data $x$ can always be written as a deterministic function of the task $y$ and of all nuisances $n$ affecting the data, as explained by the following proposition.
>> 성가신 일의 대표적인 예로는 번역 또는 순환과 같은 그룹 $G$가 데이터에 작용한다. 이 경우, 결정론적 표현 $f$는 모든 $g\in{}G$에 대해 $f(g·x)=f(x)$를 갖는 경우에만 잡음에 불변한다. 그러나 우리의 정의는 결정론적 함수나 그룹 잡음에 제한되지 않는다는 점에서 더 일반적이다. 이 일반성의 중요한 결과는 관찰된 데이터 $x$가 다음 명제에 의해 설명되었듯이 항상 작업 $y$와 데이터에 영향을 미치는 모든 잡음 $n$의 결정론적 함수로 기록될 수 있다는 것이다.

> **Proposition 2.1 (Task-nuisance decomposition, Appendix C.1)** Given a joint distribution $p(x,y)$, where $y$ is a discrete random variable, we can always find a random variable $n$ independent of $y$ such that $x=f(y,n)$, for some deterministic function $f$.
>> **제안 2.1(작업 방해 분해, 부록 C.1)** $y$가 이산 랜덤 변수인 공동 분포 $p(x,y)$가 주어지면, 일부 결정론적 함수 $f$에 대해 $x=f(y,n)$가 되도록 $y$와 무관한 무작위 변수 $n$을 항상 찾을 수 있다.

### $\mathbf{3.\;Properties\;of\;optimal\;representations}$

> To simplify the inference process, instead of working directly with the observed high dimensional data $x$, we want to use a representation $z$ that captures and exposes only the information relevant for the task $y$. Ideally, such a representation should be (a) sufficient for the task $y$, i.e. $I(y;z)=I(y;x)$, so that information about $y$ is not lost; among all sufficient representations, it should be (b) minimal, i.e. $I(z;x)$ is minimized, so that it retains as little about $x$ as possible, simplifying the role of the classifier; finally, it should be (c) invariant to the effect of nuisances $I(z;n)=0$, so that the final classifier will not overfit to spurious correlations present in the training dataset between nuisances $n$ and labels $y$. Such a representation, if it exists, would not be unique, since any bijective mapping preserves all these properties. We can use this to our advantage and further aim to make the representation (d) maximally disentangled, i.e., choose the one(s) for which $TC(z)$ is minimal. This simplifies the classifier rule, since no information will be present in the higher-order correlations between the components of $z$. 
>> 추론 프로세스를 단순화하기 위해 관찰된 고차원 데이터 $x$로 직접 작업하는 대신 작업 $y$와 관련된 정보만 캡처하고 노출하는 표현 $z$를 사용하려고 합니다. 이상적으로, 그러한 표현은 (a) $y$, 즉 $I(y;z)=I(y;x)$에 대한 정보가 손실되지 않도록 하기 위해 $y$ 작업에 충분해야 합니다. 즉, 충분한 모든 표현 중에서 (b) 최소여야 합니다. 즉, $I(z;x)$가 최소화되어 분류자의 역할을 가능한 한 적게 유지하여 (마지막으로 불변해야 합니다.o 소음 $I(z;n)=0$의 효과로, 최종 분류기가 소음 $n$과 레이블 $y$ 사이의 훈련 데이터 세트에 존재하는 가짜 상관 관계에 과적합하지 않는다. 이러한 표현은, 존재한다면, 모든 주관적 매핑이 이러한 모든 속성을 보존하기 때문에 고유하지 않을 것입니다. 우리는 이것을 우리에게 유리하게 사용할 수 있고 표현(d)을 최대적으로 분리하는 것을 더 목표로 할 수 있습니다. 즉, $TC(z)$가 최소인 것을 선택한다. 이렇게 하면 $z$의 구성 요소 간의 고차 상관 관계에 정보가 없으므로 분류자 규칙이 단순화됩니다.

> Inferring a representation that satisfies all these properties may seem daunting. However,  in this section we show that we only need to enforce (a) sufficiency and (b) minimality, from which invariance and disentanglement follow naturally thanks to the stacking of noisy layers of computation in deep networks. We will then show that sufficiency and minimality of the learned representation can also be promoted easily through implicit or explicit regularization during the training process.
>> 이러한 모든 속성을 만족시키는 표현을 추론하는 것은 위압적으로 보일 수 있다. 그러나 이 섹션에서는 (a) 충분성과 (b) 최소성만 적용하면 된다는 것을 보여주는데, 심층 네트워크에서 노이즈가 많은 계산 계층이 쌓이기 때문에 불변성과 분리가 자연스럽게 뒤따른다. 그런 다음 학습된 표현의 충분성과 최소성도 훈련 과정 동안 암시적 또는 명시적 정규화를 통해 쉽게 촉진될 수 있음을 보여줄 것이다.

> **Proposition 3.1 (Invariance and minimality, Appendix C.2)** Let $n$ be a nuisance for the task $y$ and let $z$ be a sufficient representation of the input $x$. Suppose that $z$ depends on $n$ only through $x$ ( i.e., $n\to{}x\to{}z$). Then,
>> **제안 3.1(불변성과 최소성, 부록 C.2)** $n$이 작업 $y$에 대한 성가신 존재가 되고 $z$가 입력 $x$의 충분한 표현이 되도록 한다. $z$가 $x$를 통해서만 $n$에 의존한다고 가정하자(즉, $n\to{}x\to{}z$). 그리고나서,

$$I(z;n)\leq{}I(z;x)-I(x;y).$$

> Moreover, there is a nuisance $n$ such that equality holds up to a (generally small) residual $\epsilon$
>> 또한 동일성이 (일반적으로 작은) 잔차 $\epsilon$까지 유지되는 성가신 $n$이 있다.

$$I(z;n)=I(z;x)-I(x;y)-\epsilon{}$$

> where $\epsilon{}:=I(z;y\vert{}n)-I(x;y)$. In particular $0\leq{}\epsilon{}\leq{}H(y\vert{}x)$, and $\epsilon=0$ whenever $y$ is a deterministic function of $x$. Under these conditions, a sufficient statistic $z$ is invariant (maximally insensitive) to nuisances if and only if it is minimal. 
>> $\epsilon{}:=I(z;y\vert{}n)-I(x;y)$입니다. 특히 $0\leq{}\epsilon{}\leq{}H(y\vert{}x)$, 그리고 $y$가 $x$의 결정론적 함수일 때마다 $\silon=0$입니다. 이러한 조건에서 충분한 통계 $z$는 최소인 경우에만 소음에 대해 불변(최대 무감각)합니다.

> **Remark 3.2 Since** $\epsilon{}\leq{}H(y\vert{}x)$, and usually $H(y\vert{}x)=0$ or at least $H(y\vert{}x)\ll{}I(x;z)$, we can generally ignore the extra term.
>> ** 이후 3.2 참고** $\epsilon{}\leq{}H(y\vert{}x)$와 일반적으로 $H(y\vert{}x)=0$ 또는 적어도 $H(y\vert{}x)\ll{}I(x;z)$는 일반적으로 추가 용어를 무시할 수 있다.

> An important consequence of this proposition is that we can construct invariants by simply reducing the amount of information $z$ contains about $x$, while retaining the minimum amount $I(z;x)$ that we need for the task $y$. This provides the network a way to automatically learn invariance to complex nuisances, which is complementary to the invariance imposed by the architecture. Specifically, one way of enforcing minimality explicitly, and hence invariance, is through the IB Lagrangian.
>> 이 명제의 중요한 결과는 $y$ 작업에 필요한 최소량 $I(z;x)$를 유지하면서 $z$가 약 $x$를 포함하는 정보의 양을 줄임으로써 불변량을 구성할 수 있다는 것이다. 이것은 네트워크가 복잡한 잡음에 대한 불변성을 자동으로 학습할 수 있는 방법을 제공하는데, 이는 아키텍처에 의해 부과되는 불변성을 보완한다. 구체적으로, 최소화를 명시적으로 시행하고 따라서 불변성을 적용하는 한 가지 방법은 IB 라그랑지안을 통해서이다.

> **Corollary 3.3 (Invariants from the Information Bottleneck)** Minimizing the IB Lagrangian
>> **Colorary 3.3(정보 병목 현상의 변화)** IB Lagrangian 최소화

$$L(p(z\vert{}x))=H(y\vert{}z)+\beta{}I(z;x),$$

> in the limit $\beta{}\to{}0$, yields a sufficient invariant representation $z$ of the test datum $x$ for the task $y$.
>> $\beta{}\to{}0$ 제한에서, 작업 $y$에 대한 테스트 데이터 $x$의 충분한 불변 표현 $z$를 산출한다.

> Remarkably, the IB Lagrangian can be seen as the standard cross-entropy loss, plus a regularizer $I(z;x)$ that promotes invariance. This fact, without proof, is implicitly used in Achille and Soatto (2018), who also provide an efficient algorithm to perform the optimization. Alemi et al. (2017a) also propose a related algorithm and empirically show improved resistance to adversarial nuisances. In addition to modifying the cost function, invariance can also be fostered by choice of architecture:
>> 놀랍게도, IB 라그랑지안은 표준 교차 엔트로피 손실과 불변성을 촉진하는 정규화기 $I(z;x)$로 볼 수 있다. 이 사실은 증거 없이 아킬과 수토(2018)에서 암묵적으로 사용되며, 이들은 또한 최적화를 수행하기 위한 효율적인 알고리듬을 제공한다. Alemi 외 연구진(2017a)도 관련 알고리듬을 제안하고 적대적 소음에 대한 개선된 내성을 경험적으로 보여준다. 비용 함수를 수정하는 것 외에도, 불변성은 아키텍처 선택에 의해서도 촉진될 수 있다.

> **Corollary 3.4 (Bottlenecks promote invariance)** Suppose we have the Markov chain of layers
>> **상관 3.4(병목은 불변성을 촉진함)** 우리가 층의 마르코프 연쇄를 가지고 있다고 가정하자.

$$x\to{}z_{1}\to{}z_{2}$$

> and suppose that there is a communication or computation bottleneck between $z_{1}$ and $z_{2}$ such that $I(z_{1};z_{2})<I(z_{1};x)$. Then, if $z_{2}$ is still sufficient, it is more invariant to nuisances than $z_{1}$. More precisely, for all nuisances $n$ we have $I(z_{2};n)\leq{}I(z_{1};z_{2})-I(x;y)$. Such a bottleneck can happen for example because $\dim(z_{2})<\dim(z_{1})$, e.g., after a pooling layer, or because the channel between $z_{1}$ and $z_{2}$ is noisy, e.g., because of dropout.
>> $I(z_{1};z_{2})<I(z_{1};x)$와 같이 $z_{1}$와 $z_{2}$ 사이에 통신 또는 계산 병목 현상이 있다고 가정합니다. 그렇다면 $z_{2}$가 여전히 충분하다면 $z_{1}$보다 성가신 일에 더 불변합니다. 더 정확히 말하면, 모든 성가신 $n$에 대해 우리는 $I(z_{2};n)\leq{}I(z_{1};z_{2})-I(x;y)$를 가지고 있습니다. 이러한 병목 현상은 예를 들어 $\dim(z_{2})<\dim(z_{1})$가 풀링 계층 이후에 발생하거나 $z_{1}$와 $z_{2}$ 사이의 채널이 노이즈(예: 드롭아웃)로 인해 발생할 수 있습니다.

> **Proposition 3.5** (Stacking increases invariance) Assume that we have the Markov chain of layers
>> **제안 3.5**(스택은 불변성을 증가시킨다) 층의 마르코프 연쇄를 갖는다고 가정하자.

$$x\to{z_{1}}\to{}z_{2}\to{}\cdots{}\to{}z_{L},$$

> and that the last layer $z{L}$ is sufficient of $x$ for $y$. Then $z{L}$ is more insensitive to nuisances than all the preceding layers.
>> 그리고 마지막 계층 $z{L}$가 $y$에 대해 $x$로 충분하다는 것을 의미한다. 그러면 $z{L}$는 이전의 모든 계층보다 소음에 더 민감하다.

> Notice, however, that the above corollary does not simply imply that the more layers the merrier, as it assumes that one has successfully trained the network ($z{L}$ is sufficient), which becomes increasingly difficult as the size grows. Also note that in some architectures, such as ResNets (He et al., 2016), the layers do not necessarily form a Markov chain because of skip connections; however, their “blocks” still do.
>> 그러나 위의 상관관계는 단순히 더 많은 계층이 될수록 네트워크($z{L}$이면 충분함)를 성공적으로 훈련시켰다고 가정하기 때문에 더 많은 계층이 있다는 것을 의미하지는 않으며, 이는 크기가 커질수록 점점 더 어려워진다. 또한 ResNets(He et al., 2016)와 같은 일부 아키텍처에서 계층은 건너뛰기 연결 때문에 반드시 마르코프 체인을 형성하지는 않지만, "블록"은 여전히 형성된다.

> **Proposition 3.6 (Actionable Information)** When $z=f(x)$ is a deterministic invariant, if it minimizes the IB Lagrangian it also maximizes Actionable Information (Soatto, 2013), which is $H(x):=H(f(x))$.
>> **제안 3.6(실행 가능한 정보)**$가 결정론적 불변량일 때, $z=f(x)$가 IB 라그랑주(Lagrangian)를 최소화하는 경우, 또한 실행 가능한 정보(Soato, 2013)를 최대화합니다($H(x):=H(f(x))$).

> Although Soatto (2013) addressed maximal invariants, we only consider sufficient invariants, a advocated by (Soatto and Chiuso, 2016).
>> Soatto(2013)가 최대 불변량을 다루었지만, 우리는 (Soatto and Chiuso, 2016)가 옹호하는 충분한 불변량만을 고려한다.

#### $\mathbf{Information\;in\;the\;weights}$

> Thus far we have discussed properties of representations in generality, regardless of how they are implemented or learned. Given a source of data (for example randomly generated, or from a fixed dataset), and given a (stochastic) training algorithm, the output weight $w$ of the training process can be thought as a random variable (that depends on the stochasticity of the initialization, training steps and of the data). We can therefore talk about the information that the weights contain about the dataset $D$ and the training procedure, which we denote by $I(w;D)$.
>> 지금까지 우리는 표현들이 어떻게 구현되거나 학습되는지에 관계없이 일반적으로 표현의 특성에 대해 논의하였다. 데이터 소스(예: 무작위로 생성되거나 고정된 데이터 세트에서)와 (가설적) 훈련 알고리듬이 주어진 경우, 훈련 프로세스의 출력 가중치 $w$는 무작위 변수(초기화, 훈련 단계 및 데이터의 확률성에 따라 달라짐)로 생각할 수 있다. 따라서 가중치가 포함하는 데이터 세트 $D$와 훈련 절차에 대한 정보에 대해 이야기할 수 있으며, 이는 $I(w;D)$로 표시된다.

> Two extreme cases consist of the trivial settings where we use the weights to memorize he dataset (the most extreme form of overfitting), or where the weights are constant, or pure noise (sampled from a process that is independent of the data). In between, the amount of information the weights contain about the training turns out to be an important quantity both in training deep networks, as well as in establishing properties of the resulting representation, as we discuss in the next section.
>> 두 가지 극단적인 경우는 가중치를 사용하여 데이터 세트를 기억하거나(가장 극단적인 형태의 과적합), 가중치가 일정하거나(데이터와 독립적인 프로세스에서 샘플링된) 순수한 노이즈인 사소한 설정으로 구성된다. 그 사이에 가중치에 포함된 훈련에 대한 정보의 양은 다음 섹션에서 논의한 바와 같이 심층 네트워크를 훈련하고 결과 표현의 속성을 설정하는 데 중요한 양으로 밝혀졌다.

> Note that in general we do not need to compute and optimize the quantity of information in the weights. Instead, we show that we can control it, for instance by injecting noise in the weights, drawn from a chosen distribution, in an amount that can be modulated between zero (thus in theory allowing full information about the training set to be stored in the weights) to an amount large enough that no information is left. We will leverage this property in the next sections to perform regularization.
>> 일반적으로 가중치에서 정보의 양을 계산하고 최적화할 필요가 없습니다. 대신, 우리는 예를 들어 선택된 분포에서 도출된 가중치에 0(따라서 이론상 훈련 세트에 대한 전체 정보를 가중치에 저장할 수 있음) 사이에서 변조할 수 있는 양의 노이즈를 정보가 남지 않을 정도로 충분히 큰 양으로 주입함으로써 이를 제어할 수 있음을 보여준다. 다음 섹션에서는 이 속성을 활용하여 정규화를 수행할 것입니다.

### $\mathbf{4.\;Learning\;minimal\;weights}$

> In this section, we let $p_{\theta}(x,y)$ be an (unknown) distribution from which we randomly sample a dataset $D$. The parameter $\theta$ of the distribution is also assumed to be a random variable with an (unknown) prior distribution $p(\theta)$. For example $p_{\theta}$ can be a fairly general generative model for natural images, and $\theta$ can be the parameters of the model that generated our dataset. We then consider a deep neural network that implements a map $x\to{}fw(x):=q(·\vert{}x,w)$ from an input $x$ to a class distribution $q(y\vert{}x,w)$.1 In full generality, and following a Bayesian approach, we let the weights $w$ of the network be sampled from a parametrized distribution $q(w\vert{}D)$,whose parameters are optimized during training.2 The network is then trained in order to minimize the expected cross-entropy loss 3
>> 이 섹션에서는 $p_{\theta}(x,y)$가 데이터 세트 $D$를 랜덤하게 샘플링하는 (알 수 없는) 분포가 되도록 합니다. 분포의 모수 $\theta$도 (알 수 없는) 이전 분포 $p(\theta)$를 갖는 랜덤 변수로 가정됩니다. 예를 들어, $p_{\theta}$는 자연 이미지에 대한 상당히 일반적인 생성 모델일 수 있으며 $\theta$는 데이터 세트를 생성한 모델의 매개 변수일 수 있습니다. 그런 다음 입력 $x$에서 클래스 분포 $q(y\vert{}x,w)$1에 대한 맵 $x\to{}fw(x):=q(·\vert{}x,w)$를 구현하는 심층 신경망을 고려하며, 전체 일반적이며 베이지안 접근법에 따라 매개 변수가 최적화된 분포 $q(w\vert{}D)$에서 네트워크의 가중치 $w$를 샘플링한다.2 그런 다음 예상되는 교차 엔트로피 손실을 최소화하기 위해 네트워크가 학습됩니다 3

$$H_{p,q}(y\vert{}x,w)=E_{D=(x,y)}E_{W\sim{}(w\vert{}D)}\sum_{i=1}^{N}-\log{}q(y^{i}\vert{}x^{i},w),$$

> in order for $q(y\vert{}x,w)$ to approximate $p_{\theta}(y\vert{}x)$.
>> $q(y\vert{}x,w)$가 $p_{\theta}(y\vert{}x)$에 근사하도록 하기 위해.

> One of the main problems in optimizing a DNN is that the cross-entropy loss in notoriously prone to overfitting. In fact, one can easily minimize it even for completely random labels (see Zhang et al. (2017), and Figure 1). The fact that, somehow, such highly over-parametrized functions manage to generalize when trained on real labels has puzzled theoreticians and prompted some to wonder whether this may be inconsistent with the intuitive interpretation of the bias-variance trade-off theorem, whereby unregularized complex models should overfit wildly. However, as we show next, there is no inconsistency if one measures complexity by the information content, and not the dimensionality, of the weights.
>> DNN 최적화의 주요 문제 중 하나는 교차 엔트로피 손실이 과적합되기 쉽기로 악명 높다는 것이다. 사실, 완전히 무작위 레이블에 대해서도 쉽게 최소화할 수 있다(장 외(2017년) 및 그림 1 참조). 어떻게든, 실제 레이블에 대해 훈련할 때 그러한 고도로 매개 변수화된 함수가 가까스로 일반화된다는 사실은 이론가들을 어리둥절하게 만들었고, 일부 사람들은 이것이 비정규화된 복잡한 모델이 과도하게 적합해야 하는 편향-분산 트레이드오프 정리의 직관적인 해석과 일치하지 않을 수 있는지 의문을 갖게 했다. 그러나 다음에 보여 주듯이, 가중치의 차원이 아닌 정보 내용에 의해 복잡성을 측정한다면 일관성이 없다.

> To gain some insights about the possible causes of over-fitting, we can use the following decomposition of the cross-entropy loss (we refer to Appendix C for the proof and the precise definition of each term):
>> 과적합의 가능한 원인에 대한 통찰력을 얻기 위해, 우리는 교차 엔트로피 손실의 다음과 같은 분해를 사용할 수 있다(각 항의 증명과 정확한 정의는 부록 C 참조).

$$H_{p,q}(y\vert{}x,w)=\underbrace{H(y\vert{}x,\theta{})}_{intrinsic\;error}+\underbrace{I(\theta{};y\vert{}x,w)}_{sufficiency}+E_{x,w}\underbrace{KL(p(y\vert{}x,w)\vert{}\vert{}q(y\vert{}x,w))}_{efficiency}-\underbrace{I(y:w\vert{}x,\theta{})}_{overfitting}$$

> The first term of the right-hand side of (8) relates to the intrinsic error that we would commit in predicting the labels even if we knew the underlying data distribution $p_{\theta}$; the second term measures how much information that the dataset has about the parameter $\theta$ is captured by the weights, the third term relates to the efficiency of the model and the class of functions fw with respect to which the loss is optimized. The last, and only negative, term relates to how much information about the labels, but uninformative of the underlying data distribution, is memorized in the weights. Unfortunately, without implicit or explicit regularization, the network can minimize the cross-entropy loss (LHS), by just maximizing the last term of eq. (8), i.e., by memorizing the dataset, which yields poor generalization.
>> (8)의 오른쪽 첫 번째 항은 기본 데이터 분포 $p_{\theta}$를 알고 있더라도 레이블을 예측하는 데 저지르는 본질적인 오류와 관련이 있다. 두 번째 항은 매개 변수 $\theta$에 대해 데이터 세트가 가진 정보가 가중치에 의해 캡처되는 정도를 측정한다. 세 번째 항은 효율과 관련이 있다.손실이 최적화되는 모델 및 함수 클래스 fw의 y입니다. 마지막 부정적인 용어는 레이블에 대한 정보량이지만 기본 데이터 분포에 대해서는 알 수 없는 것과 관련이 있다. 불행히도, 암시적 또는 명시적 정규화 없이 네트워크는 (8)의 마지막 항, 즉 데이터 세트를 기억함으로써 교차 엔트로피 손실(LHS)을 최소화할 수 있으며, 이는 열악한 일반화를 산출한다.

> To prevent the network from doing this, we can neutralize the effect of the negative term by adding it back to the loss function, leading to a regularized loss $L=H_{p},q(y\vert{}x,w)+I(y;w\vert{}x,\theta)$. However, computing, or even approximating, the value of $I(y, w\vert{}x,\theta)$ is at least as difficult as fitting the model itself. 
>> 네트워크가 이렇게 하는 것을 막기 위해, 우리는 손실 함수에 다시 추가함으로써 음의 항의 효과를 무력화시켜 정규화된 손실 $L=H_{p},q(y\vert{}x,w)+I(y;w\vert{}x,\theta)$로 이어질 수 있다. 그러나 계산 또는 근사치 $I(y, w\vert{}x,\theta)$의 값은 적어도 모델 자체를 적합시키는 것만큼 어렵다.

> We can, however, add an upper bound to $I(y;w\vert{}x,\theta)$ to obtain the desired result. In particular, we explore two alternate paths that lead to equivalent conclusions under different premises and assumptions: In one case, we use a PAC-Bayes upper-bound, which is $KL(q(w\vert{}D)\vert{}\vert{}p(w))$ where $p(w)$ is an arbitrary prior. In the other, we use the IB Lagrangian and upper-bound it with the information in the weights $I(w;D)$. We discuss this latter approach now, and look at the PAC-Bayes approach in Section 6.
>> 그러나 $I(y;w\vert{}x,\theta)$에 상한을 추가하여 원하는 결과를 얻을 수 있다. 특히, 우리는 서로 다른 전제 및 가정 하에서 동등한 결론을 도출하는 두 가지 대체 경로를 탐구한다. 한 가지 경우, 우리는 PAC-Bayes 상한선을 사용하는데, 이는 $KL(q(w\vert{}D)\vert{}\vert{}p(w))$이며, 여기서 $p(w)$는 임의의 선행이다. 다른 방법에서는 IB 라그랑지안을 사용하고 가중치 $I(w;D)$의 정보로 상한에 둔다. 이제 이 후자의 접근 방식에 대해 논의하고, 6절에서 PAC-Bayes 접근 방식을 살펴본다.

> Notice that to successfully learn the distribution $p_{\theta}$, we only need to memorize in $w$ the information about the latent parameters $\theta$, that is we need $I(D;w)=I(D;\theta)\leq{}H(\theta)$, which is bounded above by a constant. On the other hand, to overfit, the term $I(y;w\vert{}x,\theta)\leq{}I(D; w\vert{}\theta)$ needs to grow linearly with the number of training samples $N$. We can exploit this fact to prevent overfitting by adding a Lagrange multiplier $\beta{}$ to make the amount of information a constant with respect to $N$, leading to the regularized loss function
>> 분포 $p_{\theta}$를 성공적으로 학습하려면 잠재 매개 변수 $\theta$에 대한 정보만 $w$에 암기하면 된다는 점에 주목하십시오. 즉, 상수에 의해 위에 제한되는 $I(D;w)=I(D;\theta)\leq{}H(\theta)$가 필요합니다. 반면, 오버핏하기 위해 용어 $I(y;w\vert{}x,\theta)\leq{}I(D; w\vert{}\theta)$는 훈련 샘플 수 $N$에 따라 선형적으로 증가해야 합니다. Lagrange 곱셈기 $\beta{}$를 추가하여 정보의 양을 $N$에 대해 일정하게 하여 정규화된 손실 함수로 이어짐으로써 이 사실을 이용하여 과적합을 방지할 수 있습니다.

$$L(q(w\vert{}D))=H_{p,q}(y\vert{}x,w)+\beta{}I(w;D),$$

> which, remarkably, has the same general form of an IB Lagrangian, and in particular is similar to (1), but now interpreted as a function of the weights $w$ rather than the activations $z$. This use of the IB Lagrangian is, to the best of our knowledge, novel, as the role of the Information Bottleneck has thus far been confined to characterizing the activations of the network, and not as a learning criterion. Equation (3) can be seen as a generalization of other suggestions in the literature:
>놀랍게도, >는 IB 라그랑지안의 일반적인 형태를 가지고 있으며, 특히 (1)과 유사하지만, 지금은 활성화 $z$가 아닌 가중치 $w$의 함수로 해석된다. 지금까지 정보 병목 현상의 역할은 학습 기준이 아니라 네트워크의 활성화를 특성화하는 데 국한되었기 때문에 IB 라그랑지안의 이러한 사용은 우리가 아는 한 새로운 것이다. 식 (3)은 문헌에서 다른 제안들의 일반화로 볼 수 있다.

> **IB Lagrangian, Variational Learning and Dropout.** Minimizing the information stored at the weights $I(w;D)$ was proposed as far back as Hinton and Van Camp (1993) as a way of simplifying neural networks, but no efficient algorithm to perform the optimization was known at the time. For the particular choice $\beta{}=1$, the IB Lagrangian reduces to the variational lower-bound (VLBO) of the marginal log-likelihood $p(y\vert{}x)$. Therefore, minimizing eq. (3) can also be seen as a generalization of variational learning. A particular case of this was studied by Kingma et al. (2015), who first showed that a generalization of Dropout, called Variational Dropout, could be used in conjunction with the reparametrization trick Kingma and Welling (2014) to minimize the loss efficiently.
>> **IB Lagrangian, Variational Learning 및 중퇴.** 가중치 $I(w;D)$에 저장된 정보를 최소화하는 것은 신경망을 단순화하는 방법으로 힌튼과 밴 캠프(1993)만큼 오래 전에 제안되었지만, 당시에는 최적화를 수행하는 효율적인 알고리듬이 알려져 있지 않았다. 특정 선택 $\sigma{}=1$의 경우, IB 라그랑지안은 한계 로그 우도 $p(y\vert{}x)$의 변동 하한(VLBO)으로 감소한다. 따라서 등(3)을 최소화하는 것은 변이 학습의 일반화로도 볼 수 있다. 이것의 특정 사례는 킹마 외 연구진(2015)에 의해 연구되었는데, 그는 변분적 드롭아웃이라고 불리는 드롭아웃의 일반화가 손실을 효율적으로 최소화하기 위해 리파라미터화 트릭 킹마와 웰링(2014)과 함께 사용될 수 있다는 것을 처음 보여주었다.

> **Information in the weights as a measure of complexity.** Just as Hinton and Van Camp (1993) suggested, we also advocate using the information regularizer $I(w;D)$ as a measure of the effective complexity of a network, rather than the number of parameters dim(w), which is merely an upper bound on the complexity. As we show in experiments, this allows us to recover a version of the bias-variance trade-off where networks with lower information complexity underfit the data, and networks with higher complexity overfit. In contrast, there is no clear relationship between number of parameters and overfitting (Zhang et al., 2017). Moreover, for random labels the information complexity allows us to precisely predict the overfitting and underfitting behavior of the network (Section 7)
>> **복잡도의 척도로서의 가중치 정보.** 힌튼과 밴 캠프(1993)가 제안한 것처럼, 우리는 또한 복잡성의 상한에 불과한 매개 변수 dim(w)보다 네트워크의 효과적인 복잡성의 척도로 정보 정규화 $I(w;D)$를 사용하는 것을 지지한다. 실험에서 보여 주듯이, 이는 정보 복잡성이 낮은 네트워크가 데이터에 적합하지 않은 편향-분산 트레이드오프 버전과 복잡성이 높은 네트워크가 오버핏된 편향-분산 트레이드오프의 버전을 복구할 수 있게 한다. 대조적으로, 매개 변수의 수와 과적합 사이에는 명확한 관계가 없다(Zhang et al., 2017). 또한 랜덤 레이블의 경우 정보 복잡성을 통해 네트워크의 과적합 및 과소적합 동작을 정확하게 예측할 수 있다(섹션 7).

#### $\mathbf{4.1\;Computable\;upper-bound\;to\;the\;loss}$

> Unfortunately, computing $I(w,D)=E_{D}KL(q(w\vert{}D)KL(q(w\vert{}D)\vert{}\vert{}q(w))$ is still too complicated, since it requires us to know the marginal $q(w)$ over all possible datasets and trainings of the network. To avoid computing this term, we can use the more general upper-bound
>> 불행하게도, 네트워크의 모든 데이터 세트와 훈련에 대한 한계 $q(w)$를 알아야 하기 때문에 $I(w,D)=E_{D}KL(q(w\vert{}D)\vert{}q(w))$를 계산하는 것은 여전히 너무 복잡하다. 이 항을 계산하지 않기 위해, 우리는 더 일반적인 상한을 사용할 수 있다.

$$E_{D}KL(q(w\vert{}D)\vert{}\vert{}q(w))\leq{}E_{D}KL(q(w\vert{}D)\vert{}\vert{}q(w))+KL(q(w)\vert{}\vert{}p(w))=E_{D}KL(q(w\vert{}D)\vert{}\vert{}p(w)),$$

> where $p(w)$ is any fixed distribution of the weights. Once we instantiate the training set, we have a single sample of $D$, so the expectation over $D$ becomes trivial. This gives us the following upper bound to the optimal loss function
>> 여기서 $p(w)$는 가중치의 고정 분포이다. 훈련 세트를 인스턴스화하면 $D$의 단일 샘플이 있으므로 $D$에 대한 기대는 사소한 것이 된다. 이것은 최적 손실 함수에 대한 다음과 같은 상한을 제공한다.

$$L(q(w\vert{}D))=H_{p,q}(y\vert{}x,w)+\beta{}KL(q(w\vert{}D)\vert{}\vert{}p(w))$$

> Generally, we want to pick $p(w)$ in order to give the sharpest upper-bound, and to be a fully factorized distribution, i.e., a distribution with independent components, in order to make the computation of the KL term easier. The sharpest upper-bound to $KL(q(w\vert{}D)\vert{}\vert{}q(w))$ that can be obtained using a factorized distribution $p$ is obtained when $p(w):=\tilde{q}(w)=\prod_{i}q(w_{i})$ where $q(w_{i})$ denotes the marginal distributions of the components of $q(w)$. Notice that. once a training procedure is fixed, this may be approximated by training multiple times and approximating each marginal weight distribution. With this choice of prior, our final loss function becomes
>> 일반적으로, 우리는 가장 날카로운 상한을 제공하기 위해 $p(w)$를 선택하고, KL 항의 계산을 더 쉽게 하기 위해 완전히 인수 분해된 분포, 즉 독립 성분을 가진 분포를 선택하려고 한다. 인수 분해 분포 $p$를 사용하여 얻을 수 있는 $p(w):=\tilde{q}(w)=\prod_{i}q(w_{i})$에 대한 가장 날카로운 상한은 $q(w_{i})$가 $q(w)$의 구성 요소의 주변 분포를 나타낼 때 얻는다. 알아두세요. 일단 훈련 절차가 고정되면, 이것은 여러 번 훈련하고 각 한계 체중 분포를 근사화함으로써 근사치를 구할 수 있다. 이러한 사전 선택으로, 우리의 최종 손실 함수는

$$L(q(w\vert{}D))=H_{p,q}(y\vert{}x,w)+\beta{}KL(q(w\vert{}D)\vert{}\vert{}\tilde{q}(w))$$

> for some fixed distribution $\tilde{q}$ that approximates the real marginal distribution $q(w)$. The IB Lagrangian for the weights in eq. (3) can be seen as a generally intractable special case of eq. (5) that gives the sharpest upper-bound to our desired loss in this family of losses.
>> 실제 한계 분포 $q(w)$에 근접한 일부 고정 분포 $\tilde{q}$에 대해. 등(3)의 가중치에 대한 IB 라그랑지안은 일반적으로 다루기 어려운 특수한 경우로 볼 수 있으며, 이 손실 제품군에서 원하는 손실에 대해 가장 날카로운 상한을 제공한다.

> In the following, to keep the notation uncluttered, we will denote our upper bound $KL(q(w\vert{}D)\vert{}\vert{}\tilde{q}(w))$ to the mutual information $I(w;D)$ simply by $\tilde{I}(w;D)$, where
>> 다음에서, 표기법을 흐트러짐 없이 유지하기 위해, 우리는 상호 정보 $I(w;D)$에 대한 우리의 상한 $KL(q(w\vert{}D)\vert{}\vert{}\tilde{q}(w))$를 단순히 $\tilde{I}(w;D)$에 의해 나타낼 것이다.

$$\tilde{I}(w;D):=KL(q(w\vert{}D)\vert{}\vert{}\tilde{q}(w)=KL(q(w\vert{}D)\vert{}\vert{}\prod_{i}q(w_{i})).$$

#### $\mathbf{4.2\;Bounding\;the\;information\;in\;the\;weights\;of\;a\;network}$

> To derive precise and empirically verifiable statements about $\tilde{I}(w;D)$, we need a setting where this can be expressed analytically and optimized efficiently on standard architectures. To this end, following Kingma et al. (2015), we make the following modeling choices.
>> $\tilde{I}(w;D)$에 대한 정확하고 경험적으로 검증 가능한 진술을 도출하기 위해서는 표준 아키텍처에서 이를 분석적으로 표현하고 효율적으로 최적화할 수 있는 설정이 필요합니다. 이를 위해 킹마 외(2015)에 이어 다음과 같은 모델링 선택을 합니다.

> **Modeling assumptions.** Let $w$ denote the vector containing all the parameters (weights) in the network, and let $W^{k}$ denote the weight matrix at layer k. We assume an improper log-uniform prior on w, that is $\tilde{q}(wi)=c/\vert{}wi\vert{}$. Notice that this is the only scale-invariant prior (Kingma et al., 2015), and closely matches the real marginal distributions of the weights in a trained network (Achille and Soatto, 2018); we parametrize the weight distribution $q(w_{i}\vert{}D)$ during training as
>> **모델링 가정.** $w$가 네트워크의 모든 매개 변수(가중치)를 포함하는 벡터를 나타내도록 하고, $W^{k}$가 계층 k의 가중치 행렬을 나타내도록 한다. w, 즉 $\tilde{q}(wi)=c/\vert{}wi\vert{}$에 앞서 부적절한 로그 균일성을 가정한다. 이것은 유일한 척도 불변 사전이며(킹마 외, 2015), 훈련된 네트워크에서 가중치의 실제 한계 분포와 밀접하게 일치한다(아킬 및 소토, 2018). 우리는 훈련 중 가중치 분포 $q(w_{i}\vert{}D)$를 다음과 같이 매개 변수화한다.

$$w_{i}\vert{}D\sim{}\epsilon_{i}\hat{w},$$

> where $\hat{w}$ is a learned mean, and $\epsilon_{i}\sim{}\log{}N(−\alpha{}/2,\alpha{})$ is i.i.d. multiplicative log-normal noise with mean 1 and variance $\exp(\alpha{})$−1.4 Note that while Kingma et al. (2015) uses this arametrization as a local approximation of the Bayesian posterior for a given (log-uniform) prior, we rather define the distribution of the weights $w$ after training on the dataset $D$ to be $q(w\vert{}D)$.
>> 여기서 $\hat{w}$는 학습된 평균이고, $\epsilon_{i}\sim{}\log{}N(−\alpha{}/2,\alpha{})$는 평균 1과 분산 $\exp(\alpha{}+{i})$-1.4인 곱셈 로그 정규 노이즈이다. 킹마 외(2015)는 이 매개 변수화를 주어진(로그 균일) 이전에 베이지안 사후(basian postarial)의 로컬 근사치로 사용하지만, 우리는 오히려 $w$ 훈련 후 가중치 분포를 정의한다.$D$는 $q(w\vert{}D)$가 된다.

> **Proposition 4.1 (Information in the weights, Theorem C.4)** Under the previous modeling assumptions, the upper-bound to the information that the weights contain about the dataset is
>> **제안 4.1(무게에 대한 정보, 정리 C.4)** 이전 모델링 가정 하에서 가중치가 데이터 집합에 대해 포함하는 정보에 대한 상한은

$$I(w;D)\leq\tilde{I}(w;D)=-\frac{1}{2}\sum_{i=1}^{\dim{(w)}}\log{}\alpha_{i}+C,$$

> where the constant C is arbitrary due to the improper prior.
>> 여기서 상수 C는 부적절한 선행으로 인해 임의적이다.

> **Remark 4.2 (On the constant C)** To simplify the exposition, since the optimization is unaffected by any additive constant, in the following we abuse the notation and, under the modeling assumptions stated above, we rather define $\tilde{I}(w;D):=-\frac{1}{2}\sum_{i=1}^{\dim(w)}\log{}\alpha{}$. Neklyudov et al. (2017) also suggest a principled way of dealing with the arbitrary constant by using a proper log-uniform prior.
>> ***4.2 (상수 C에 대하여)** 설명을 단순화하기 위해 최적화는 어떠한 가산 상수의 영향을 받지 않으므로, 다음에서 우리는 표기법을 남용하고 위에 언급된 모델링 가정 하에서 오히려 $\tilde{I}(w;D):=-\frac{1}{2}\sum_{i=1}^{\dim(w)}\log{}\alpha{}$를 정의한다. 네클류도프 외 연구진(2017)은 또한 적절한 log-uniform prior를 사용하여 임의의 상수를 처리하는 원칙적인 방법을 제안한다.

> Note that computing and optimizing this upper-bound to the information in the weights is relatively simple and efficient using the reparametrization trick of Kingma et al. (2015).
>> 가중치의 정보에 대한 이 상한을 계산하고 최적화하는 것은 Kingma et al. (2015)의 리파라메트리제이션 트릭을 사용하여 비교적 간단하고 효율적이다.

#### $\mathbf{4.3\;Flat\;minima\;have\;low\;information}$

> Thus far we have suggested that adding the explicit information regularizer $I(w;D)$ prevents the network from memorizing the dataset and thus avoid overfitting, which we also confirm empirically in Section 7. However, real networks are not commonly trained with this regularizer, thus seemingly undermining the theory. However, even when not explicitly present, the term $I(w;D)$ is implicit in the use of SGD. In particular, Chaudhari and Soatto (2018) show that, under certain conditions, SGD introduces an entropic bias of a very similar form to the information in the weights described thus far, where the amount of information can be controlled by the learning rate and the size of mini-batches.
>> 지금까지 명시적 정보 정규화기 $I(w;D)$를 추가하면 네트워크가 데이터 세트를 기억하지 못하므로 과적합을 방지할 수 있다고 제안했으며, 이는 섹션 7에서도 경험적으로 확인하였다. 그러나, 실제 네트워크는 일반적으로 이 정규화기로 훈련되지 않으므로, 겉보기에는 이론을 훼손한다. 그러나 명시적으로 존재하지 않는 경우에도 $I(w;D)$라는 용어는 SGD 사용에 암시적이다. 특히 Chaudhari와 Soatto(2018)는 특정 조건에서 SGD가 지금까지 설명한 가중치의 정보와 매우 유사한 형태의 엔트로피 편향을 도입한다는 것을 보여주는데, 여기서 학습률과 미니 배치의 크기에 의해 정보의 양을 제어할 수 있다.

> Additional indirect empirical evidence is provided by the fact that some variants of SGD (Chaudhari et al., 2017) bias the optimization toward “flat minima”, that are local minima whose Hessian has mostly small eigenvalues. These minima can be interpreted exactly as having low information $I(w;D)$, as suggested early on by Hochreiter and Schmidhuber (1997): Intuitively, since the loss landscape is locally flat, the weights may be stored at lower precision without incurring in excessive inference error. As a consequence of previous claims, we can then see flat minima as having better generalization properties and, as we will see in Section 5, the associated representation of the data is more insensitive to nuisances and more disentangled. For completeness, here we derive a more precise relationship between flatness (measured by the nuclear norm of the loss Hessian), and the information content based on our model.
>> 추가적인 간접 경험적 증거는 SGD(Chaudhari et al., 2017)의 일부 변형들이 헤시안 고윳값이 대부분 작은 국소 미니마인 "평탄 미니마"에 대한 최적화를 편향시킨다는 사실에 의해 제공된다. 이러한 최소값은 호크라이터와 슈미트후버(1997)가 초기에 제안한 바와 같이 낮은 정보 $I(w;D)$를 갖는 것으로 정확하게 해석될 수 있다. 직관적으로, 손실 경관은 국소적으로 평평하기 때문에 과도한 추론 오류 없이 가중치를 낮은 정밀도로 저장할 수 있다. 이전 주장의 결과로, 우리는 플랫 미니마가 더 나은 일반화 속성을 가진 것으로 볼 수 있으며, 섹션 5에서 보게 될 것처럼, 데이터의 관련 표현은 소음에 더 둔감하고 더 분리된다. 완전성을 위해, 여기서는 평탄성(손실 헤시안 핵 규범으로 측정됨)과 모델을 기반으로 한 정보 콘텐츠 사이의 보다 정확한 관계를 도출한다.

> **Proposition 4.3 (Flat minima have low information, Appendix C.5)** Let $\hat{w}$ be a local minimum of the cross-entropy loss $H_{p},q(y\vert{}x,w)$, and let H be the Hessian at that point. Then, for the optimal choice of the posterior $w\vert{}D=\epsilon{}\odot{}\hat{w}$ centered at wˆ that optimizes the IB Lagrangian, we have
>> **제안 4.3(플랫 미니마는 정보가 낮음, 부록 C.5)** $\hat{w}$를 교차 엔트로피 손실 $H_{p},q(y\vert{}x,w)$의 로컬 최소값으로 하고, 그 시점에서 H를 헤센인으로 한다. 그렇다면, IB 라그랑지안을 최적화하는 w that에 중심을 둔 후부 $w\vert{}D=\epsilon{}\odot{}\hat{w}$의 최적 선택을 위해, 우리는 다음과 같다.

$$I(w;D)\leq{}\tilde{I}(w;D)\leq{}\frac{1}{2}[\log{}\vert{}\vert{}\hat{w}\vert{}\vert{}_{2}^{2}+\log{}\vert{}\vert{}H\vert{}\vert{}_{*}-K\log{}(K^{2}\beta{}/2)]$$

> where $K=dim(w)$ and $k·k∗$ denotes the nuclear norm.
>> 여기서 $K=dim(w)$ 및 $k·k*$는 nuclear norm을 나타낸다.

> Notice that a converse inequality, that is, low information implies flatness, needs not hold, so there is no contradiction with the results of Dinh et al. (2017). Also note that for \tilde{I}(w;D) to be invariant to reparametrization one has to consider the constant C, which we have ignored (Remark 4.2). The connection between flatness and overfitting has also been studied by Neyshabur et al. (2017), including the effect of the number of parameters in the model.
>> 역 불평등, 즉 낮은 정보는 평탄성을 암시하므로 유지할 필요가 없으므로 Dinh 외 연구진(2017)의 결과와 모순이 없다. 또한 \tilde{의 경우I}(w;D)가 매개 변수화에 불변하기 위해서는 우리가 무시한 상수 C를 고려해야 한다(주 4.2). 평탄도와 과적합 사이의 연관성은 모델의 매개 변수 수의 영향을 포함하여 Neyshabur 외 연구진(2017)에 의해 연구되었다.

> In the next section, we prove one of our main results, that networks with low information in the weights realize invariant and disentangled representations. Therefore, invariance and disentanglement emerge naturally when training a network with implicit (SGD) or explicit (IB Lagrangian) regularization, and are related to flat minima.
>> 다음 섹션에서는 가중치에서 낮은 정보를 가진 네트워크가 불변하고 얽혀 있지 않은 표현을 실현한다는 주요 결과 중 하나를 증명한다. 따라서 암묵적(SGD) 또는 명시적(IB 라그랑지안) 정규화로 네트워크를 훈련할 때 불변성과 분리가 자연스럽게 나타나며 플랫 미니마와 관련이 있다.

### $\mathbf{5.\;Duality\;of\;the\;Bottleneck}$

> The following proposition gives the fundamental link in our model between information in the weights, and hence flatness of the local minima, minimality of the representation, and disentanglement.
>> 다음 명제는 가중치의 정보, 따라서 국소 최소값의 평탄성, 표현의 최소성 및 분리 사이의 우리 모델에서 근본적인 연결을 제공한다.

> **Proposition 5.1 (Appendix C.6)** Let $z=W_{x}$, and assume as before $W=\epsilon{}\hat{W}$, with $\epsilon_{i,j}\hat{W}\log{}N(−\alpha{}/2,\alpha{})$. Further assume that the marginals of $p(z)$ and $p(z\vert{}x)$ are both approximately Gaussian (which is reasonable for large dim(x) by the Central Limit Theorem). Then,
>> **제안 5.1(부록 C.6)*** $z=W_{x}$를 $\epsilon_{i,j}\hat{W}\log{}N(−\alpha{}/2,\alpha{})$로 하여 $W=\epsilon{}\hat{W}$ 이전과 같이 가정한다. 또한 $p(z)$와 $p(z\vert{}x)$의 한계는 모두 대략 가우스(중앙 한계 정리에 의해 큰 dim(x)에 대해 합리적)라고 가정한다. 그리고나서,

$$I(z;x)+TC(z)=-\frac{1}{2}\sum_{i=1}^{\dim{}(z)}E_{x}\log{}\frac{\tilde{\alpha{}}\hat{W}^{2}\cdot{}x^{2}}{\hat{W}\cdot{}Cov{}(x)\hat{W}+\tilde{\alpha}\hat{W}^{2}\cdot{}E(x^{2})},$$

> where $W_{i}$ denotes the i-th row of the matrix $W$, and $\alpha{}\tilde{i}$ is the noise variance $\alpha{}\tilde{i}=\exp(\alpha{})−1$. In particular, $I(z;x)+TC(z)$ is a monotone decreasing function of the weight variances $\alpha{}$.
>> 여기서 $W_{i}$는 행렬 $W$의 i번째 행을 나타내며, $\alpha{}\tilde{i}$는 노이즈 분산 $\alpha{}\tilde{i}=\exp(\alpha{})-1$이다. 특히 $I(z;x)+TC(z)$는 가중치 분산 $\alpha{}$의 단조 감소 함수이다.

> The above identity is difficult to apply in practice, but with some additional hypotheses, we can derive a cleaner uniform tight bound on $I(z;x)+TC(z)$.
>> 위의 동일성은 실제로 적용하기 어렵지만, 몇 가지 추가 가설을 통해 $I(z;x)+에 대해 더 깨끗한 균일한 엄격한 경계를 도출할 수 있다.TC(z)$.

> **Proposition 5.2 (Uniform bound for one layer, Appendix C.7)** Let $z=W_{x}$, where $W=\epsilon{}\odot{}\vert{}$ , where $\epsilon_{i,j}\sim{}\log{}N(−\alpha{}/2, \alpha{})$; assume that the components of $x$ are uncorrelated, and that their kurtosis is uniformly bounded.5 Then, there is a strictly increasing function $g(\alpha{})$ s.t. we have the uniform bound
>> **제안 5.2(단일 레이어에 대한 균일한 경계, 부록 C.7)** $z=W_{x}$, 여기서 $W=\epsilon{}\odot{}\vert{}$, 여기서 $\epsilon_{i,j}\sim{}\log{}N(−\alpha{}/2, \alpha{})$;는 $x$의 구성요소가 상관 관계가 없으며 첨도가 균일하게 경계된다고 가정한다.5 그렇다면, 엄격하게 증가하는 함수 $g(\alpha{})$ s.t.가 있다. 우리는 균일한 경계를 갖는다.

$$g(\alpha{}\leq{}\frac{I(x;z)+TC(z)}{\dim{z}}\leq{}g(\alpha)+c,$$

> where $c=O(1/dim(x))\leq{}1,g(\alpha{})=−\log{}(1-e −\alpha{})/2$ and $\alpha{}$ is related to $\tilde{I}(w;D)$ by $\alpha{}=\exp{−I(W; D)/dim(W)}$. In particular, $I(x;z)+TC(z)$ is tightly bounded by $\tilde{I}(W;D)$ and increases strictly with it.
>> 여기서 $c=O(1/dim(x))\leq{}1,g(\alpha{})=−\log{}(1-e −\alpha{})/2$와 $\alpha{}$는 $\alpha{}=\exp{−I(W; D)/dim(W)}$에 의해 $\tilde{I}(w;D)$와 관련이 있다. 특히, $I(x;z)+TC(z)$는 $\tilde{I}(W;D)$에 의해 단단히 경계되어 있고, $\tilde{I}(W;D)$에 따라 엄격하게 증가한다.

> The above theorems tells us that whenever we decrease the information in the weights, either by explicit regularization, or by implicit regularization (e.g., using SGD), we automatically improve the minimality, and hence, by Proposition 3.1, the invariance, and the disentanglement of the learner representation. In particular, we obtain as a corollary that SGD is biased toward learning invariant and disentangled representations of the data. Using the Markov property of the layers, we can easily extend this bound to multiple layers:
>> 위의 정리는 명시적 정규화 또는 암묵적 정규화(예: SGD 사용)를 통해 가중치에서 정보를 줄일 때마다 자동으로 최소성을 향상시켜 제안 3.1에 의해 불변성, 학습자 표현의 분리를 개선한다는 것을 알려준다. 특히, 우리는 SGD가 데이터의 불변하고 얽혀 있지 않은 표현을 학습하는 데 편향되어 있다는 것을 귀납적으로 얻는다. 레이어의 마르코프 속성을 사용하면 이 경계를 여러 레이어로 쉽게 확장할 수 있다.

> **Corollary 5.3 (Multi-layer case, Appendix C.8)** Let $W^{k}$ for k=1, ..., L be weight matrices, with $W^{k}=\epsilon{}^{k}\odot\hat{W}^{k}$ and $\epsilon{}^{k}=\log{}N(−\alpha{}^{k}/2,\alpha{}^{k})$, and let $z_{i+1}=\phi(W^{k}z_{k})$, where $z_{0}=x$ and $\phi$ is any nonlinearity. Then,
>> **상관 5.3(다층 사례, 부록 C.8)** $W^{k}=\epsilon{}^{k}\odot\hat{에서 k=1, ..., L에 대한 $W^{k}$를 가중치 행렬로 한다.W}^{k}$ 및 $\epsilon{}^{k}=\log{}N(-\alpha{}^{k}/2,\alpha{}^{k})$를 두고, 여기서 $z_{i+1}=\phi(W^{k}z_{k}$ 및 $\phi$는 비선형이다. 그리고나서,

$$I(z_{L};x)\leq{}\underset{K<L}{\min{}}(dim(z_{k})[g(\alpha^{k})+1])$$

> where $\alpha{}^{k}=\exp(−I(W^{k};D)/dim(W^{k})$.
>> 여기서 $\alpha{}^{k}=\exp(-I(W^{k};D)/dim(W^{k})$.

> **Remark 5.4 (Tightness)** While the bound in Proposition 5.2 is tight, the bound in the multilayer case needs not be. This is to be expected: Reducing the information in the weights creates a bottleneck, but we do not know how much information about $x$ will actually go through this bottleneck. Often, the final layers will let most of the information through, while initial layers will drop the most.
>> ***주 5.4(긴축성)** 발의안 5.2의 경계가 엄격한 반면, 다층 케이스의 경계가 그럴 필요는 없다. 이는 예상된 바와 같습니다. 가중치에서 정보를 줄이면 병목 현상이 발생하지만 $x$에 대한 정보가 실제로 이 병목 현상을 얼마나 통과할지는 알 수 없다. 종종, 최종 계층은 대부분의 정보를 통과시키는 반면, 초기 계층은 가장 많이 떨어집니다.

> **Remark 5.5 (Training-test transfer)** We note that we did not make any (explicit) assumption about the test set having the same distribution of the training set. Instead, we make the less restrictive assumption of sufficiency: If the test distribution is entirely different from the training one – one may not be able to achieve sufficiency. This prompts interesting questions about measuring the distance between tasks (as opposed to just distance between distributions), which will be studied in future work.
>> ** 비고 5.5 (교육-시험 이전)** 우리는 교육 세트의 분포가 동일한 시험 세트에 대해 어떠한 (명백한) 가정도 하지 않았다는 점에 주목한다. 대신, 우리는 충분성에 대한 덜 제한적인 가정을 한다. 시험 분포가 훈련 분포와 완전히 다를 경우, 충분한 결과를 얻지 못할 수 있다. 이것은 (분포 사이의 거리뿐만 아니라) 작업 간의 거리 측정에 대한 흥미로운 질문을 유발하며, 이는 향후 연구에서 연구될 것이다.

### $\mathbf{6.\;Connection\;with\;PAC-Bayes\;bounds}$

> In this section we show that using a PAC-Bayes bound, we arrive at the same regularized loss function eq. (5) we obtained using the Information Bottleneck, without the need of any approximation. By Theorem 2 of McAllester (2013), we have that for any fixed $λ>1/2$, prior $p(w)$, and any weight distribution $q(w\vert{}D)$, the test error $L^{test}(q(w\vert{}D))$ that the network commits using the weight distribution $q(w\vert{}D)$ is upper-bounded in expectation by
>> 이 섹션에서는 PAC-Bayes 경계를 사용하여 근사치 없이 정보 병목 현상을 사용하여 얻은 정규화된 손실 함수 등(5)에 도달한다는 것을 보여준다. McAllester (2013)의 정리 2에 의해, 우리는 임의의 고정된 $λ>1/2$, 이전 $p(w)$, 그리고 임의의 가중치 분포 $q(w\vert{}D)$에 대해, 네트워크가 가중치 분포 $q(w\vert{}D)$를 사용하여 저지르는 시험 오차 $L^{test}(q(w\vert{}D))$는 다음과 같이 기대하여 상한이다.

$$E_{D}[L^{test}(q(w\vert{}D))]<\frac{1}{N(1-\frac{1}{2\lambda})}(H_{p,q}(y\vert{}x,w)+\lambda{}L_{\max}E_{D}[KL(q(w\vert{}D)\vert{}\vert{}p(w))]),$$

> where $L_{max}$ is the maximum per-sample loss function, which for a classification problem we can assume to be upper-bounded, for example by clipping the cross-entropy loss at chance level. Notice that right hand side coincides, modulo a multiplicative constant, with eq. (4) that we derived as an approximation of the IB Lagrangian for the weights (eq. (3)).
>> 여기서 $L_{max}$는 샘플당 최대 손실 함수이며, 분류 문제의 경우 예를 들어 기회 수준에서 교차 엔트로피 손실을 클리핑하여 상한이라고 가정할 수 있다. 오른쪽 변은 가중치에 대한 IB 라그랑지안의 근사치로 도출한 (4)와 곱셈 상수가 일치한다는 점에 유의한다(예: (3)).

> Now, recall that since we have
>> 다음을 참고하자

$$E_{D}[KL(q(w\vert{}D)\vert{}\vert{}q(w))]=E_{D}[KL(q(w\vert{}D)\vert{}\vert{}p(w))]-KL(q(w)\vert{}\vert{}p(w))\leq{}E_{D}[KL(q(w\vert{}D)\vert{}\vert{}p(w))],$$

> the sharpest PAC-Bayes upper-bound to the test error is obtained when $p(w)=q(w)$, in which case eq. (7) reduces (modulo a multiplicative constant) to the IB Lagrangian of the weights. That is, the IB Lagrangian for the weights can be considered as a special case of PAC-Bayes giving the sharpest bound.
>> 시험 오차에 대한 가장 날카로운 PAC-Bayes 상한은 $p(w)=q(w)$일 때 얻어지며, 이 경우 (7)는 가중치의 IB 라그랑지안까지 감소합니다. 즉, 가중치에 대한 IB 라그랑지안은 가장 날카로운 경계를 제공하는 PAC-Bayes의 특수한 경우로 간주될 수 있습니다.

> Unfortunately, as we noticed in Section 4, the joint marginal $q(w)$ of the weights is not tractable. To circumvent the problem, we can instead consider that the sharpest PAC-Bayes upper-bound that can be obtained using a tractable factorized prior $p(w)$, which is obtained exactly when $p(w)=\tilde{q}(w)=\prod_{i}q(w_{i})$ is the product of the marginals, leading again to our practical loss eq. (5).
>> 불행하게도, 우리가 섹션 4에서 주목한 바와 같이, 가중치의 공동 한계 $q(w)$는 취급할 수 없습니다. 문제를 피하기 위해 대신 $p(w)=\tilde{q}(w)=\tilde{i}q(w_{i})$일 때 정확히 얻을 수 있는 다루기 쉬운 인수 분해 이전 $p(w)$를 사용하여 얻을 수 있는 가장 날카로운 PAC-Bayes 상한선이 실제 손실 eq(5)로 이어진다고 고려할 수 있습니다.

> On a last note, recall that under our modeling assumptions the marginal $\tilde{q}(w)$ is assumed to be an improper log-uniform distribution. While this has the advantage of being a noninformative prior that closely matches the real marginal of the weights of the network, it also has the disadvantage that it is only defined modulo an additive constant, therefore making the bound on the test error vacuous under our model.
>> 마지막으로, 모델링 가정 하에서 한계 $\tilde{q}(w)$는 부적절한 로그 균일 분포로 가정된다는 것을 기억하십시오. 이는 네트워크 가중치의 실제 한계와 밀접하게 일치하는 비정보적 선행이라는 장점이 있지만, 모듈로 정의되는 추가 상수만 있으므로 우리의 모델에서 테스트 오류에 대한 경계를 모호하게 만든다는 단점도 있습니다.

> The PAC-Bayes bounds has also been used by Dziugaite and Roy (2017) to study the generalization property of deep neural networks and their connection with the optimization algorithm. They use a Gaussian prior and posterior, leading to a non-vacuous generalization bound.
>> PAC-Bayes 경계는 또한 Dziugaite와 Roy(2017)에 의해 심층 신경망의 일반화 특성과 최적화 알고리듬과의 연결을 연구하는 데 사용되었습니다. 이들은 가우시안 전후를 사용하여 비진공 일반화 경계로 이어집니다.

### $\mathbf{7.\;Empirical\;validation}$

#### $\mathbf{7.1\;Transition\;from\;overfitting\;to\;underfitting}$

> As pointed out by Zhang et al. (2017), when a standard convolutional neural network (CNN) is trained on CIFAR-10 to fit random labels, the network is able to (over)fit them perfectly. This is easily explained in our framework: It means that the network is complex enough to memorize all the labels but, as we show here, it has to pay a steep price in terms of information complexity of the weights (Figure 2) in order to do so. On the other hand, when the information in the weights is bounded using and information regularizer, overfitting is prevented in a theoretically predictable way.
>> Zhang 등(2017)이 지적한 바와 같이, 표준 컨볼루션 신경망(CNN)이 CIFAR-10에 대해 무작위 레이블을 적합하도록 훈련될 때, 네트워크는 이들을 완벽하게 적합시킬 수 있습니다. 이는 우리의 프레임워크에서 쉽게 설명됩니다. 즉, 네트워크는 모든 레이블을 기억할 수 있을 정도로 복잡하지만, 여기서 보여 주듯이, 그렇게 하기 위해서는 가중치의 정보 복잡성 측면에서 엄청난 대가를 치러야 합니다(그림 2). 한편, 가중치의 정보가 와 정보 정규화기를 사용하여 제한되면 이론적으로 예측 가능한 방법으로 과적합이 방지됩니다.

> In particular, in the case of completely random labels, we have $I(y;w\vert{}x,\theta)=I(y;w)\leq{}I(w;D)$, where the first equality holds since $y$ is by construction random, and therefore independent of $x$ and $\theta$. In this case, the inequality used to derive eq. (3) is an equality, and the IBL is an optimal regularizer, and, regardless of the dataset size $N$, for $\beta{}>1$ it should completely prevent memorization, while for $\beta{}<1$ overfitting is possible. To see this, notice that since the labels are random, to decrease the classification error by $\log{}\vert{}Y\vert{}$, where $\vert{}Y\vert{}$ is the number of possible classes, we need to memorize a new label. But to do so, we need to store more information in the weights of the network, therefore increasing the second term $I(w;D)$ by a corresponding quantity. This trade-off is always favorable when $\beta{}<1$, but it is not when $\beta{}>1$. Therefore, the theoretically the optimal solution to eq. (1) is to memorize all the labels in the first case, and not memorize anything in the latter.
>> 특히, 완전 랜덤 레이블의 경우, $y$ 이후 첫 번째 등식이 시공 랜덤으로 유지되는 $I(y;w\vert{}x,\theta)=I(y;w)\leq{}I(w;D)$가 있으며, 따라서 $x$ 및 $\theta$와 독립적입니다. 이 경우, 등식 (3)을 도출하는 데 사용되는 부등식은 동등하며, IBL은 최적의 정규화기이며, 데이터 세트 크기 $N$에 관계없이 $\beta{}>1$의 경우 암기를 완전히 방지해야 하며, $\beta{}<1$ 오버피팅의 경우 가능하다. 이를 보려면 레이블이 랜덤이므로 분류 오류를 $\log{}\vert{}Y\vert{}$($\vert{}Y\vert{}$는 가능한 클래스 수)만큼 줄이려면 새 레이블을 기억해야 합니다. 그러나 그러기 위해서는 네트워크의 가중치에 더 많은 정보를 저장해야 하므로, 두 번째 항 $I(w;D)$를 그에 상응하는 양만큼 증가시켜야 합니다. 이 균형은 $\beta{}<1$일 때 항상 유리하지만 $\beta{}>1$일 때는 그렇지 않습니다. 따라서 이론적으로 (1)에 대한 최적의 해결책은 첫 번째 경우에는 모든 레이블을 기억하고, 후자의 경우에는 아무것도 기억하지 않는 것입니다.

> As discussed, for real neural networks we cannot directly minimize eq. (1), and we need to use a computable upper bound to $I(w;D)$ instead (Section 4.2). Even so, the empirical ehavior of the network, shown in Figure 1, closely follows this prediction, and for various sizes of the dataset clearly shows a phase transition between overfitting and underfitting near the critical value $\beta{}=1$. Notice instead that for real labels the situation is different:
>> 논의된 바와 같이, 실제 신경망의 경우 우리는 등(1)을 직접 최소화할 수 없으며 대신 $I(w;D)$에 대한 계산 가능한 상한을 사용해야 합니다(섹션 4.2). 그럼에도 불구하고, 그림 1에 표시된 네트워크의 경험적 동작은 이러한 예측을 밀접하게 따르고 있으며, 다양한 크기의 데이터 세트에 대해 임계 값 $\beta{}=1$에 가까운 과적합과 과소적합 사이의 위상 전환을 분명히 보여줍니다. 대신 실제 라벨의 경우 상황이 다릅니다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)Emergence-of-Invariance-and-Disentanglement-in-Deep-Representations/Figure-2.JPG)

> Figure 2: (Left) Plot of the training error on CIFAR-10 with random labels as a function of the parameter $\beta{}$ for different models (see the appendix for details). As expected, all models show a sharp phase transition from complete overfitting to underfitting before the critical value $\beta{}=1$. (Right) We measure the quantity of information in the weights necessary to overfit as we vary the percentage of corrupted labels under the same settings of Figure 1. To fit increasingly random labels, the network needs to memorize more information in the weights; the increase needed to fit entirely random labels is about the same magnitude as the size of a label (2.30 nats/sample).
>> 그림 2: (왼쪽) 다양한 모델에 대한 매개 변수 $\beta{}$의 함수로서 무작위 레이블을 사용한 CIFAR-10의 교육 오류 그림입니다(자세한 내용은 부록 참조). 예상대로, 모든 모델은 임계 값 $\sigma{}=1$ 이전에 완전한 과적합에서 과적합으로 급격한 위상 전환을 보여줍니다. (오른쪽) 그림 1의 동일한 설정에서 손상된 레이블의 비율을 변경함에 따라 과적합에 필요한 가중치에서 정보의 양을 측정합니다. 점점 더 무작위적인 레이블을 적합시키려면 네트워크는 가중치에 있는 더 많은 정보를 기억해야 합니다. 완전히 무작위 레이블을 적합시키는 데 필요한 증가량은 레이블 크기(2.30 nats/샘플)와 거의 같습니다.

> The model is still able to overfit when $\beta{}<1$, but importantly there is a large interval of $\beta{}>1$ where the model can fit the data without overfitting to it. Indeed, as soon as $\beta{}N∝I(w;D)$ is larger than the constant $H(\theta)$, the model trained on real data fits real labels without excessive overfitting (Figure 1). 
>> 모델은 $\beta{}<1$일 때 여전히 오버핏할 수 있지만, 중요한 것은 모델이 데이터를 오버핏하지 않고 적합시킬 수 있는 $\beta{}>1$의 큰 간격이 있다는 것입니다. 실제로 $\beta{}N∝I(w;D)$가 상수 $H(\theta)$보다 크자마자 실제 데이터에 대해 훈련된 모델은 과도한 과적합 없이 실제 레이블을 적합시킵니다(그림 1).

> Notice that, based on this reasoning, we expect the presence of a phase transition between an overfitting and an underfitting regime at the critical value $\beta{}=1$ to be largely independent on the network architecture: To verify this, we train different architectures on a subset of 10000 samples from CIFAR-10 with random labels. As we can see on the left plot of Figure 2, even very different architectures show a phase transition at a similar value of $\beta{}$. We also notice that in the experiment ResNets has a sharp transition close to the critical $\beta{}$.
>> 이 추론을 기반으로 임계 값 $\beta{}=1$에서 과적합과 과소적합 체제 사이의 위상 전환이 네트워크 아키텍처에 크게 독립적일 것으로 예상한다는 점에 주목하십시오. 이를 검증하기 위해 무작위 레이블을 사용하여 CIFAR-10의 10,000개 샘플 서브셋에 대해 서로 다른 아키텍처를 교육합니다. 그림 2의 왼쪽 그림에서 볼 수 있듯이, 매우 다른 아키텍처도 유사한 값 $\beta{}$에서 위상 전환을 보여줍니다. 또한 실험에서 ResNets는 중요한 $\beta{}$에 가까운 급격한 전환을 가지고 있다는 것을 알 수 있습니다.

> In the right plot of Figure 2 we measure the quantity information in the weights for different levels of corruption of the labels. To do this, we fix $\beta{}<1$ so that the network is able to overfit, and for various level of corruption we train until convergence, and then compute $I(w;D)$ for the trained model. As expected, increasing the randomness of the labels increases the quantity of information we need to fit the dataset. For completely random labels, $I(w;D)$ increases by$\sim{}3$ nats/sample, which the same order of magnitude as the quantity required to memorize a 10-class labels (2.30 nats/sample), as shown in Figure 2.
>> 그림 2의 오른쪽 그림에서 우리는 라벨의 다양한 손상 수준에 대한 가중치의 수량 정보를 측정합니다. 이를 위해 $\beta{}<1$을 수정하여 네트워크가 오버핏될 수 있도록 하고, 수렴할 때까지 다양한 수준의 손상에 대해 훈련한 다음 훈련된 모델에 대해 $I(w;D)$를 계산한다. 예상대로 레이블의 랜덤성이 증가하면 데이터 세트에 맞는 데 필요한 정보의 양이 증가합니다. 완전히 무작위 레이블의 경우, 그림 2와 같이 $I(w;D)$는 $\sim{}3$ nats/sample만큼 증가하며, 이는 10개 클래스 레이블(2.30 nats/sample)을 기억하는 데 필요한 양과 동일한 크기 순서입니다.

#### $\mathbf{7.2\;Bias-variance\;trade-off}$

> The Bias-Variance trade-off is sometimes informally stated as saying that low-complexity models tend to underfit the data, while excessively complex models may instead overfit, so that one should select an adequate intermediate complexity. This is apparently at odds with the common practice in Deep Learning, where increasing the depth or the number of weights of the network, and hence increasing the “complexity” of the model measured by the number of parameters, does not seem to induce overfitting. Consequently, a number of alternative measures of complexity have been proposed that capture the intuitive biasvariance trade-off curve, such as different norms of the weights (Neyshabur et al., 2015).
>> Bias-Variance 트레이드오프는 때때로 저복잡도 모델이 데이터를 과소 적합시키는 경향이 있는 반면, 지나치게 복잡한 모델은 대신 과적합할 수 있으므로 적절한 중간 복잡성을 선택해야 한다고 비공식적으로 언급됩니다. 이는 네트워크의 깊이 또는 가중치 수를 증가시켜 매개 변수의 수로 측정된 모델의 "복잡도"를 증가시키는 것이 과적합을 유도하지 않는 것처럼 보이는 딥 러닝의 일반적인 관행과 명백히 상충됩니다. 결과적으로, 가중치의 다른 규범과 같은 직관적인 편향 분산 트레이드오프 곡선을 포착하는 여러 가지 복잡성 대안 측정이 제안되었습니다(Neyshabur et al., 2015).

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)Emergence-of-Invariance-and-Disentanglement-in-Deep-Representations/Figure-3.JPG)

> Figure 3: Plots of the test error obtained training the All-CNN architecture on CIFAR-10 (no data augmentation). (Left) Test error as we increase the number of weights in the network using weight decay but without any additional explicit regularization. Notice that increasing the number of weights the generalization error plateaus rather than increasing. (Right) Changing the value of $\beta{}$, which controls the amount of information in the weights, we obtain the characteristic curve of the bias-variance trade-off. This suggests that the quantity of information in the weights correlates well with generalization. 
>> 그림 3: CIFAR-10에 대한 All-CNN 아키텍처 교육에서 얻은 테스트 오류 그림(데이터 확대 없음)입니다. (왼쪽) 추가 명시적 정규화 없이 가중치 감소를 사용하여 네트워크에서 가중치 수를 늘릴 때 발생하는 테스트 오류입니다. 가중치가 증가하지 않고 일반화 오류 평원의 가중치 수를 늘리면 가중치에서 정보량을 제어하는 $\beta{}$의 값을 변경하면 편향-분산 트레이드오프의 특성 곡선을 얻을 수 있습니다. 이는 가중치의 정보량이 일반화와 잘 상관된다는 것을 나타냅니다.

> From the discussion above, we have seen that the quantity of information in the weights, or alternatively its computable upperbound $\tilde{I}(w;D)$, also provides a natural choice to measure model complexity in relation to overfitting. In particular, we have already seen that models need to store increasingly more information to fit increasingly random labels (Figure 2). In Figure 3 we show that by controlling $\tilde{I}(w;D)$, which can be done easily by modulating $\beta{}$, we recover the right trend for the bias-variance tradeoff, whereas models with too little information tend to underfit, while models memorizing too much information tend to overfit.
>> 위에서 논의한 결과, 가중치 또는 계산 가능한 상한 $\tilde{I}(w;D)$에 있는 정보의 양이 과적합과 관련하여 모델 복잡성을 측정할 수 있는 자연스러운 선택을 제공한다는 것을 알게 되었습니다. 특히, 우리는 모델이 점점 더 무작위적인 레이블을 맞추기 위해 점점 더 많은 정보를 저장해야 한다는 것을 이미 확인했습니다(그림 2). 그림 3에서 우리는 $\beta{}$를 변조하여 쉽게 수행할 수 있는 $\tilde{I}(w;D)$를 제어함으로써 편향-분산 트레이드오프의 올바른 추세를 회복하는 반면, 정보가 너무 적은 모델은 언더핏하는 경향이 있는 반면, 너무 많은 정보를 기억하는 모델은 오버핏하는 경향이 있음을 보여준다.

#### $\mathbf{7.3\;Nuisance\;invariance}$

> Corollary 5.3 shows that by decreasing the information in the weights $I(w;D)$, which can be done for example using eq. (3), the learned representation will be increasingly minimal, and therefore insensitive to nuisance factors n, as measured by $I(z;n)$. Here, we adapt a technique from the GAN literature Sønderby et al. (2017) that allows us to explicitly measure $I(z;n)$ and validate this effect, provided we can sample from the nuisance distribution $p(n)$ and from $p(x\vert{}n)$; that is, if given a nuisance $n$ we can generate data $x$ affected by that nuisance. Recall that by definition we have
>> 상관 관계 5.3은 예를 들어 등식 (3)을 사용하여 수행할 수 있는 가중치 $I(w;D)$의 정보를 줄임으로써 학습된 표현이 점점 더 최소가 되고, 따라서 $I(z;n)$로 측정된 성가신 요소 n에 민감하지 않게 된다는 것을 보여줍니다. 여기서, 우리는 귀찮은 분포 $p(n)$와 $p(x\vert{}n)$에서 샘플링할 수 있다면, $I(z;n)$를 명시적으로 측정하고 이 효과를 검증할 수 있는 GAN 문헌 Sunderby 등(2017)의 기술을 채택한다. 즉, 귀찮은 $n$이 주어지면 해당 방해에 영향을 받는 데이터 $x$를 생성할 수 있다. 우리가 정의상 가지고 있는 것을 기억하십시오.

$$I(z;n)=E_{n\sim{}p(n)}KL(p(z\vert{}n)\vert{}\vert{}P(z))=E_{n\sim{}p(n)}E_{z\sim{}p(z\vert{}n)}\log{}[p(z\vert{}n)/p(z)].$$

> To approximate the expectations via sampling we need a way to approximate the likelihood ratio $\log{}p(z\vert{}n)/p(z)$. This can be done as follows: Let $D(z;n)$ be a binary discriminator that given the representation $z$ and the nuisance $n$ tries to decide whether $z$ is sampled from the posterior distribution $p(z\vert{}n)$ or from the prior $p(z)$. Since by hypothesis we can generate samples from both distributions, we can generate data to train this discriminator. Intuitively, if the discriminator is not able to classify, it means that $z$ is insensitive to changes of n. Precisely, since the optimal discriminator is
>> 샘플링을 통해 기대치를 근사하려면 우도비 $\log{}p(z\vert{}n)/p(z)$를 근사화하는 방법이 필요합니다. 이 작업은 다음과 같이 수행할 수 있습니다. $D(z;n)$가 표현 $z$가 주어지고 성가신 $n$은 $z$가 사후 분포 $p(z\vert{}n)$에서 샘플링되는지 이전 $p(z)$에서 샘플링되는지 여부를 결정하려고 하는 이진 판별기가 되도록 합니다. 가설을 통해 두 분포 모두에서 표본을 생성할 수 있으므로 이 판별자를 훈련시키기 위한 데이터를 생성할 수 있습니다. 직관적으로, 판별기가 분류할 수 없다면, $z$는 n의 변화에 민감하지 않다는 것을 의미합니다. 정확히, 최적 판별기는

$$D*(z;n)=\frac{p(z)}{p(z)+p(z\vert{}n)}$$

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-24-(GAN)Emergence-of-Invariance-and-Disentanglement-in-Deep-Representations/Figure-4.JPG)

> Figure 4: (Left) A few training samples generated adding nuisance clutter $n$ to the MNIST dataset. (Right) Reducing the information in the weights makes the representation $z$ learned by the digit classifier increasingly invariant to nuisances ($I(n;z)$ decreases), while sufficiency is retained ($I(z;y)=I(x;y)$ is constant). As expected, $I(z;n)$ is smaller but has a similar behavior to the theoretical bound in Theorem 5.3. 
>> 그림 4: (왼쪽) 생성된 몇 가지 훈련 샘플은 MNIST 데이터 세트에 성가신 잡동사니 $n$을 추가합니다. (오른쪽) 가중치에서 정보를 줄이면 자릿수 분류기에 의해 학습된 표현 $z$는 소음($I(n;z)$)에 대해 점점 더 불변하게 됩니다. 반면, 충분성은 유지됩니다($I(z;y)=I(x;y)$는 상수). 예상대로, $I(z;n)$는 더 작지만 정리 5.3의 이론적 한계와 유사한 동작을 합니다.

> if we assume that $D$ is close to the optimal discriminator $D∗$ , we have
>> 만약 $D$가 최적의 판별기 $D ,$에 가깝다고 가정하면, 우리는 다음과 같습니다.

$$I(z;n)=E_{n\sim{p(z)}}KL(p(z\vert{}n)\vert{}\vert{}p(z))=E_{n\sim{p(n)}}E_{z\sim{p(z\vert{}n)}}\log{}[p(z\vert{}n)/p(z)].$$

> therefore we can use D to estimate the log-likelihood ratio, and so also the mutual information $I(z;n)$. Notice however that this comes with no guarantees on the quality of the approximation.
>> 따라서 D를 사용하여 로그 우도비를 추정할 수 있으며, 상호 정보 $I(z;n)$도 추정할 수 있습니다. 그러나 이것은 근사치의 품질을 보장하지 않습니다.

> To test this algorithm, we add random occlusion nuisances to MNIST digits (Figure 4). In this case, the nuisance $n$ is the occlusion pattern, while the observed data $x$ is the occluded digit. For various values of $\beta{}$, we train a classifier on this data in order to learn a representation $z$, and, for each representation obtained this way, we train a discriminator as described above and we compute the resulting approximation of $I(z;n)$. The results in Figure 4 show that decreasing the information in the weights makes the representation increasingly more insensitive to n.
>> 이 알고리즘을 테스트하기 위해 MNIST 숫자에 랜덤 폐색 잡음을 추가합니다(그림 4). 이 경우 성가신 $n$은 폐색 패턴이고, 관찰된 데이터 $x$는 폐색된 숫자입니다. $\beta{}$의 다양한 값에 대해 표현 $z$를 학습하기 위해 이 데이터에 대한 분류기를 훈련하고, 이러한 방식으로 얻은 각 표현에 대해 위에서 설명한 대로 판별기를 훈련하고 $I(z;n)$의 결과 근사치를 계산한다. 그림 4의 결과는 가중치에서 정보를 줄이면 표현이 n에 대해 점점 더 둔감해진다는 것을 보여줍니다.

### $\mathbf{8.\;Discussion\;and\;conclusion}$

> In this work, we have presented bounds, some of which are tight, that connect the amount of information in the weights, the amount of information in the activations, the invariance property of the network, and the geometry of the residual loss. These results leverage the structure of deep networks, in particular the multiplicative action of the weights, and the Markov property of the layers. This leads to the surprising result that reducing information stored in the weights about the past (dataset) results in desirable properties of the learned internal representation of the test datum (future). 
>> 본 연구에서는 가중치의 정보량, 활성화의 정보량, 네트워크의 불변성 특성 및 잔여 손실의 형상을 연결하는 경계를 제시했습니다. 이러한 결과는 심층 네트워크의 구조, 특히 가중치의 곱셈 작용과 계층의 마르코프 속성을 활용합니다. 이는 과거(데이터 세트)에 대한 가중치에 저장된 정보를 줄이면 테스트 데이터(미래)의 학습된 내부 표현의 바람직한 특성이 나타난다는 놀라운 결과로 이어집니다.

> Our notion of representation is intrinsically stochastic. This simplifies the computation as well as the derivation of information-based relations. However, note that even if we start with a deterministic representation w, Proposition 4.3 gives us a way of converting it to a stochastic representation whose quality depends on the flatness of the minimum. Our theory uses, but does not depend on, the Information Bottleneck Principle, which dates back to over two decades ago, and can be re-derived in a different frameworks, for instance PAC-Bayes, which yield the same results and additional bounds on the test error. 
>> 우리의 표현 개념은 본질적으로 확률적입니다. 이를 통해 계산과 정보 기반 관계의 도출이 단순화됩니다. 그러나, 우리가 결정론적 표현 w로 시작한다고 해도, 제안 4.3은 그것을 최소의 평탄도에 따라 품질이 달라지는 확률적 표현으로 변환하는 방법을 제공한다는 것을 주목하십시오. 우리의 이론은 20년 전 이상으로 거슬러 올라가는 정보 병목 현상 원리를 사용하지만 의존하지 않습니다. 예를 들어 PAC-Bayes와 같은 다른 프레임워크에서 다시 파생될 수 있습니다. PAC-Bayes는 동일한 결과와 테스트 오류에 대한 추가 한계를 산출합니다.

> This work focuses on the inference and learning of optimal representations, that seek to get the most out of the data we have for a specific task. This does not guarantee a good outcome since, due to the Data Processing Inequality, the representation can be easier to use but ultimately no more informative than the data themselves. An orthogonal but equally interesting issue is how to get the most informative data possible, which is the subject of active learning, experiment design, and perceptual exploration. Our work does not address transfer learning, where a representation trained to be optimal for a task is instead used for a different task, which will be subject of future investigations.
>> 이 작업은 특정 작업에 대해 우리가 가진 데이터를 최대한 활용하고자 하는 최적의 표현의 추론과 학습에 중점을 둡니다. 데이터 처리 불평등 때문에 표현은 더 사용하기 쉬울 수 있지만 궁극적으로 데이터 자체보다 더 많은 정보를 제공할 수 없기 때문에 이것은 좋은 결과를 보장하지 않습니다. 직교적이지만 똑같이 흥미로운 문제는 능동적인 학습, 실험 설계 및 지각 탐구의 주제인 가능한 가장 유익한 데이터를 얻는 방법입니다. 우리의 작업은 작업에 대해 최적으로 훈련된 표현이 대신 다른 작업에 사용되는 전이 학습을 다루지 않으며, 이는 향후 연구의 대상이 될 것입니다.

### $\mathbf{Acknowledgments}$

> Supported by ONR N00014-17-1-2072, ARO W911NF-17-1-0304, AFOSR FA9550-15-1-0229 and FA8650-11-1-7156. We wish to thank our reviewers and David McAllester, Kevin Murphy, Alessandro Chiuso for the many insightful comments and suggestions.
>> ONR N00014-17-1-2072, ARO W911NF-17-1-0304, AFOSR FA9550-15-1-0229 및 FA8650-11-1-7156에서 지원됩니다. 많은 통찰력 있는 의견과 제안에 대해 데이비드 맥알레스터, 케빈 머피, 알레산드로 치우소 평론가 여러분께 감사드립니다.

---

#### $\mathbf{References}$

<a href="#footnote_1_2" name="footnote_1_1">[1]</a> Alessandro Achille and Stefano Soatto. Information dropout: Learning optimal representations through noisy computation. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), PP(99):1–1, 2018.

<a href="#footnote_2_2" name="footnote_2_1">[2]</a> Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. In Proceedings of the International Conference on Learning Representations (ICLR), 2017a.

<a href="#footnote_3_2" name="footnote_3_1">[3]</a> Alexander A. Alemi, Ben Poole, Ian Fischer, Joshua V. Dillon, Rif A. Saurous, and Kevin Murphy. Fixing a Broken ELBO. ArXiv e-prints, November 2017b.

<a href="#footnote_4_2" name="footnote_4_1">[4]</a> Fabio Anselmi, Lorenzo Rosasco, and Tomaso Poggio. On invariance and selectivity in representation learning. Information and Inference, 5(2):134–158, 2016.

<a href="#footnote_5_2" name="footnote_5_1">[5]</a> Zhaojun Bai, Gark Fahey, and Gene Golub. Some large-scale matrix computation problems. Journal of Computational and Applied Mathematics, 74(1-2):71–89, 1996.

<a href="#footnote_6_2" name="footnote_6_1">[6]</a> Yoshua Bengio. Learning deep architectures for ai. Foundations and trends in Machine Learning, 2(1):1–127, 2009.

<a href="#footnote_7_2" name="footnote_7_1">[7]</a> Sterling K. Berberian. Borel spaces, April 1988.

<a href="#footnote_8_2" name="footnote_8_1">[8]</a> Joan Bruna and St´ephane Mallat. Classification with scattering operators. In IEEE Conference on Computer Vision and Pattern Recognition, pages 1561–1566, 2011.

<a href="#footnote_9_2" name="footnote_9_1">[9]</a> Pratik Chaudhari and Stefano Soatto. Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks. Proc. of the International Conference on Learning Representations (ICLR), 2018.

<a href="#footnote_10_2" name="footnote_10_1">[10]</a> Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-sgd: Biasing gradient descent into wide valleys. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.

<a href="#footnote_11_2" name="footnote_11_1">[11]</a> Djork-Arn´e Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015.

<a href="#footnote_12_2" name="footnote_12_1">[12]</a> Thomas M Cover and Joy A Thomas. Elements of information theory. John Wiley & Sons, 2012.

<a href="#footnote_13_2" name="footnote_13_1">[13]</a> Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. Proceedings of the 34th International Conference on Machine Learning (ICML), 2017.

<a href="#footnote_14_2" name="footnote_14_1">[14]</a> Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. arXiv preprint arXiv:1703.11008, 2017.

<a href="#footnote_15_2" name="footnote_15_1">[15]</a> Jerome Friedman, Trevor Hastie, and Robert Tibshirani. The elements of statistical learning, volume 1. Springer series in statistics New York, 2001.

<a href="#footnote_16_2" name="footnote_16_1">[16]</a> Ulf Grenander. General Pattern Theory. Oxford University Press, 1993.

<a href="#footnote_17_2" name="footnote_17_1">[17]</a> Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

<a href="#footnote_18_2" name="footnote_18_1">[18]</a> Geoffrey E Hinton and Drew Van Camp. Keeping the neural networks simple by minimizing the description length of the weights. In Proceedings of the 6th annual conference on Computational learning theory, pages 5–13. ACM, 1993.

<a href="#footnote_19_2" name="footnote_19_1">[19]</a> Sepp Hochreiter and J¨urgen Schmidhuber. Flat minima. Neural Computation, 9(1):1–42, 1997.

<a href="#footnote_20_2" name="footnote_20_1">[20]</a> Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In Proceedings of the International Conference on Learning Representations (ICLR), 2014.

<a href="#footnote_21_2" name="footnote_21_1">[21]</a> Diederik P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. In Advances in Neural Information Processing Systems 28, pages 2575–2583, 2015.

<a href="#footnote_22_2" name="footnote_22_1">[22]</a> Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

<a href="#footnote_23_2" name="footnote_23_1">[23]</a> Yann LeCun. Learning invariant feature hierarchies. In Proceedings of the European Conference on Computer Vision (ECCV), pages 496–505, 2012.

<a href="#footnote_24_2" name="footnote_24_1">[24]</a> Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

<a href="#footnote_25_2" name="footnote_25_1">[25]</a> Holden Lee, Rong Ge, Andrej Risteski, Tengyu Ma, and Sanjeev Arora. On the ability of neural nets to express distributions. In Proceedings of Machine Learning Research, volume 65, pages 1–26, 2017.

<a href="#footnote_26_2" name="footnote_26_1">[26]</a> David McAllester. A pac-bayesian tutorial with a dropout bound. arXiv preprint arXiv:1307.2118, 2013.

<a href="#footnote_27_2" name="footnote_27_1">[27]</a> Dmitry Molchanov, Arsenii Ashukha, and Dmitry Vetrov. Variational dropout sparsifies deep neural networks. In Proceedings of the 34 th International Conference on Machine Learning (ICML), 2017.

<a href="#footnote_28_2" name="footnote_28_1">[28]</a> Kirill Neklyudov, Dmitry Molchanov, Arsenii Ashukha, and Dmitry P Vetrov. Structured bayesian pruning via log-normal multiplicative noise. In Advances in Neural Information Processing Systems 30, pages 6775–6784. Curran Associates, Inc., 2017.

<a href="#footnote_29_2" name="footnote_29_1">[29]</a> Behnam Neyshabur, Ruslan R Salakhutdinov, and Nati Srebro. Path-sgd: Path-normalized optimization in deep neural networks. In Advances in Neural Information Processing Systems, pages 2422–2430, 2015.

<a href="#footnote_30_2" name="footnote_30_1">[30]</a> Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro. Exploring generalization in deep learning. In Advances in Neural Information Processing Systems, pages 5949–5958, 2017.

<a href="#footnote_31_2" name="footnote_31_1">[31]</a> Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In Proceedings of the International Conference on Learning Representations (ICLR), 2016.

<a href="#footnote_32_2" name="footnote_32_1">[32]</a> Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information. arXiv preprint arXiv:1703.00810, 2017.

<a href="#footnote_33_2" name="footnote_33_1">[33]</a> Stefano Soatto. Actionable information in vision. In Machine learning for computer vision. Springer, 2013.

<a href="#footnote_34_2" name="footnote_34_1">[34]</a> Stefano Soatto and Alessandro Chiuso. Visual representations: Defining properties and deep approximations. In Proceedings of the International Conference on Learning Representations (ICLR). 2016.

<a href="#footnote_35_2" name="footnote_35_1">[35]</a> Casper Kaae Sønderby, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc Husz´ar. Amortised map inference for image super-resolution. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.

<a href="#footnote_36_2" name="footnote_36_1">[36]</a> Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.

<a href="#footnote_37_2" name="footnote_37_1">[37]</a> Ganesh Sundaramoorthi, Peter Petersen, V. S. Varadarajan, and Stefano Soatto. On the set of images modulo viewpoint and contrast changes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2009.

<a href="#footnote_38_2" name="footnote_38_1">[38]</a> Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In Information Theory Workshop (ITW), 2015 IEEE, pages 1–5. IEEE, 2015.

<a href="#footnote_39_2" name="footnote_39_1">[39]</a> Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. In The 37th annual Allerton Conference on Communication, Control, and Computing, pages 368–377, 1999.

<a href="#footnote_40_2" name="footnote_40_1">[40]</a> Greg Ver Steeg and Aram Galstyan. Maximally informative hierarchical representations of high-dimensional data. In Proceedings of the 18th International Conference on Artificial Intelligence and Statistics, 2015.

<a href="#footnote_41_2" name="footnote_41_1">[41]</a> Shuo Yang, Ping Luo, Chen-Change Loy, and Xiaoou Tang. From facial parts responses to face detection: A deep learning approach. In Proceedings of the IEEE International Conference on Computer Vision, pages 3676–3684, 2015.

<a href="#footnote_42_2" name="footnote_42_1">[42]</a> Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.