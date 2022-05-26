---
layout: post 
title: "(GAN)InfoGAN Translation"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review, 1.2.2.5. GAN, 1.2.2.2. CNN]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

# $\mathbf{InfoGAN:\;Interpretable\;Representation\;Learning\;by}$
# $\mathbf{Information\;Maximizing\;Generative\;Adversarial\;Nets}$

$Abstract$

> This paper describes InfoGAN, an information-theoretic extension to the Generative Adversarial Network that is able to learn disentangled representations in a completely unsupervised manner. InfoGAN is a generative adversarial network that also maximizes the mutual information between a small subset of the latent variables and the observation. We derive a lower bound of the mutual information objective that can be optimized efficiently. Specifically, InfoGAN successfully disentangles writing styles from digit shapes on the MNIST dataset, pose from lighting of 3D rendered images, and background digits from the central digit on the SVHN dataset. It also discovers visual concepts that include hair styles, presence/absence of eyeglasses, and emotions on the CelebA face dataset. Experiments show that InfoGAN learns interpretable representations that are competitive with representations learned by existing supervised methods.
>> 이 논문은 완전히 감독되지 않은 방식으로 얽힌 표현을 학습할 수 있는 생성적 적대 네트워크의 정보 이론적 확장인 InfoGAN을 설명한다. InfoGAN은 잠재 변수의 작은 부분 집합과 관측치 사이의 상호 정보를 최대화하는 생성 적대적 네트워크이다. 우리는 효율적으로 최적화될 수 있는 상호 정보 목표의 하한을 도출한다. 특히, InfoGAN은 MNIST 데이터 세트의 숫자 모양에서 쓰기 스타일을 분리하고, 3D 렌더링된 이미지의 조명에서 포즈를 취하고, SVHN 데이터 세트의 중앙 숫자에서 배경 숫자를 분리하는 데 성공했다. 또한 CelebA 얼굴 데이터 세트에서 헤어 스타일, 안경 유무 및 감정을 포함하는 시각적 개념을 발견한다. 실험에 따르면 InfoGAN은 기존 지도 방법으로 학습한 표현과 경쟁적인 해석 가능한 표현을 학습한다.

$1\;Introduction$

> Unsupervised learning can be described as the general problem of extracting value from unlabelled data which exists in vast quantities. A popular framework for unsupervised learning is that of representation learning [1, 2], whose goal is to use unlabelled data to learn a representation that exposes important semantic features as easily decodable factors. A method that can learn such representations is likely to exist [2], and to be useful for many downstream tasks which include classification, regression, visualization, and policy learning in reinforcement learning.
>> 비지도 학습은 방대한 양으로 존재하는 레이블링되지 않은 데이터에서 가치를 추출하는 일반적인 문제로 설명할 수 있다. 비지도 학습의 인기 있는 프레임워크는 표현 학습[1, 2)으로, 레이블이 지정되지 않은 데이터를 사용하여 중요한 의미적 특징을 쉽게 디코딩할 수 있는 요소로 노출하는 표현을 학습하는 것이 목표이다. 이러한 표현을 학습할 수 있는 방법은 [2]가 존재할 가능성이 높으며, 강화 학습에서 분류, 회귀, 시각화 및 정책 학습을 포함하는 많은 다운스트림 작업에 유용할 수 있다.

> While unsupervised learning is ill-posed because the relevant downstream tasks are unknown at training time, a disentangled representation, one which explicitly represents the salient attributes of a data instance, should be helpful for the relevant but unknown tasks. For example, for a dataset of faces, a useful disentangled representation may allocate a separate set of dimensions for each of the following attributes: facial expression, eye color, hairstyle, presence or absence of eyeglasses, and the identity of the corresponding person. A disentangled representation can be useful for natural tasks that require knowledge of the salient attributes of the data, which include tasks like face recognition and object recognition. It is not the case for unnatural supervised tasks, where the goal could be, for example, to determine whether the number of red pixels in an image is even or odd. Thus, to be useful, an unsupervised learning algorithm must in effect correctly guess the likely set of downstream classification tasks without being directly exposed to them.
>> 비지도 학습은 훈련 시간에 관련 다운스트림 작업을 알 수 없기 때문에 좋지 않지만, 데이터 인스턴스의 두드러진 속성을 명시적으로 나타내는 분리된 표현은 관련적이지만 알려지지 않은 작업에 도움이 되어야 한다. 예를 들어, 얼굴 데이터 세트의 경우, 유용한 분리 표현은 얼굴 표정, 눈 색깔, 헤어스타일, 안경 유무 및 해당 사람의 신원 각각에 대해 별도의 치수 세트를 할당할 수 있다. 분리된 표현은 얼굴 인식 및 객체 인식과 같은 작업을 포함하는 데이터의 두드러진 속성에 대한 지식이 필요한 자연 작업에 유용할 수 있다. 예를 들어, 이미지의 빨간색 픽셀 수가 짝수인지 홀수인지를 결정하는 것이 목표가 될 수 있는 부자연스러운 감독 작업은 해당되지 않는다. 따라서, 유용하기 위해, 비지도 학습 알고리듬은 실제로 직접 노출되지 않고 가능한 일련의 다운스트림 분류 작업을 정확하게 추측해야 한다.

> A significant fraction of unsupervised learning research is driven by generative modelling. It is motivated by the belief that the ability to synthesize, or “create” the observed data entails some form of understanding, and it is hoped that a good generative model will automatically learn a disentangled representation, even though it is easy to construct perfect generative models with arbitrarily bad representations. The most prominent generative models are the variational autoencoder (VAE) [3] and the generative adversarial network (GAN) [4].
>> 비지도 학습 연구의 상당 부분은 생성 모델링에 의해 주도된다. 그것은 관찰된 데이터를 합성하거나 "만드는" 능력이 어떤 형태의 이해를 수반한다는 믿음에서 동기가 부여되며, 비록 임의의 나쁜 표현으로 완벽한 생성 모델을 구성하기는 쉽지만 좋은 생성 모델이 자동으로 분리된 표현을 학습하기를 바란다. 가장 눈에 띄는 생성 모델은 가변 자동 인코더(VAE)[3]와 생성 적대적 네트워크(GAN)[4]이다.

> In this paper, we present a simple modification to the generative adversarial network objective that encourages it to learn interpretable and meaningful representations. We do so by maximizing the mutual information between a fixed small subset of the GAN’s noise variables and the observations, which turns out to be relatively straightforward. Despite its simplicity, we found our method to be surprisingly effective: it was able to discover highly semantic and meaningful hidden representations on a number of image datasets: digits (MNIST), faces (CelebA), and house numbers (SVHN). The quality of our unsupervised disentangled representation matches previous works that made use of supervised label information [5–9]. These results suggest that generative modelling augmented with a mutual information cost could be a fruitful approach for learning disentangled representations.
>> 본 논문에서, 우리는 해석 가능하고 의미 있는 표현을 배우도록 장려하는 생성 적대적 네트워크 목표에 대한 간단한 수정을 제시한다. 우리는 GAN의 잡음 변수의 고정된 작은 부분 집합과 비교적 간단한 관찰 사이의 상호 정보를 최대화함으로써 그렇게 한다. 단순함에도 불구하고, 우리는 우리의 방법이 놀랍도록 효과적이라는 것을 발견했다. 그것은 숫자(MNIST), 얼굴(CelebA), 집 번호(SVHN)와 같은 많은 이미지 데이터 세트에서 매우 의미 있고 의미 있는 숨겨진 표현을 발견할 수 있었다. 감독되지 않은 분리 표현의 품질은 감독된 레이블 정보를 사용한 이전 작업과 일치한다[5–9]. 이러한 결과는 상호 정보 비용으로 증강된 생성 모델링이 분리된 표현을 학습하는 데 유용한 접근법이 될 수 있음을 시사한다.

> In the remainder of the paper, we begin with a review of the related work, noting the supervision that is required by previous methods that learn disentangled representations. Then we review GANs, which is the basis of InfoGAN. We describe how maximizing mutual information results in interpretable representations and derive a simple and efficient algorithm for doing so. Finally, in the experiments section, we first compare InfoGAN with prior approaches on relatively clean datasets and then show that InfoGAN can learn interpretable representations on complex datasets where no previous unsupervised approach is known to learn representations of comparable quality.
>> 본 논문의 나머지 부분에서는 분리된 표현을 학습하는 이전 방법에 필요한 감독에 주목하면서 관련 작업에 대한 검토로 시작한다. 그런 다음 InfoGAN의 기본인 GAN을 검토한다. 우리는 상호 정보를 최대화하는 것이 해석 가능한 표현을 초래하는 방법을 설명하고 이를 위한 간단하고 효율적인 알고리듬을 도출한다. 마지막으로, 실험 섹션에서, 우리는 먼저 InfoGAN을 비교적 깨끗한 데이터 세트에 대한 이전 접근법과 비교 가능한 품질의 표현을 학습하는 이전의 비지도 접근법이 알려져 있지 않은 복잡한 데이터 세트에서 해석 가능한 표현을 학습할 수 있음을 보여준다.

$2\;Related\;Work$

> There exists a large body of work on unsupervised representation learning. Early methods were based on stacked (often denoising) autoencoders or restricted Boltzmann machines [10–13]. A lot of promising recent work originates from the Skip-gram model [14], which inspired the skip-thought vectors [15] and several techniques for unsupervised feature learning of images [16].
>> 감독되지 않은 표현 학습에 대한 많은 연구가 있다. 초기 방법은 적층(종종 노이즈 제거) 자동 인코더 또는 제한된 볼츠만 기계[10–13]를 기반으로 했습니다. 많은 유망한 최근 연구는 스킵-그램 모델[14]에서 비롯되었으며, 이는 스킵-사고 벡터[15]와 이미지의 감독되지 않은 기능 학습을 위한 몇 가지 기술에 영감을 주었다[16].

> Another intriguing line of work consists of the ladder network [17], which has achieved spectacular results on a semi-supervised variant of the MNIST dataset. More recently, a model based on the VAE has achieved even better semi-supervised results on MNIST [18]. GANs [4] have been used by Radford et al. [19] to learn an image representation that supports basic linear algebra on code space. Lake et al. [20] have been able to learn representations using probabilistic inference over Bayesian programs, which achieved convincing one-shot learning results on the OMNI dataset.
>> 또 다른 흥미로운 작업 라인은 MNIST 데이터 세트의 준지도 변형에서 놀라운 결과를 달성한 사다리 네트워크[17]로 구성된다. 보다 최근에는, VAE를 기반으로 한 모델이 MNIST에서 훨씬 더 나은 준지도 결과를 달성했다[18]. GAN[4]은 Radford 등에 의해 사용되었다. [19] 코드 공간에서 기본 선형 대수를 지원하는 이미지 표현을 학습한다. 호수 등. [20] OMNI 데이터 세트에서 설득력 있는 원샷 학습 결과를 달성한 베이지안 프로그램에 대한 확률적 추론을 사용하여 표현을 학습할 수 있었다.

> In addition, prior research attempted to learn disentangled representations using supervised data. One class of such methods trains a subset of the representation to match the supplied label using supervised learning: bilinear models [21] separate style and content; multi-view perceptron [22] separate face identity and view point; and Yang et al. [23] developed a recurrent variant that generates
a sequence of latent factor transformations. Similarly, VAEs [5] and Adversarial Autoencoders [9] were shown to learn representations in which class label is separated from other variations.
>> 또 다른 흥미로운 작업 라인은 MNIST 데이터 세트의 준지도 변형에서 놀라운 결과를 달성한 사다리 네트워크[17]로 구성된다. 보다 최근에는, VAE를 기반으로 한 모델이 MNIST에서 훨씬 더 나은 준지도 결과를 달성했다[18]. GAN[4]은 Radford 등에 의해 사용되었다. [19] 코드 공간에서 기본 선형 대수를 지원하는 이미지 표현을 학습한다. 호수 등. [20] OMNI 데이터 세트에서 설득력 있는 원샷 학습 결과를 달성한 베이지안 프로그램에 대한 확률적 추론을 사용하여 표현을 학습할 수 있었다.

> Recently several weakly supervised methods were developed to remove the need of explicitly labeling variations. disBM [24] is a higher-order Boltzmann machine which learns a disentangled representation by “clamping” a part of the hidden units for a pair of data points that are known to match in all but one factors of variation. DC-IGN [7] extends this “clamping” idea to VAE and successfully learns graphics codes that can represent pose and light in 3D rendered images. This line of work yields impressive results, but they rely on a supervised grouping of the data that is generally not available. Whitney et al. [8] proposed to alleviate the grouping requirement by learning from consecutive frames of images and use temporal continuity as supervisory signal.
>> 최근에 명시적으로 레이블링 변형의 필요성을 제거하기 위해 몇 가지 약하게 지도된 방법이 개발되었다. disBM[24]은 한 가지 변형의 요인을 제외하고 일치하는 것으로 알려진 데이터 포인트 쌍에 대해 숨겨진 단위의 일부를 "클램핑"하여 분리된 표현을 학습하는 고차 볼츠만 기계이다. DC-IGN[7]은 이 "클램핑" 아이디어를 VAE로 확장하고 3D 렌더링된 이미지에서 포즈와 빛을 나타낼 수 있는 그래픽 코드를 성공적으로 학습한다. 이 작업 라인은 인상적인 결과를 제공하지만 일반적으로 사용할 수 없는 데이터의 감독 그룹에 의존한다. 휘트니 외 [8] 연속적인 이미지 프레임에서 학습하여 그룹화 요구 사항을 완화하고 감독 신호로 시간 연속성을 사용할 것을 제안하였다.

> Unlike the cited prior works that strive to recover disentangled representations, InfoGAN requires no supervision of any kind. To the best of our knowledge, the only other unsupervised method that learns disentangled representations is hossRBM [13], a higher-order extension of the spike-and-slab restricted Boltzmann machine that can disentangle emotion from identity on the Toronto Face Dataset [25]. However, hossRBM can only disentangle discrete latent factors, and its computation cost grows exponentially in the number of factors. InfoGAN can disentangle both discrete and continuous latent factors, scale to complicated datasets, and typically requires no more training time than regular GAN.
>> 분리된 표현을 복구하기 위해 노력하는 인용된 이전 작업과 달리 InfoGAN은 어떤 종류의 감독도 요구하지 않는다. 우리가 아는 한, 분리된 표현을 학습하는 유일한 비지도 방법은 토론토 페이스 데이터 세트의 감정과 정체성을 분리할 수 있는 스파이크 앤 슬랩 제한 볼츠만 기계의 고차 확장인 hossRBM[13]이다. 그러나 hossRBM은 이산 잠재 요인만 분리할 수 있으며, 계산 비용은 요인 수에서 기하급수적으로 증가한다. InfoGAN은 이산 및 연속 잠재 요인을 모두 분리하고 복잡한 데이터 세트로 확장할 수 있으며 일반적으로 일반 GAN보다 더 많은 훈련 시간이 필요하지 않다.

$3\;Background:\;Generative\;Adversarial\;Networks$

> Goodfellow et al. [4] introduced the Generative Adversarial Networks (GAN), a framework for training deep generative models using a minimax game. The goal is to learn a generator distribution $P_{G}(x)$ that matches the real data distribution $P_{data}(x)$. Instead of trying to explicitly assign probability to every $x$ in the data distribution, GAN learns a generator network $G$ that generates samples from the generator distribution $P_{G}$ by transforming a noise variable $z\sim{P_{noise}}(z)$ into a sample G(z). This generator is trained by playing against an adversarial discriminator network $D$ that aims to distinguish between samples from the true data distribution Pdata and the generator’s distribution $P_{G}$. So for a given generator, the optimal discriminator is $D(x) = P_{data}(x)/(P_{data}(x) + P_{G}(x))$. More formally, the minimax game is given by the following expression:
>> 굿펠로 외 [4] 미니맥스 게임을 사용하여 심층 생성 모델을 훈련하기 위한 프레임워크인 생성적 적대 네트워크(GAN)를 도입했다. 목표는 실제 데이터 분포 $P_{data}(x)$와 일치하는 생성기 분포 $P_{G}(x)$를 학습하는 것이다. GAN은 데이터 분포의 모든 $x$에 명시적으로 확률을 할당하려고 시도하는 대신, 노이즈 변수 $z\sim{을 변환하여 생성기 분포 $P_{G}$에서 샘플을 생성하는 생성기 네트워크 $G$를 학습한다.P_{noise}}(z)$를 샘플 G(z)로 변환합니다. 이 생성기는 실제 데이터 분포 Pddata와 생성기의 분포 $P_{G}$의 샘플을 구별하는 것을 목표로 하는 적대적 판별기 네트워크 $D$에 대항하여 훈련된다. 따라서 주어진 생성기의 경우 최적의 판별기는 $D(x) = P_{data}(x)/(P_{data}(x) + P_{G}(x)$이다. 미니맥스 게임은 다음과 같은 식으로 표현된다.

$$\underset{G}{\min}\underset{D}{\max}V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log{D(x)}]+\mathbb{E}_{z\sim{noise}}[\log{(1-D(G(z)))}].$$

$4\;Mutual\;Information\;for\;Inducing\;Latent\;Codes$

> The GAN formulation uses a simple factored continuous input noise vector $z$, while imposing no restrictions on the manner in which the generator may use this noise. As a result, it is possible that the noise will be used by the generator in a highly entangled way, causing the individual dimensions of $z$ to not correspond to semantic features of the data.
>> GAN 공식은 단순 인수 연속 입력 노이즈 벡터 $z$를 사용하는 반면, 발전기가 이 노이즈를 사용할 수 있는 방법에 제한을 두지 않는다. 결과적으로, $z$의 개별 차원이 데이터의 의미적 특징과 일치하지 않는 매우 얽힌 방식으로 노이즈가 제너레이터에 의해 사용될 수 있다.

> However, many domains naturally decompose into a set of semantically meaningful factors of variation. For instance, when generating images from the MNIST dataset, it would be ideal if the model automatically chose to allocate a discrete random variable to represent the numerical identity of the digit (0-9), and chose to have two additional continuous variables that represent the digit’s angle and thickness of the digit’s stroke. It is the case that these attributes are both independent and salient, and it would be useful if we could recover these concepts without any supervision, by simply specifying that an MNIST digit is generated by an independent 1-of-10 variable and two independent continuous variables.
>> 그러나 많은 도메인은 자연스럽게 의미적으로 의미 있는 변동 요인 집합으로 분해된다. 예를 들어, MNIST 데이터 세트에서 이미지를 생성할 때 모델이 자동으로 이산 랜덤 변수를 할당하여 자릿수의 각도와 자릿수의 스트로크 두께를 나타내는 두 개의 추가 연속 변수를 갖는 것이 이상적이다. 이러한 속성은 독립적이고 두드러진 경우이며, MNIST 숫자가 독립적 10분의 1 변수와 2개의 독립적 연속 변수에 의해 생성된다는 것을 단순히 지정함으로써 감독 없이 이러한 개념을 복구할 수 있다면 유용할 것이다.

> In this paper, rather than using a single unstructured noise vector, we propose to decompose the input noise vector into two parts: (i) $z$, which is treated as source of incompressible noise; (ii) $c$ which we will call the latent code and will target the salient structured semantic features of the data distribution. Mathematically, we denote the set of structured latent variables by $c_{1}, c_{2}, . . . , c_{L}$. In its simplest form, we may assume a factored distribution, given by $P(c_{1}, c_{2}, . . . , c_{L}) = \prod_{i=1}^{L}P(c_{i})$. For ease of notation, we will use latent codes $c$ to denote the concatenation of all latent variables $c_{i}$.
>> 본 논문에서는 단일 비정형 노이즈 벡터를 사용하는 대신 입력 노이즈 벡터를 두 부분으로 분해할 것을 제안한다. (i) 압축 불가능한 노이즈의 소스로 취급되는 $z$; (ii) 잠재 코드라고 하며 데이터 분포의 두드러진 구조적 의미적 특징을 대상으로 할 $c$. 수학적으로, 우리는 구조화된 잠재 변수 집합을 $c_{1}, c_{2}, \cdots c_{L}$로 나타낸다. 가장 간단한 형태에서, 우리는 $P(c_{1}, c_{2}, . . . , c_{L}) = \prod_{i=1}^{L}P(c_{i})$로 주어진 인자 분포를 가정할 수 있다. 표기의 용이성을 위해 모든 잠재 변수 $c_{i}$의 연결을 나타내기 위해 잠재 코드 $c$를 사용할 것이다.

> We now propose a method for discovering these latent factors in an unsupervised way: we provide the generator network with both the incompressible noise $z$ and the latent code $c$ so the form of the generator becomes $G(z, c)$. However, in standard GAN, the generator is free to ignore the additional latent code $c$ by finding a solution satisfying $P_{G}(x|c) = P_{G}(x)$. To cope with the problem of trivial codes, we propose an information-theoretic regularization: there should be high mutual information between latent codes $c$ and generator distribution $G(z, c)$. Thus $I(c; G(z, c))$ should be high.
>> 우리는 이제 이러한 잠재 요인을 비지도 방식으로 발견하는 방법을 제안한다. 우리는 발전기 네트워크에 압축할 수 없는 노이즈 $z$와 잠재 코드 $c$를 모두 제공하여 발전기의 형태가 $G(z, c)$가 되도록 한다. 그러나 표준 GAN에서 생성기는 $P_{G}(x|c) = P_{G}(x)$를 만족하는 솔루션을 찾아 추가 잠재 코드 $c$를 자유롭게 무시할 수 있다. 사소한 코드 문제에 대처하기 위해 정보 이론적 정규화를 제안한다. 잠재 코드 $c$와 생성기 분포 $G(z, c)$ 사이에 높은 상호 정보가 있어야 한다. 따라서 $I(c; G(z, c))$는 높아야 한다.

> In information theory, mutual information between $x$ and $Y , I(X; Y)$, measures the “amount of information” learned from knowledge of random variable $Y$ about the other random variable $X$. The mutual information can be expressed as the difference of two entropy terms:
>> 정보 이론에서, $x$와 $Y, I(X; Y)$ 사이의 상호 정보는 다른 무작위 변수 $X$에 대한 무작위 변수 $Y$의 지식으로부터 학습된 "정보의 양"을 측정한다. 상호 정보는 두 엔트로피 용어의 차이로 표현될 수 있다.

$$I(X;Y)=H(X)−H(X|Y)=H(Y)−H(Y|X)$$

> This definition has an intuitive interpretation: $I(X; Y)$ is the reduction of uncertainty in $x$ when $Y$ is observed. If $X$ and $Y$ are independent, then $I(X; Y ) = 0$, because knowing one variable reveals nothing about the other; by contrast, if $X$ and $Y$ are related by a deterministic, invertible function, then maximal mutual information is attained. This interpretation makes it easy to formulate a cost: given any $x\sim{P_{G}(x)}$, we want $P_{G}(c|x)$ to have a small entropy. In other words, the information in the latent code $c$ should not be lost in the generation process. Similar mutual information inspired objectives have been considered before in the context of clustering [26–28]. Therefore, we propose to solve the following information-regularized minimax game:
>> 이 정의는 직관적인 해석을 가지고 있다. $I(X; Y)$는 $Y$가 관찰될 때 $x$의 불확실성 감소이다. $X$와 $Y$가 독립적이라면, 한 변수를 알면 다른 변수에 대해 아무것도 드러나지 않기 때문에 $I(X; Y) = 0$이다. 반대로 $X$와 $Y$가 결정론적, 반전 가능한 함수에 의해 관련된다면, 최대 상호 정보가 달성된다. 이 해석은 $x\sim{P_{G}(x)}$에 주어진 비용을 쉽게 공식화할 수 있게 한다. 우리는 $P_{G}(c|x)$가 작은 엔트로피를 가지기를 원한다. 즉, 잠재 코드 $c$의 정보가 생성 과정에서 손실되어서는 안 된다. 유사한 상호 정보 영감을 받은 목표는 클러스터링의 맥락에서 이전에 고려되었다[26–28]. 따라서 다음과 같은 정보 정규화 미니맥스 게임을 해결할 것을 제안한다.

$$\underset{G}{\min}\underset{D}{\max}V_{I}(D,G)=V(D,G)-\lambda{I(c;G(z,c))}$$

$5\;Variational\;Mutual\;Information\;Maximization$

> In practice, the mutual information term $I(c; G(z, c))$ is hard to maximize directly as it requires access to the posterior $P(c|x)$. Fortunately we can obtain a lower bound of it by defining an auxiliary distribution $Q(c|x)$ to approximate $P(c|x)$:
>> 실제로 상호 정보 용어 $I(c; G(z, c))$는 사후 $P(c|x)$에 대한 액세스가 필요하기 때문에 직접 최대화하기 어렵다. 다행히도, 우리는 보조 분포 $Q(c|x)$를 $P(c|x)$에 근사하도록 정의함으로써 그것의 하한을 얻을 수 있다.

$$I(c;G(z,c))=H(c)-H(c|G(z,c))\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$=\mathbb{E}_{x\sim{G(z,c)}}[\mathbb{E}_{c'\sim{P(c|x)}}[\log{P(c'|x)}]]+H(c)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$=\mathbb{E}_{x\sim{G(z,c)}}\underbrace{[D_{KL}(P(\cdot{|x})\parallel{Q(\cdot{|x})})}_{\geq{0}}+\mathbb{E}_{c'\sim{P(c|x)}}[\log{Q(c'|x)}]]+H(c)$$

$$\geq{\mathbb{E}_{x\sim{G(z,c)}}[\mathbb{E}_{c'\sim{P(c|x)}}[\log{Q(c'|x)}]]+H(c)}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

> This technique of lower bounding mutual information is known as Variational Information Maximization [29]. We note in addition that the entropy of latent codes $H(c)$ can be optimized over as well since for common distributions it has a simple analytical form. However, in this paper we opt for simplicity by fixing the latent code distribution and we will treat $H(c)$ as a constant. So far we have bypassed the problem of having to compute the posterior $P(c|x)$ explicitly via this lower bound but we still need to be able to sample from the posterior in the inner expectation. Next we state a simple lemma, with its proof deferred to Appendix, that removes the need to sample from the posterior.
>> 하한 상호 정보의 이 기법을 변동 정보 최대화[29]라고 한다. 또한 공통 분포의 경우 간단한 분석 형태를 가지기 때문에 잠재 코드 $H(c)$의 엔트로피를 최적화할 수 있다. 그러나 이 논문에서 우리는 잠재 코드 분포를 수정하여 단순성을 선택하고 $H(c)$를 상수로 처리할 것이다. 지금까지 우리는 이 하한을 통해 명시적으로 사후 $P(c|x)$를 계산해야 하는 문제를 우회했지만, 여전히 내부 기대에서 사후에서 샘플링할 수 있어야 한다. 다음으로, 우리는 후부에서 표본을 추출할 필요성을 제거하는, 증거를 부록으로 미루는 간단한 개요를 설명한다.

> **Lemma 5.1** For random variables $X, Y$ and *function* $f(x, y)$ under suitable regularity conditions:
>> **레마 5.1*** 적절한 규칙성 조건에서 랜덤 변수 $X, Y$ 및 *function* $f(x, y)$의 경우:

$E_{x∼X,y∼Y|x}[f(x, y)] = E_{x∼X,y∼Y|x,x'∼X|y}[f(x',y)]$.

> By using Lemma A.1, we can define a variational lower bound, $L_{I} (G, Q)$, of the mutual information, $I(c; G(z, c))$:
>> Lemma A.1을 사용하여 상호 정보 $I(c; G(z, c))$의 변동 하한 $L_{I}(G, Q)$를 정의할 수 있다.

$$L_{I}(G, Q)=E_{c∼P(c),x\sim G(z,c)}[\log{Q(c|x)}] + H(c)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$= E_{x\sim{G(z,c)}}[\mathbb{E}_{C'\sim{P(c|x)}}[\log{Q(c'|x)}]]+H(c)$$

$$\leq I(c;G(z,c))\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$


> We note that $L_{I} (G, Q)$ is easy to approximate with Monte Carlo simulation. In particular, $L_{I}$ can be maximized w.r.t. $Q$ directly and w.r.t. $G$ via the reparametrization trick. Hence $L_{I} (G, Q)$ can be added to GAN’s objectives with no change to GAN’s training procedure and we call the resulting algorithm Information Maximizing Generative Adversarial Networks (InfoGAN).
>> $L_{I}(G, Q)$는 몬테카를로 시뮬레이션으로 근사하기 쉽다. 특히, $L_{I}$는 리파라미터화 트릭을 통해 직접 w.r.t. $Q$와 w.r.t. $G$를 최대화할 수 있다. 따라서 $L_{I}(G,Q)$는 GAN의 훈련 절차를 변경하지 않고 GAN의 목표에 추가할 수 있으며, 우리는 그 결과 알고리듬을 생성적 적대 네트워크 최대화(InfoGAN)라고 부른다.

> Eq (4) shows that the lower bound becomes tight as the auxiliary distribution $Q$ approaches the true posterior distribution: $\mathbb{E}_{x}[D_{KL}(P(·|x)\parallel Q(·|x))]\to0$. In addition, we know that when the variational lower bound attains its maximum $L_{I} (G, Q) = H(c)$ for discrete latent codes, the bound becomes tight and the maximal mutual information is achieved. In Appendix, we note how InfoGAN can be connected to the Wake-Sleep algorithm [30] to provide an alternative interpretation.
>> Eq(4)는 보조 분포 $Q$가 실제 후방 분포에 접근할수록 하한이 엄격해진다는 것을 보여준다. $\mathbb{E}_{x}[D_{KL}(P(·|x)\parallel Q(·|x))]\to0$입니다. 또한, 우리는 변동 하한이 이산 잠재 코드에 대해 최대 $L_{I}(G, Q) = H(c)$에 도달하면 경계가 엄격해지고 최대 상호 정보가 달성된다는 것을 안다. 부록에서 InfoGAN을 Wake-Sleep 알고리듬[30]에 연결하여 대체 해석을 제공하는 방법에 주목한다.

> Hence, InfoGAN is defined as the following minimax game with a variational regularization of mutual information and a hyperparameter $\lambda$:
>> 따라서 InfoGAN은 상호 정보의 변형 정규화와 하이퍼 매개 변수 $\lambda$를 가진 다음과 같은 미니맥스 게임으로 정의된다.

$$\underset{G,Q}{\min}\underset{D}{\max}V_{InfoGAN}(D,G,Q)=V(D,G)-\lambda{L_{I}}(G,Q)$$

$6\;Implementation$

> In practice, we parametrize the auxiliary distribution $Q$ as a neural network. In most experiments, $Q$ and $D$ share all convolutional layers and there is one final fully connected layer to output parameters for the conditional distribution $Q(c|x)$, which means InfoGAN only adds a negligible computation cost to GAN. We have also observed that $L_{I} (G, Q)$ always converges faster than normal GAN objectives and hence InfoGAN essentially comes for free with GAN.
>> 실제로, 우리는 보조 분포 $Q$를 신경망으로 매개 변수화한다. 대부분의 실험에서 $Q$와 $D$는 모든 컨볼루션 레이어를 공유하고 조건부 분포 $Q(c|x)$에 대한 매개 변수를 출력하기 위해 완전히 연결된 레이어가 하나 있는데, 이는 InfoGAN이 GAN에 무시할 수 있는 계산 비용만 추가한다는 것을 의미한다. 또한 $L_{I}(G,Q)$는 항상 일반적인 GAN 목표보다 빠르게 수렴하므로 InfoGAN은 GAN에서 기본적으로 무료로 제공된다.

> For categorical latent code $c_{i}$ , we use the natural choice of softmax nonlinearity to represent $Q(c_{i}|x)$. For continuous latent code $c_{j}$ , there are more options depending on what is the true posterior $P(c_{j}|x)$. In our experiments, we have found that simply treating $Q(c_{j}|x)$ as a factored Gaussian is sufficient.
>> 범주형 잠재 코드 $c_{i}$의 경우 소프트맥스 비선형성의 자연스러운 선택을 사용하여 $Q(c_{i}|x)$를 나타낸다. 연속 잠재 코드 $c_{j}$의 경우 실제 사후 $P(c_{j}|x)$가 무엇인지에 따라 더 많은 옵션이 있다. 우리의 실험에서, 우리는 $Q(c_{j}|x)$를 인수 가우스로서 처리하는 것만으로도 충분하다는 것을 발견했다.

> Even though InfoGAN introduces an extra hyperparameter $\lambda$, it’s easy to tune and simply setting to 1 is sufficient for discrete latent codes. When the latent code contains continuous variables, a smaller $\lambda$ is typically used to ensure that $\lambda L_{I} (G, Q)$, which now involves differential entropy, is on the same scale as GAN objectives.
>> InfoGAN이 추가 하이퍼 매개 변수 $\lambda$를 도입하더라도 조정하기 쉽고 이산 잠재 코드의 경우 1로 설정하는 것만으로도 충분하다. 잠재 코드에 연속 변수가 포함되어 있는 경우, 더 작은 $\lambda$가 일반적으로 사용되어 이제 차등 엔트로피를 포함하는 $\lambda L_{I}(G,Q)$가 GAN 목표와 동일한 척도를 갖도록 보장한다.

> Since GAN is known to be difficult to train, we design our experiments based on existing techniques introduced by DC-GAN [19], which are enough to stabilize InfoGAN training and we did not have to introduce new trick. Detailed experimental setup is described in Appendix.
>> GAN은 훈련하기 어려운 것으로 알려져 있기 때문에 DC-GAN[19]이 도입한 기존 기술을 기반으로 실험을 설계하는데, 이는 InfoGAN 훈련을 안정화하기에 충분하고 새로운 트릭을 도입할 필요가 없었다. 자세한 실험 설정은 부록에 설명되어 있습니다.GAN은 훈련하기 어려운 것으로 알려져 있기 때문에 DC-GAN[19]이 도입한 기존 기술을 기반으로 실험을 설계하는데, 이는 InfoGAN 훈련을 안정화하기에 충분하고 새로운 트릭을 도입할 필요가 없었다. 자세한 실험 설정은 부록에 설명되어 있습니다.

$7\;Experiments$

> The first goal of our experiments is to investigate if mutual information can be maximized efficiently. The second goal is to evaluate if InfoGAN can learn disentangled and interpretable representations by making use of the generator to vary only one latent factor at a time in order to assess if varying such factor results in only one type of semantic variation in generated images. DC-IGN [7] also uses this method to evaluate their learned representations on 3D image datasets, on which we also apply InfoGAN to establish direct comparison.
>> 우리 실험의 첫 번째 목표는 상호 정보가 효율적으로 최대화될 수 있는지 조사하는 것이다. 두 번째 목표는 InfoGAN이 생성자를 사용하여 한 번에 하나의 잠재 요인만 변화시켜 생성된 이미지에 한 가지 유형의 의미 변화만 초래하는지 여부를 평가함으로써 분리되고 해석 가능한 표현을 학습할 수 있는지 여부를 평가하는 것이다. DC-IGN[7]은 또한 이 방법을 사용하여 3D 이미지 데이터 세트에서 학습된 표현을 평가하며, 우리는 또한 InfoGAN을 적용하여 직접 비교를 설정한다.

$7.1\;Mutual\;Information\;Maximization$

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Figure-1.JPG)

> Figure 1: Lower bound $L_{I}$ over training iterations
>> 그림 1: 교육 반복에 대한 하한 $L_{I}$

> To evaluate whether the mutual information between latent codes $c$ and generated images $G(z, c)$ can be maximized efficiently with proposed method, we train InfoGAN on MNIST dataset with a uniform categorical distribution on latent codes $c\sim Cat(K = 10, p = 0.1)$. In Fig 1, the lower bound $L_{I}(G,Q)$ is quickly maximized to $H(c) ≈ 2.30$, which means the bound (4) is tight and maximal mutual information
is achieved.
>> 잠재 코드 $c$와 생성된 이미지 $G(z, c)$ 사이의 상호 정보를 제안된 방법으로 효율적으로 극대화할 수 있는지 평가하기 위해 잠재 코드 $c\sim Cat(K = 10, p = 0.1)$에 대한 균일한 범주 분포를 사용하여 MNIST 데이터 세트에서 InfoGAN을 훈련한다. 그림 1에서 하한 $L_{I}(G,Q)$는 빠르게 $H(c) → 2.30$로 최대화되며, 이는 경계 (4)가 엄격하고 상호 정보가 최대화됨을 의미한다.
달성되었습니다.

> As a baseline, we also train a regular GAN with an auxiliary distribution $Q$ when the generator is not explicitly encouraged to maximize the mutual information with the latent codes. Since we use expressive neural network to parametrize $Q$, we can assume that $Q$ reasonably approximates the true posterior $P(c|x)$ and hence there is little mutual information between latent codes and generated images in regular GAN. We note that with a different neural network architecture, there might be a higher mutual information between latent codes and generated images even though we have not observed such case in our experiments. This comparison is meant to demonstrate that in a regular GAN, there is no guarantee that the generator will make use of the latent codes.
>> 기준으로서, 우리는 또한 발전기가 잠재 코드로 상호 정보를 최대화하도록 명시적으로 권장되지 않을 때 보조 분포 $Q$를 가진 정규 GAN을 훈련시킨다. 표현 신경망을 사용하여 $Q$를 매개 변수화하기 때문에 $Q$는 실제 사후 $P(c|x)$에 합리적으로 근사하므로 일반 GAN에서 잠재 코드와 생성된 이미지 사이에 상호 정보가 거의 없다고 가정할 수 있다. 우리는 실험에서 그러한 경우를 관찰하지 않았음에도 불구하고, 다른 신경망 아키텍처를 사용하면 잠재 코드와 생성된 이미지 사이에 더 높은 상호 정보가 있을 수 있다는 점에 주목한다. 이 비교는 일반 GAN에서 발전기가 잠재 코드를 사용한다는 보장이 없다는 것을 입증하기 위한 것이다.

$7.2\;Disentangled\;Representation$

> To disentangle digit shape from styles on MNIST, we choose to model the latent codes with one categorical code, $c_{1}\sim{Cat(K = 10, p = 0.1)}$, which can model discontinuous variation in data, and two continuous codes that can capture variations that are continuous in nature: $c_{2}, c_{3}\sim Unif(−1, 1)$.
>> MNIST의 스타일에서 숫자 모양을 분리하기 위해 데이터의 불연속적 변동을 모델링할 수 있는 하나의 범주 코드 $c_{1}\sim{Cat(K = 10, p = 0.1)}$와 본질적으로 연속적인 변동을 캡처할 수 있는 두 개의 연속 코드 $c_{2}, c_{3}\sim Unif(−1, 1)$로 잠재 코드를 모델링하기로 선택한다.

> In Figure 2, we show that the discrete code $c_{1}$  captures drastic change in shape. Changing categorical code $c_{1}$  switches between digits most of the time. In fact even if we just train InfoGAN without any label, $c_{1}$  can be used as a classifier that achieves 5% error rate in classifying MNIST digits by matching each category in $c_{1}$  to a digit type. In the second row of Figure 2a, we can observe a digit 7 is classified as a 9.
>> 그림 2에서, 우리는 이산 코드 $c_{1}$가 형상의 급격한 변화를 포착한다는 것을 보여준다. 범주 코드 $c_{1}$를 변경하면 대부분의 경우 숫자 간에 전환됩니다. 실제로 레이블 없이 InfoGAN만 훈련하더라도 $c_{1}$는 $c_{1}$의 각 범주를 숫자 유형에 일치시켜 MNIST 숫자를 분류하는 데 5%의 오류율을 달성하는 분류기로 사용될 수 있다. 그림 2a의 두 번째 줄에서, 우리는 숫자 7이 9로 분류되는 것을 볼 수 있다.

> Continuous codes $c_{2}, c_{3}$ capture continuous variations in style: $c_{2}$ models rotation of digits and $c_{3}$ controls the width. What is remarkable is that in both cases, the generator does not simply stretch or rotate the digits but instead adjust other details like thickness or stroke style to make sure the resulting images are natural looking. As a test to check whether the latent representation learned by InfoGAN is generalizable, we manipulated the latent codes in an exaggerated way: instead of plotting latent codes from −1 to 1, we plot it from −2 to 2 covering a wide region that the network was never trained on and we still get meaningful generalization.
>> 연속 코드 $c_{2}, c_{3}$는 스타일의 연속적인 변화를 포착한다. $c_{2}$는 숫자의 회전을 모델링하고 $c_{3}$는 폭을 제어한다. 주목할 만한 점은 두 경우 모두 제너레이터가 단순히 숫자를 늘리거나 회전시키는 것이 아니라 두께나 스트로크 스타일과 같은 다른 세부 정보를 조정하여 결과 이미지가 자연스럽게 보이도록 한다는 것입니다. InfoGAN에 의해 학습된 잠재 표현이 일반화 가능한지 확인하기 위한 테스트로, 우리는 -1부터 1까지 잠재 코드를 그리는 대신 네트워크가 훈련되지 않은 광범위한 영역을 커버하는 -2부터 2까지를 플롯하고 여전히 의미 있는 일반화를 얻는다.

> Next we evaluate InfoGAN on two datasets of 3D images: faces [31] and chairs [32], on which DC-IGN was shown to learn highly interpretable graphics codes.
>> 다음으로, 우리는 3D 이미지의 두 가지 데이터 세트인 얼굴[31]과 의자[32]에 대해 InfoGAN을 평가하는데, 이 데이터 세트에서 DC-IGN은 해석 가능한 그래픽 코드를 학습하는 것으로 나타났다.

> On the faces dataset, DC-IGN learns to represent latent factors as azimuth (pose), elevation, and lighting as continuous latent variables by using supervision. Using the same dataset, we demonstrate that InfoGAN learns a disentangled representation that recover azimuth (pose), elevation, and lighting on the same dataset. In this experiment, we choose to model the latent codes with five continuous codes, $c_{i}\sim Unif(−1, 1)$ with $1 ≤ i ≤ 5$.
>> 얼굴 데이터 세트에서 DC-IGN은 감독을 사용하여 잠재 요인을 방위(포즈), 고도 및 조명을 연속 잠재 변수로 표현하는 방법을 학습한다. 동일한 데이터 세트를 사용하여 InfoGAN이 동일한 데이터 세트에서 방위각(포즈), 고도 및 조명을 복구하는 분리된 표현을 학습한다는 것을 보여준다. 이 실험에서, 우리는 5개의 연속 코드인 $c_{i}\sim Unif(-1, 1)$와 $1μi ≤ 5$로 잠재 코드를 모델링하기로 선택한다.

> Since DC-IGN requires supervision, it was previously not possible to learn a latent code for a variation that’s unlabeled and hence salient latent factors of variation cannot be discovered automatically from data. By contrast, InfoGAN is able to discover such  variation on its own: for instance, in Figure 3d a latent code that smoothly changes a face from wide to narrow is learned even though this variation was neither explicitly generated or labeled in prior work.
>> DC-IGN은 감독을 필요로 하기 때문에 이전에는 레이블이 지정되지 않은 변동에 대한 잠재 코드를 학습할 수 없었으므로 변동의 두드러진 잠재 요인을 데이터에서 자동으로 발견할 수 없었다. 대조적으로, InfoGAN은 그러한 변화를 스스로 발견할 수 있다. 예를 들어, 그림 3d에서는 이러한 변화가 이전 연구에서 명시적으로 생성되거나 레이블링되지 않았음에도 불구하고 얼굴을 넓음에서 좁음으로 부드럽게 바꾸는 잠재 코드가 학습된다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Figure-2.JPG)

> Figure 2: Manipulating latent codes on MNIST: In all figures of latent code manipulation, we will use the convention that in each one latent code varies from left to right while the other latent codes and noise are fixed. The different rows correspond to different random samples of fixed latent codes and noise. For instance, in (a), one column contains five samples from the same category in $c_{1}$ , and a row shows the generated images for 10 possible categories in $c_{1}$  with other noise fixed. In (a), each category in $c_{1}$  largely corresponds to one digit type; in (b), varying $c_{1}$  on a GAN trained without information regularization results in non-interpretable variations; in (c), a small value of $c_{2}$ denotes left leaning digit whereas a high value corresponds to right leaning digit; in (d), $c_{3}$ smoothly controls the width. We reorder (a) for visualization purpose, as the categorical code is inherently unordered.
>> 그림 2: MNIST의 잠재 코드 조작: 잠재 코드 조작의 모든 수치에서, 우리는 각각의 잠재 코드가 왼쪽에서 오른쪽으로 변화하는 반면 다른 잠재 코드와 노이즈는 고정된다는 규칙을 사용할 것이다. 상이한 행들은 고정된 잠재 코드들 및 노이즈의 상이한 랜덤 샘플들에 대응한다. 예를 들어, (a)의 한 열은 $c_{1}$의 동일한 범주에서 5개의 샘플을 포함하고, 행은 다른 노이즈가 고정된 $c_{1}$의 가능한 범주 10개에 대해 생성된 이미지를 보여준다. (a)에서 $c_{1}$의 각 범주는 대체로 한 자리 유형에 대응한다. (b)에서 정보 정규화 없이 훈련된 GAN에서 $c_{1}$를 변경하면 해석할 수 없는 변동이 발생한다. (c)에서 $c_{2}$의 작은 값은 왼쪽 기울어진 숫자를 나타내는 반면, 높은 값은 오른쪽 기울어진 숫자에 대응한다. (d)에서 $c_{3}$는 매끄럽게 넓은 범위를 제어한다.t. 범주 코드는 본질적으로 순서가 없기 때문에 시각화 목적으로 (a) 순서를 변경한다.

> On the chairs dataset, DC-IGN can learn a continuous code that representes rotation. InfoGAN again is able to learn the same concept as a continuous code (Figure 4a) and we show in addition that InfoGAN is also able to continuously interpolate between similar chair types of different widths using a single continuous code (Figure 4b). In this experiment, we choose to model the latent factors with four categorical codes, $c_{1}, c_{2}, c_{3}, c_{4}\sim Cat(K = 20, p = 0.05)$ and one continuous code $c5\sim Unif(−1, 1)$.
>> 의자 데이터 세트에서 DC-IGN은 회전을 나타내는 연속 코드를 학습할 수 있다. InfoGAN은 다시 연속 코드와 동일한 개념을 학습할 수 있으며(그림 4a), 또한 InfoGAN이 단일 연속 코드를 사용하여 서로 다른 폭의 유사한 의자 유형 간에 연속적으로 보간할 수 있음을 보여준다(그림 4b). 이 실험에서, 우리는 4개의 범주 코드인 $c_{1}, c_{2}, c_{3}, c_{4}\sim Cat(K = 20, p = 0.05)$와 하나의 연속 코드 $c5\sim Unif(-1, 1)$로 잠재 요인을 모델링하기로 선택한다.

> Next we evaluate InfoGAN on the Street View House Number (SVHN) dataset, which is significantly more challenging to learn an interpretable representation because it is noisy, containing images of variable-resolution and distracting digits, and it does not have multiple variations of the same object. In this experiment, we make use of four 10−dimensional categorical variables and two uniform
continuous variables as latent codes. We show two of the learned latent factors in Figure 5.
>> 다음으로 InfoGAN을 Street View House Number(SVHN) 데이터 세트에서 평가하는데, 이는 가변 해상도와 산만한 숫자의 이미지가 포함되어 있고 동일한 객체의 여러 변형이 없기 때문에 해석 가능한 표현을 학습하는 것이 훨씬 더 어렵다. 이 실험에서, 우리는 4개의 10차원 범주형 변수와 2개의 균일한 범주형 변수를 사용한다.
연속형 변수를 잠재 코드로 사용할 수 있습니다. 우리는 그림 5에서 학습된 잠재 요인 중 두 가지를 보여준다.

> Finally we show in Figure 6 that InfoGAN is able to learn many visual concepts on another challenging dataset: CelebA [33], which includes 200, 000 celebrity images with large pose variations and background clutter. In this dataset, we model the latent variation as 10 uniform categorical variables, each of dimension 10. Surprisingly, even in this complicated dataset, InfoGAN can recover azimuth
as in 3D images even though in this dataset no single face appears in multiple pose positions. Moreover InfoGAN can disentangle other highly semantic variations like presence or absence of glasses, hairstyles and emotion, demonstrating a level of visual understanding is acquired without any supervision.
>> 마지막으로 그림 6에서 InfoGAN이 다른 까다로운 데이터 세트에서 많은 시각적 개념을 학습할 수 있음을 보여준다. 셀럽A[33]는 큰 포즈 변형과 배경 잡동사니가 있는 200,000개의 연예인 이미지를 포함한다. 이 데이터 세트에서, 우리는 잠재 변동을 10개의 균일한 범주형 변수로 모델링한다. 놀랍게도, 이 복잡한 데이터 세트에서도 InfoGAN은 방위각을 복구할 수 있다.
이 데이터 집합에서 여러 포즈 위치에 단일 얼굴이 나타나지 않더라도 3D 영상에서처럼 말이다. 더욱이 InfoGAN은 안경 유무, 헤어스타일 및 감정과 같은 다른 고도의 의미론적 변화를 분리할 수 있으며, 이는 시각적 이해 수준이 어떠한 감독 없이 획득된다는 것을 보여준다.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Figure-3.JPG)

> Figure 3: Manipulating latent codes on 3D Faces: We show the effect of the learned continuous latent factors on the outputs as their values vary from −1 to 1. In (a), we show that one of the continuous latent codes consistently captures the azimuth of the face across different shapes; in (b), the continuous code captures elevation; in (c), the continuous code captures the orientation of lighting; and finally in (d), the continuous code learns to interpolate between wide and narrow faces while preserving other visual features. For each factor, we present the representation that most resembles prior supervised results [7] out of 5 random runs to provide direct comparison.
>> 그림 3: 3D 면의 잠재 코드 조작: 우리는 -1부터 1까지 값이 달라짐에 따라 학습된 연속 잠재 인자가 출력에 미치는 영향을 보여준다. (a)에서는 연속 잠재 코드 중 하나가 다른 모양에 걸쳐 얼굴의 방위각을 일관되게 포착한다는 것을 보여준다. (b)에서는 연속 코드가 표고를 포착하고, (c)에서는 연속 코드가 조명의 방향을 포착한다.; 마지막으로 (d)에서 연속 코드는 다른 시각적 특징을 보존하면서 넓고 좁은 얼굴 사이를 보간하는 방법을 학습한다. 각 요인에 대해, 우리는 직접 비교를 제공하기 위해 5개의 무작위 실행 중 이전 감독 결과[7]와 가장 유사한 표현을 제시한다.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Figure-4.JPG)

> Figure 4: Manipulating latent codes on 3D Chairs: In (a), we show that the continuous code captures the pose of the chair while preserving its shape, although the learned pose mapping varies across different types; in (b), we show that the continuous code can alternatively learn to capture the widths of different chair types, and smoothly interpolate between them. For each factor, we present the representation that most resembles prior supervised results [7] out of 5 random runs to provide direct comparison.
>> 그림 4: 3D 의자에서 잠재 코드 조작: (a)에서, 우리는 학습된 포즈 매핑이 다른 유형에 따라 다르지만, 의자의 모양을 유지하면서 연속 코드가 의자의 포즈를 포착한다는 것을 보여준다. (b)에서, 우리는 연속 코드가 다른 의자 유형의 폭을 포착하고 부드럽게 인터폴라를 학습할 수 있다는 것을 보여준다.그들 사이를 갈라놓다 각 요인에 대해, 우리는 직접 비교를 제공하기 위해 5개의 무작위 실행 중 이전 감독 결과[7]와 가장 유사한 표현을 제시한다.

$8\;Conclusion$

> This paper introduces a representation learning algorithm called Information Maximizing Generative Adversarial Networks (InfoGAN). In contrast to previous approaches, which require supervision, InfoGAN is completely unsupervised and learns interpretable and disentangled representations on challenging datasets. In addition, InfoGAN adds only negligible computation cost on top of GAN and is easy to train. The core idea of using mutual information to induce representation can be applied to other methods like VAE [3], which is a promising area of future work. Other possible extensions to this work include: learning hierarchical latent representations,  improving semi-supervised learning with better codes [34], and using InfoGAN as a high-dimensional data discovery tool.
>> 이 논문은 정보 최대화 생성 적대적 네트워크(InfoGAN)라는 표현 학습 알고리듬을 소개한다. 감독이 필요한 이전의 접근 방식과 달리, InfoGAN은 완전히 감독되지 않고 까다로운 데이터 세트에서 해석 가능하고 분리된 표현을 학습한다. 또한 InfoGAN은 GAN 위에 무시할 수 있는 계산 비용만 추가하고 훈련하기 쉽다. 상호 정보를 사용하여 표현을 유도하는 핵심 아이디어는 향후 작업의 유망한 영역인 VAE[3]와 같은 다른 방법에도 적용될 수 있다. 이 작업의 다른 가능한 확장으로는 계층적 잠재 표현 학습, 더 나은 코드로 준지도 학습 개선[34], InfoGAN을 고차원 데이터 검색 도구로 사용하는 등이 있다.

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Figure-5.JPG)

> Figure 5: Manipulating latent codes on SVHN: In (a), we show that one of the continuous codes captures variation in lighting even though in the dataset each digit is only present with one lighting condition; In (b), one of the categorical codes is shown to control the context of central digit: for example in the 2nd column, a digit 9 is (partially) present on the right whereas in 3rd column, a digit 0 is present, which indicates that InfoGAN has learned to separate central digit from its context.
>> 그림 5: SVHN의 잠재 코드 조작: (a)에서, 우리는 데이터 세트에 각 숫자가 하나의 조명 조건만으로 존재하더라도 연속 코드 중 하나가 조명의 변화를 포착한다는 것을 보여준다. (b)에서 범주 코드 중 하나가 중앙 숫자의 컨텍스트를 제어하는 것으로 나타난다. 예를 들어, 두 번째 열에서는 숫자 9가 (부분적)이다. 세 번째 열에는 숫자 0이 있는데, 이는 InfoGAN이 가운데 숫자를 컨텍스트에서 분리하는 방법을 학습했음을 나타냅니다.

![Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Figure-6.JPG)

> Figure 6: Manipulating latent codes on CelebA: (a) shows that a categorical code can capture the azimuth of face by discretizing this variation of continuous nature; in (b) a subset of the categorical code is devoted to signal the presence of glasses; (c) shows variation in hair style, roughly ordered from less hair to more hair; (d) shows change in emotion, roughly ordered from stern to happy.
>> 그림 6: CelebA의 잠재 코드 조작: (a) 범주형 코드가 연속적인 성질의 이변화를 이산화함으로써 얼굴의 방위각을 포착할 수 있음을 보여준다. (b) 범주형 코드의 부분 집합은 안경의 존재를 알리기 위해 사용된다. (c) 머리카락이 적은 것에서 머리카락으로 대략적으로 정렬된 헤어스타일의 변화를 보여준다. (d) 변화 i를 보여준다.엄한 것에서 행복한 것으로 대강 정렬된 감정.

$A\;Proof\;of\;Lemma\;5.1$

> **Lemma A.1** For random variables $X, Y$ and function $f(x, y)$ under suitable regularity conditions:
>> **레마 A.1*** 적절한 규칙성 조건에서 랜덤 변수 $X, Y$ 및 함수 $f(x, y)$의 경우:

$\mathbb{E}_{x∼X,y∼Y|x}[f(x, y)] = \mathbb{E}_{x∼X,y∼Y |x,x'∼X|y}[f(x', y)]$.

**Proof**

$$\mathbb{E}_{x∼X,y∼Y|x}[f(x, y)] = \int_{x}P(x)\int_{y}P(y|x)f(x,y)dydx\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$
$$=\int_{x}\int_{y}P(x,y)f(x,y)dydx\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$
$$=\int_{x}\int_{y}P(x,y)f(x,y)\int_{x'}P(x'|y)f(x',y)dx'dydx$$
$$=\int_{x}P(x)\int_{y}P(y|x)\int_{x'}P(x'|y)f(x',y)dx'dydx\;\;\;$$
$$=\mathbb{E}_{x∼X,y∼Y |x,x'∼X|y}[f(x', y)]\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$B\;Interpretation\;as\;“Sleep-Sleep”\;Algorithm$

> We note that InfoGAN can be viewed as a Helmholtz machine [1]: $P_{G}(x|c)$ is the generative distribution and $Q(c|x)$ is the recognition distribution. Wake-Sleep algorithm [2] was proposed to train Helmholtz machines by performing “wake” phase and “sleep” phase updates.
>> InfoGAN은 헬름홀츠 머신[1]으로 볼 수 있다. $P_{G}(x|c)$는 생성 분포이고 $Q(c|x)$는 인식 분포이다. 웨이크-슬립 알고리즘[2]은 "웨이크" 단계와 "슬립" 단계 업데이트를 수행하여 헬름홀츠 기계를 훈련시키기 위해 제안되었다.

> The “wake” phase update proceeds by optimizing the variational lower bound of $\log P_{G}(x)$ w.r.t. 
>> "웨이크" 단계 업데이트는 $\log P_{G}(x)$ w.r.t의 변동 하한을 최적화하여 진행한다.

generator:

$$\underset{G}{\max}\mathbb{E}_{x\sim{Data,c\sim{Q(c|x)}}}[\log{P_{G}(x|c)}]$$

> The “sleep” phase updates the auxiliary distribution $Q$ by “dreaming” up samples from current generator distribution rather than drawing from real data distribution:
>> "sleep" 단계는 실제 데이터 분포에서 추출하는 대신 현재 생성기 분포에서 샘플을 "드림업"하여 보조 분포 $Q$를 업데이트합니다.

$$\underset{Q}{\max}\mathbb{E}_{x\sim{P(c),x\sim{P_{G}(x|c)}}}[\log{Q(c|x)}]$$

> Hence we can see that when we optimize the surrogate loss $L_{I}$ w.r.t. Q, the update step is exactly the “sleep” phase update in Wake-Sleep algorithm. InfoGAN differs from Wake-Sleep when we optimize $L_{I}$ w.r.t. G, encouraging the generator network $G$ to make use of latent codes $c$ for the whole prior distribution on latent codes $P(c)$. Since InfoGAN also updates generator in “sleep” phase, our method can be interpreted as “Sleep-Sleep” algorithm. This interpretation highlights InfoGAN’s difference from previous generative modeling techniques: the generator is explicitly encouraged to convey information in latent codes and suggests that the same principle can be applied to other generative models.
>> 따라서 대리 손실 $L_{I}$ w.r.t.Q를 최적화할 때 업데이트 단계는 Wake-Sleep 알고리듬에서 정확히 "sleep" 단계 업데이트임을 알 수 있다. InfoGAN은 $L_{I}$ w.r.t.G를 최적화할 때 Wake-Sleep과 달라서, 발전기 네트워크 $G$가 잠재 코드 $P(c)$에 대한 전체 사전 분포에 대해 잠재 코드 $c$를 사용하도록 권장한다. InfoGAN은 또한 "sleep" 단계에서 제너레이터를 업데이트하므로, 우리의 방법은 "sleep-sleep" 알고리듬으로 해석될 수 있다. 이 해석은 InfoGAN의 이전 생성 모델링 기법과의 차이를 강조한다. 생성기는 잠재 코드로 정보를 전달하도록 명시적으로 권장되며 동일한 원리를 다른 생성 모델에도 적용할 수 있음을 시사한다.

$C\;Experiment\;Setup$

> For all experiments, we use Adam [3] for online optimization and apply batch normalization [4] after most layers, the details of which are specified for each experiment. We use an up-convolutional architecture for the generator networks [5]. We use leaky rectified linear units (lRELU) [6] with leaky rate 0.1 as the nonlinearity applied to hidden layers of the discrminator networks, and normal rectified linear units (RELU) for the generator networks. Unless noted otherwise, learning rate is 2e-4 for $D$ and 1e-3 for G; $\lambda$ is set to 1.
>> 모든 실험에 대해 온라인 최적화를 위해 Adam [3]을 사용하고 각 실험에 대해 세부 정보가 지정된 대부분의 레이어 후에 배치 정규화 [4]를 적용한다. 우리는 발전기 네트워크에 업컨볼루션 아키텍처를 사용한다[5]. 우리는 판별기 네트워크의 숨겨진 레이어에 적용되는 비선형성으로 누출율 0.1의 누출 정류 선형 단위(lRELU)[6]와 발전기 네트워크의 일반 정류 선형 단위(RELU)를 사용한다. 달리 언급되지 않는 한 학습률은 $D$의 경우 2e-4, G의 경우 1e-3이다. $\lambda$는 1로 설정된다.

> For discrete latent codes, we apply a softmax nonlinearity over the corresponding units in the recognition network output. For continuous latent codes, we parameterize the approximate posterior through a diagonal Gaussian distribution, and the recognition network outputs its mean and standard deviation, where the standard deviation is parameterized through an exponential transformation of the network output to ensure positivity.
>> 이산 잠재 코드의 경우 인식 네트워크 출력의 해당 장치에 대해 소프트맥스 비선형성을 적용한다. 연속 잠재 코드의 경우, 우리는 대각선 가우스 분포를 통해 대략적인 사후를 매개 변수화하고, 인식 네트워크는 평균과 표준 편차를 출력하며, 여기서 표준 편차는 네트워크 출력의 지수 변환을 통해 매개 변수화하여 긍정성을 보장한다.

> The details for each set of experiments are presented below.
>> 각 실험 세트에 대한 자세한 내용은 아래에 나와 있습니다.

$C.1\;MNIST$

![Table 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Table-1.JPG)

> The network architectures are shown in Table 1. The discriminator $D$ and the recognition network $Q$ shares most of the network. For this task, we use 1 ten-dimensional categorical code, 2 continuous latent codes and 62 noise variables, resulting in a concatenated dimension of 74.
>> 네트워크 아키텍처는 표 1에 나와 있습니다. 판별기 $D$와 인식 네트워크 $Q$는 네트워크의 대부분을 공유한다. 이 작업을 위해 1개의 10차원 범주 코드, 2개의 연속 잠재 코드 및 62개의 노이즈 변수를 사용하여 74개의 연결된 차원을 얻는다.

$C.2\;SVHN$

![Table 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Table-2.JPG)

> The network architectures are shown in Table 2. The discriminator $D$ and the recognition network $Q$ shares most of the network. For this task, we use 4 ten-dimensional categorical code, 4 continuous latent codes and 124 noise variables, resulting in a concatenated dimension of 168.
>> 그 네트워크 아키텍처들 표 2에서 보여 준다.그 판별기달러 D$과 인식이 네트워크 $ Q$의 네트워크의 대부분.이 과제로 우리는 4ten-dimensional 범주 부호, 4연속 잠재적인 코드와 124소음 변수, 168의 연계된 차원에서 결과를 사용한다.

$C.3\;CelebA$

![Table 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Table-3.JPG)

> The network architectures are shown in Table 3. The discriminator $D$ and the recognition network $Q$ shares most of the network. For this task, we use 10 ten-dimensional categorical code and 128 noise variables, resulting in a concatenated dimension of 228.
>> 네트워크 아키텍처는 표 3에 나와 있습니다. 판별기 $D$와 인식 네트워크 $Q$는 네트워크의 대부분을 공유한다. 이 작업을 위해 10개의 10차원 범주 코드와 128개의 노이즈 변수를 사용하여 228개의 연결된 차원을 얻는다.

$C.4\;Faces$

![Table 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Table-4.JPG)

> The network architectures are shown in Table 4. The discriminator $D$ and the recognition network $Q$ shares the same network, and only have separate output units at the last layer. For this task, we use 5 continuous latent codes and 128 noise variables, so the input to the generator has dimension 133.
>> 네트워크 아키텍처는 표 4에 나와 있습니다. 판별기 $D$와 인식 네트워크 $Q$는 동일한 네트워크를 공유하며, 마지막 계층에서만 별도의 출력 단위를 갖는다. 이 작업을 위해 우리는 5개의 연속 잠재 코드와 128개의 노이즈 변수를 사용하므로 발전기에 대한 입력은 133차원을 갖는다.

![Table 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Table-5.JPG)

> We used separate configurations for each learned variation, shown in Table 5.
>> 표 5에 표시된 것처럼 각 학습된 변형에 대해 별도의 구성을 사용했습니다.

$C.5\;Chairs$

![Table 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Table-6.JPG)

> The network architectures are shown in Table 6. The discriminator $D$ and the recognition network $Q$ shares the same network, and only have separate output units at the last layer. For this task, we use 1 continuous latent code, 3 discrete latent codes (each with dimension 20), and 128 noise variables, so the input to the generator has dimension 189.
>> 네트워크 아키텍처는 표 6에 나와 있습니다. 판별기 $D$와 인식 네트워크 $Q$는 동일한 네트워크를 공유하며, 마지막 계층에서만 별도의 출력 단위를 갖는다. 이 작업을 위해, 우리는 1개의 연속 잠재 코드, 3개의 이산 잠재 코드(각각 차원이 20) 및 128개의 노이즈 변수를 사용하므로, 생성기에 대한 입력은 차원이 189이다.

![Table 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)InfoGAN-Interpretable-Representation-Learning-by-Information-Maximizing-Generative-Adversarial-Nets-translation/Table-7.JPG)

> We used separate configurations for each learned variation, shown in Table 7. For this task, we found it necessary to use different regularization coefficients for the continuous and discrete latent codes.
>> 표 7에 표시된 것처럼 각 학습된 변형에 대해 별도의 구성을 사용했습니다. 이 작업을 위해 연속 및 이산 잠재 코드에 대해 서로 다른 정규화 계수를 사용할 필요가 있음을 발견했다.

