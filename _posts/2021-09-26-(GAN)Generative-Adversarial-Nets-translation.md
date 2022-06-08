---
layout: post
title: "(GAN)Generative Adversarial Nets Translation"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2. Deep Learning, 1.2.2.5. GAN, 1.7. Literature Review]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

$$Generative\;Adversarial\;Nets$$

$Abstract$

> We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model $G$ that captures the data distribution, and a discriminative model $D$ that estimates the probability that a sample came from the training data rather than $G$.
>> 우리는 데이터 분포를 측적하는 생성 모델 $G$ 와 $G$ 가 아닌 샘플에서 얻게되는 훈련 데이터의 확률을 추정하는 식별 모델 $D$를 동시에 훈련함으로서 적대적 과정을 통해 생성 모델을 추정하는 새로운 프레임워크를 제시한다.

>The training procedure for $G$ is to maximize the probability of $D$ making a mistake.
>> $G$ 의 훈련과정은 $D$ 가 실수를 야기할 확률을 최대로 한다.

> This framework corresonds to a minimax two-player game.
>> 이 프레임워크는 두명에서하는 minimax 게임이다.

> In the space of arbitrary functions $G$ and $D$ , a unique solution exists, with $G$ recovering the training data distribution and $D$ equal to $ \frac{1}{2} $ everywhere.
>> 임의의 함수 $G$ 와 $D$ 의 공간에는 훈련 데이터 분포를 회복시키는 $G$ 와 항상 $ \frac{1}{2} $ 이 되는 $D$ 로써 고유한 해결책이 존재한다. 

> In the case where $G$ and $D$ are defined by multilayer perceptrons, the entire system can be trained with backpropagation.
>> $G$ 와 $D$ 가 다층 퍼셉트론으로 정의되는 경우, 전체 시스템에서 역전파로 훈련된다.

> There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples.
>> 훈련 또는 샘플 생성 중 Markov 연쇄 혹은 근사 추론 네트워크를 전개할 필요가 없다.

> Experiments demonstrate the potential of the framework through qualitative and  quantitative evaluation of the generated samples.
>> 실험을 증거로 생성 샘플에 대한 양질적인 평가를 통해 프레임워크의 잠재력을 보여줄 수 있다.

---

$1\;Introduction$

> The promise of deep learning is to discover rich, hierarchical models that represent probability distributions over the kinds of data encountered in artificial intelligence applications, such as natural images, audio waveforms containing speech, and symbols in natural language corpora.
>> 딥러닝의 약속은 자연 이미지, 음성을 포함한 오디오 파형, 자연어 말뭉치의 기호와 같은 인공지능 어플리케이션이 접하고 있는 데이터의 종류에 대한 확률 분포를 나타내는 풍부하고 계층적 모델을 발경하는 것이다.

> So far, the most striking successes in deep learning have involved discriminative models, usally those that map a high-dimensional, rich sensory input to a class label.
>> 지금까지 딥러닝의 두드러지는 성공은 차별적 모델로서, 보통 고차원적이고, 풍부한 감각의 입력을 클래스 레이블에 매핑하는 모델이었다.

> These striking successes have primarily been based on the backpropagation and dropout algorithms, using piecewise linear units which have a particularly well-behaved gradient.
>> 이 두드러지는 성공은 대체로 역전파와 dropout 알고리즘이 기반이 되며, 특히 잘 작동하는 기울기를 가지는 조각별 선형(piecewise linear) 단위를 사용한다.

> Deep generative models have had less of an impact, due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context.
>> 심층 생성 모델은 최대 가능성 추정 및 관련 전략에서 발생하는 많은 다루기 어려운 확률적 계산을 근사화하는 어려움과 생성 맥락에서 부분 선형 단위의 이점을 활용하는 어려움으로 인해 영향을 덜 받았다.

> We propose a new generative model estimation procedure that sidesteps these difficulties.
>> 우리는 이러한 어려움을 회피하는 새로운 생성 모델 추정 절차를 제안한다.

> In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution. The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles. This framework can yield specific training algorithms for many kinds of model and optimization algorithm. In this article, we explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron, and the discriminative model is also a multilayer perceptron. We refer to this special case as adversarial nets. In this case, we can train both models using only the highly successful backpropagation and dropout algorithms [16] and sample from the generative model using only forward propagation. No approximate inference or Markov chains are necessary.
>> 제안된 적대적 네트 프레임워크에서 생성 모델은 샘플이 모델 분포에서 나온 것인지 또는 데이터 분포에서 나온 것인지를 결정하는 것을 배우는 차별적 모델인 적과 겨루게 된다. 생성 모델은 위조지폐를 만들어 적발하지 않고 사용하려 하는 위조지폐 팀과 유사하다고 볼 수 있고, 차별 모델은 경찰과 유사해 위조지폐를 적발하려 한다. 이 게임의 경쟁은 두 팀 모두 가짜가 진짜와 구별할 수 없을 때까지 그들의 방법을 개선하도록 한다. 이 프레임워크는 많은 종류의 모델 및 최적화 알고리듬에 대한 특정 훈련 알고리듬을 산출할 수 있다. 본 논문에서는 생성 모델이 다층 퍼셉트론을 통해 무작위 노이즈를 전달하여 샘플을 생성하는 특별한 경우를 살펴보고, 차별 모델도 다층 퍼셉트론이다. 우리는 이 특별한 경우를 적대적 네트라고 부른다. 이 경우, 우리는 매우 성공적인 역전파 및 드롭아웃 알고리듬[16]만을 사용하여 두 모델과 전방 전파만을 사용하여 생성 모델의 샘플을 모두 훈련할 수 있다. 근사적인 추론이나 마르코프 연쇄는 필요하지 않다.

$2\;Related\;work$

> Until recently, most work on deep generative models focused on models that provided a parametric specification of a probability distribution function. The model can then be trained by maximizing the log likelihood. In this family of model, perhaps the most succesful is the deep Boltzmann machine [25]. Such models generally have intractable likelihood functions and therefore require numerous approximations to the likelihood gradient. These difficulties motivated the development of “generative machines”–models that do not explicitly represent the likelihood, yet are able to generate samples from the desired distribution. Generative stochastic networks [4] are an example of a generative machine that can be trained with exact backpropagation rather than the numerous approximations required for Boltzmann machines. This work extends the idea of a generative machine by eliminating the Markov chains used in generative stochastic networks.
>> 최근까지, 대부분의 사람들은 확률 분포 함수의 매개 변수 사양을 제공하는 모델에 중점을 두고 심층 생성 모델을 연구한다. 그런 다음 로그 가능성을 최대화하여 모델을 훈련시킬 수 있다. 이 모델 제품군에서, 아마도 가장 성공적인 것은 깊은 볼츠만 기계일 것입니다 [25]. 이러한 모델은 일반적으로 다루기 어려운 우도 함수를 가지므로 우도 기울기에 대한 수많은 근사치가 필요하다. 이러한 어려움은 "생성 기계"의 개발을 자극했는데, 이 모델은 가능성을 명시적으로 나타내지 않지만 원하는 분포로부터 샘플을 생성할 수 있다. 생성 확률적 네트워크[4]는 볼츠만 기계에 필요한 수많은 근사치가 아닌 정확한 역 전파로 훈련될 수 있는 생성 기계의 예이다. 이 작업은 생성 확률 네트워크에 사용되는 마르코프 체인을 제거하여 생성 기계의 아이디어를 확장한다.

> Our work backpropagates derivatives through generative processes by using the observation that
>> 우리의 연구는 다음과 같은 관찰을 사용하여 생성 과정을 통해 파생물을 역전파한다.

$$\lim_{\sigma\to 0}\triangledown_{x}E_{\epsilon\sim N(0,\sigma^{2}I)}f(x+\epsilon) = \triangledown_{x}f(x).$$

> We were unaware at the time we developed this work that Kingma and Welling [26] and Rezende et al. [23] had developed more general stochastic backpropagation rules, allowing one to backpropagate through Gaussian distributions with finite variance, and to backpropagate to the covariance parameter as well as the mean. These backpropagation rules could allow one to learn the conditional variance of the generator, which we treated as a hyperparameter in this work. Kingma and Welling [18] and Rezende et al. [23] use stochastic backpropagation to train variational autoencoders (VAEs). Like generative adversarial networks, variational autoencoders pair a differentiable generator network with a second neural network. Unlike generative adversarial networks, the second network in a VAE is a recognition model that performs approximate inference. GANs require differentiation through the visible units, and thus cannot model discrete data, while VAEs require differentiation through the hidden units, and thus cannot have discrete latent  variables. Other VAElike approaches exist [12, 22] but are less closely related to our method.
>> Kingma와 Welling[26]과 Rezende 등이 이 연구를 개발할 당시 우리는 알지 못했습니다. [23] 보다 일반적인 확률적 역전파 규칙을 개발하여 유한 분산으로 가우스 분포를 통해 역전파하고 평균뿐만 아니라 공분산 매개 변수로 역전파할 수 있었다. 이러한 역 전파 규칙을 통해 발전기의 조건부 분산을 학습할 수 있으며, 우리는 이를 본 연구에서 하이퍼 매개 변수로 취급했다. 킹마와 웰링[18]과 레젠디 외 [23] 확률적 역 전파를 사용하여 가변 자동 인코더(VAE)를 훈련시킨다. 생성적 적대 네트워크와 마찬가지로, 변형 자동 인코더는 미분 가능한 생성기 네트워크를 두 번째 신경망과 쌍을 이룬다. 생성적 적대 네트워크와 달리, VAE의 두 번째 네트워크는 대략적인 추론을 수행하는 인식 모델이다. GAN은 가시적 단위를 통한 미분이 필요하며, 따라서 이산 데이터를 모델링할 수 없는 반면, VAE는 숨겨진 단위를 통한 미분이 필요하므로 이산 잠재 변수를 가질 수 없다. 다른 VAE와 유사한 접근법이 [12, 22] 존재하지만 우리의 방법과 덜 밀접하게 관련되어 있다.

> Previous work has also taken the approach of using a discriminative criterion to train a generative model [29, 13]. These approaches use criteria that are intractable for deep generative models. These methods are difficult even to approximate for deep models because they involve ratios of probabilities which cannot be approximated using variational approximations that lower bound the probability. Noise-contrastive estimation (NCE) [13] involves training a generative model by learning the weights that make the model useful for discriminating data from a fixed noise distribution. Using a previously trained model as the noise distribution allows training a sequence of models of increasing quality. This can be seen as an informal competition mechanism similar in spirit to the formal competition used in the adversarial networks game. The key limitation of NCE is that its “discriminator” is defined by the ratio of the probability densities of the noise distribution and the model distribution, and thus requires the ability to evaluate and backpropagate through both densities.
>> Kingma와 Welling[18]과 Rezende 등이 이 연구를 개발할 당시 우리는 알지 못했습니다. [23] 보다 일반적인 확률적 역전파 규칙을 개발하여 유한 분산으로 가우스 분포를 통해 역전파하고 평균뿐만 아니라 공분산 매개 변수로 역전파할 수 있었다. 이러한 역 전파 규칙을 통해 발전기의 조건부 분산을 학습할 수 있으며, 우리는 이를 본 연구에서 하이퍼 매개 변수로 취급했다. 킹마와 웰링[18]과 레젠디 외 [23] 확률적 역 전파를 사용하여 가변 자동 인코더(VAE)를 훈련시킨다. 생성적 적대 네트워크와 마찬가지로, 변형 자동 인코더는 미분 가능한 생성기 네트워크를 두 번째 신경망과 쌍을 이룬다. 생성적 적대 네트워크와 달리, VAE의 두 번째 네트워크는 대략적인 추론을 수행하는 인식 모델이다. GAN은 가시적 단위를 통한 미분이 필요하며, 따라서 이산 데이터를 모델링할 수 없는 반면, VAE는 숨겨진 단위를 통한 미분이 필요하므로 이산 잠재 변수를 가질 수 없다. 다른 VAE와 유사한 접근법이 [12, 22] 존재하지만 우리의 방법과 덜 밀접하게 관련되어 있다.

> Some previous work has used the general concept of having two neural networks compete. The most relevant work is predictability minimization [26]. In predictability minimization, each hidden unit in a neural network is trained to be different from the output of a second network, which predicts the value of that hidden unit given the value of all of the other hidden units. This work differs from predictability minimization in three important ways: 1) in this work, the competition between the networks is the sole training criterion, and is sufficient on its own to train the network. Predictability minimization is only a regularizer that encourages the hidden units of a neural network to be statistically independent while they accomplish some other task; it is not a primary training criterion. 2) The nature of the competition is different. In predictability minimization, two networks’ outputs are compared, with one network trying to make the outputs similar and the other trying to make the outputs different. The output in question is a single scalar. In GANs, one network produces a rich, high dimensional vector that is used as the input to another network, and attempts to choose an input that the other network does not know how to process. 3) The specification of the learning process is different. Predictability minimization is described as an optimization problem with an objective function to be minimized, and learning approaches the minimum of the objective function. GANs are based on a minimax game rather than an optimization problem, and have a value function that one agent seeks to maximize and the other seeks to minimize. The game terminates at a saddle point that is a minimum with respect to one player’s strategy and a maximum with respect to the other player’s strategy.
>> 일부 이전 연구에서는 두 개의 신경망이 경쟁하는 일반적인 개념을 사용했다. 가장 관련성이 높은 작업은 예측 가능성 최소화입니다 [26]. 예측 가능성 최소화에서, 신경망의 각 숨겨진 유닛은 다른 모든 숨겨진 유닛의 값이 주어지면 해당 숨겨진 유닛의 값을 예측하는 두 번째 네트워크의 출력과 다르게 훈련된다. 이 작업은 예측 가능성 최소화와는 세 가지 중요한 면에서 다르다. 1) 본 연구에서는 네트워크 간의 경쟁이 유일한 훈련 기준이며, 네트워크를 훈련시키기에 충분하다. 예측 가능성 최소화는 신경망의 숨겨진 단위가 다른 작업을 수행하는 동안 통계적으로 독립하도록 장려하는 정규화일 뿐이다. 이는 기본 훈련 기준이 아니다. 2) 경쟁의 특성은 다르다. 예측 가능성 최소화에서 두 네트워크의 출력을 비교하는데, 한 네트워크는 출력을 비슷하게 만들려고 하고 다른 네트워크는 출력을 다르게 만들려고 한다. 문제의 출력은 단일 스칼라입니다. GAN에서, 한 네트워크는 다른 네트워크에 대한 입력으로 사용되는 풍부한 고차원 벡터를 생성하고, 다른 네트워크가 처리하는 방법을 모르는 입력을 선택하려고 시도한다. 3) 학습 프로세스의 사양은 다르다. 예측 가능성 최소화는 최소화할 목적 함수가 있는 최적화 문제로 설명되며 학습은 목적 함수의 최소값에 근접한다. GAN은 최적화 문제가 아닌 미니맥스 게임에 기반을 두고 있으며, 한 에이전트가 최대화를 꾀하고 다른 에이전트가 최소화를 추구하는 가치 함수를 가지고 있다. 게임은 한 플레이어의 전략과 관련하여 최소, 다른 플레이어의 전략과 관련하여 최대인 안장 지점에서 종료된다.

> Generative adversarial networks has been sometimes confused with the related concept of “adversarial examples” [28]. Adversarial examples are examples found by using gradient-based optimization directly on the input to a classification network, in order to find examples that are similar to the data yet misclassified. This is different from the present work because adversarial examples are not a mechanism for training a generative model. Instead, adversarial examples are primarily an analysis tool for showing that neural networks behave in intriguing ways, often confidently classifying two images differently with high confidence even though the difference between them is imperceptible to a human observer. The existence of such adversarial examples does suggest that generative adversarial network training could be inefficient, because they show that it is possible to make modern discriminative networks confidently recognize a class without emulating any of the human-perceptible attributes of that class.
>> 생성적 적대 네트워크는 때때로 "적대적 사례"의 관련 개념과 혼동된다[28]. 적대적 예는 데이터와 유사하지만 잘못 분류된 예를 찾기 위해 분류 네트워크에 대한 입력에 직접 그레이디언트 기반 최적화를 사용하여 찾은 예제이다. 이는 적대적 사례가 생성 모델을 훈련시키기 위한 메커니즘이 아니기 때문에 현재 작업과 다르다. 대신, 적대적 예는 신경망이 흥미로운 방식으로 작동한다는 것을 보여주는 일차적인 분석 도구이며, 종종 두 이미지 간의 차이가 인간 관찰자에게 감지되지 않더라도 높은 신뢰도로 자신 있게 두 이미지를 다르게 분류한다. 이러한 적대적 사례의 존재는 생성적 적대적 네트워크 훈련이 비효율적일 수 있음을 시사하는데, 이는 현대의 차별적 네트워크가 그 클래스의 인간이 인식할 수 있는 속성을 모방하지 않고 자신 있게 클래스를 인식하도록 만드는 것이 가능하다는 것을 보여주기 때문이다.

$3\;Adversarial\;nets$

> The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons. To learn the generator’s distribution $p_{g}$ over data $x$, we define a prior on input noise variables $p_{z}(z)$, then represent a mapping to data space as $G(z; \theta_{g})$, where $G$ is a differentiable function represented by a multilayer perceptron with parameters $\theta_{g}$. We also define a second multilayer perceptron $D(x; \theta_{d})$ that outputs a single scalar. $D(x)$ represents the probability that $x$ came from the data rather than $p_{g}$. We train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$. We simultaneously train $G$ to minimize $\log{(1 − D(G(z)))}$. In other words, $D$ and $G$ play the following two-player minimax game with value function $V (G, D)$:
>> 적대적 모델링 프레임워크는 모델이 모두 다층 퍼셉트론일 때 가장 쉽게 적용할 수 있다. 데이터 $x$에 대한 생성기 분포 $p_{g}$를 학습하기 위해, 우리는 입력 노이즈 변수 $p_{z}(z)$에 대한 선행 값을 정의한 다음, 데이터 공간에 대한 매핑을 $G(z; \theta_{g})$로 표현한다. 여기서 $G$는 매개 변수 $\theta_{g}$를 가진 다층 퍼셉트론으로 표현되는 미분 함수이다. 우리는 또한 단일 스칼라를 출력하는 두 번째 다층 퍼셉트론 $D(x; \theta_{d})$를 정의한다. $D(x)$는 $x$가 $p_{g}$가 아닌 데이터에서 나왔을 확률을 나타낸다. $G$의 훈련 예시와 샘플 모두에 올바른 레이블을 할당할 확률을 최대화하기 위해 $D$를 훈련한다. 우리는 동시에 $G$를 훈련시켜 $\log{(1 - D(G(z))}$를 최소화한다. 즉, $D$와 $G$는 가치 함수 $V(G, D)$로 다음과 같은 2인용 미니맥스 게임을 한다.

$$\underset{G}{\min}\underset{D}{\max}V(D,G)=E_{x\sim p_{data}(x)}[\log{D(x)}]+E_{z\sim p_z(z)}[\log{(1-D(G(z)))}].$$

> In the next section, we present a theoretical analysis of adversarial nets, essentially showing that the training criterion allows one to recover the data generating distribution as $G$ and $D$ are given enough capacity, i.e., in the non-parametric limit. See Figure 1 for a less formal, more pedagogical explanation of the approach. In practice, we must implement the game using an iterative, numerical approach. Optimizing $D$ to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting. Instead, we alternate between k steps of optimizing $D$ and one step of optimizing $G$. This results in $D$ being maintained near its optimal solution, so long as $G$ changes slowly enough. The procedure is formally presented in Algorithm 1. In practice, equation 1 may not provide sufficient gradient for $G$ to learn well. Early in learning, when $G$ is poor, $D$ can reject samples with high confidence because they are clearly different from the training data. In this case, $\log{(1 − D(G(z)))}$ saturates. Rather than training $G$ to minimize $\log{(1 − D(G(z)))}$ we can train $G$ to maximize $\log{D(G(z))}$. This objective function results in the same fixed point of the dynamics of $G$ and $D$ but provides much stronger gradients early in learning.
>> 다음 섹션에서는 적대적 네트에 대한 이론적 분석을 제시하며, 기본적으로 훈련 기준이 $G$ 및 $D$에 충분한 용량이 주어짐에 따라 데이터 생성 분포를 복구할 수 있음을 보여준다. 즉, 비모수 한계에서. 접근 방식에 대한 덜 형식적이고 교육학적인 설명은 그림 1을 참조하십시오. 실제로, 우리는 반복적이고 수치적인 접근법을 사용하여 게임을 구현해야 한다. 훈련의 내부 루프에서 $D$를 완료하기 위해 최적화하는 것은 계산적으로 금지되며, 유한한 데이터 세트에서 과적합이 발생할 수 있다. 대신 $D$를 최적화하는 k단계와 $G$를 최적화하는 한 단계를 번갈아 실행한다. 이로 인해 $G$가 충분히 느리게 변경되는 한 $D$는 최적의 솔루션 근처에 유지된다. 그 절차는 알고리즘 1에 공식적으로 제시되어 있다. 실제로 방정식 1은 $G$가 잘 학습하기에 충분한 기울기를 제공하지 않을 수 있다. 학습 초기에 $G$가 부족할 때 $D$는 훈련 데이터와 분명히 다르기 때문에 높은 신뢰도로 샘플을 거부할 수 있다. 이 경우 $\log{(1 - D(G(z))}$가 포화된다. $G$를 훈련시켜 $\log{(1 - D(z))}$를 최소화하는 대신 $G$를 훈련시켜 $\log{D(z)}$를 최대화할 수 있다. 이 목적 함수는 $G$와 $D$의 역학의 동일한 고정점을 가져오지만 학습 초기에 훨씬 더 강력한 그레이디언트를 제공한다.

$4\;Theoretical\;Results$

> The generator $G$ implicitly defines a probability distribution $p_{g}$ as the distribution of the samples $G(z)$ obtained when $z\sim p_{z}$. Therefore, we would like Algorithm 1 to converge to a good estimator of pdata, if given enough capacity and training time. The results of this section are done in a nonparametric setting, e.g. we represent a model with infinite capacity by studying convergence in the space of probability density functions.
>> 생성기 $G$는 확률 분포 $p_{g}$를 $z\sim p_{z}$일 때 얻은 샘플 $G(z)$의 분포로 암시적으로 정의한다. 따라서 충분한 용량과 훈련 시간이 주어진다면 알고리듬 1이 좋은 pdata 추정기로 수렴되기를 바란다. 이 섹션의 결과는 비모수 설정에서 수행된다. 예를 들어, 우리는 확률 밀도 함수의 공간에서 수렴을 연구하여 무한한 용량을 가진 모델을 나타낸다.

> We will show in section 4.1 that this minimax game has a global optimum for $p_{g} = p_{data}$. We will then show in section 4.2 that Algorithm 1 optimizes Eq 1, thus obtaining the desired result.
>> 섹션 4.1에서 이 미니맥스 게임이 $p_{g} = p_{data}$에 대한 전역 최적임을 보여줄 것이다. 그런 다음 섹션 4.2에서 알고리즘 1이 Eq 1을 최적화하여 원하는 결과를 얻는다는 것을 보여줄 것이다.

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2021-09-26-(GAN)Generative-Adversarial-Nets-translation/Figure-1.JPG)

> Figure 1: Generative adversarial nets are trained by simultaneously updating the discriminative distribution ($D$, blue, dashed line) so that it discriminates between samples from the data generating distribution (black, dotted line) $p_{x}$ from those of the generative distribution $p_{g}(G)$ (green, solid line). The lower horizontal line is the domain from which $z$ is sampled, in this case uniformly. The horizontal line above is part of the domain of $x$. The upward arrows show how the mapping $x = G(z)$ imposes the non-uniform distribution $p_{g}$ on transformed samples. $G$ contracts in regions of high density and expands in regions of low density of $p_{g}$. (a) Consider an adversarial pair near convergence: $p_{g}$ is similar to pdata and $D$ is a partially accurate classifier. (b) In the inner loop of the algorithm $D$ is trained to discriminate samples from data, converging to $D\ast{}(x) = \frac{pdata(x)}{pdata(x)+p_{g}(x)}$. (c) After an update to $G$, gradient of $D$ has guided $G(z)$ to flow to regions that are more likely to be classified as data. (d) After several steps of training, if $G$ and $D$ have enough capacity, they will reach a point at which both cannot improve because $p_{g} = p_{data}$. The discriminator is unable to differentiate between the two distributions, i.e. $D(x)=\frac{1}{2}$ .
>> 그림 1: 생성 적대적 네트워크는 데이터 생성 분포(검은색, 점선) $p_{g}(G)$(녹색, 실선)의 샘플과 데이터 생성 분포(검은색, 점선) $p_{x}$의 샘플을 구별하도록 차별적 분포($D$, 파란색, 점선)를 동시에 업데이트하여 훈련된다. 아래쪽 수평선은 $z$가 샘플링되는 도메인이다. 이 경우 균일하다. 위의 수평선은 $x$ 도메인의 일부입니다. 위쪽 화살표는 매핑 $x = G(z)$가 변환된 샘플에 대해 비결정 분포 $p_{g}$를 부과하는 방법을 보여준다. $G$는 고밀도 영역에서 수축하고 $p_{g}$의 저밀도 영역에서 확장된다. (a) 수렴에 가까운 적대적 쌍을 고려한다. $p_{g}$는 pdata와 유사하며 $D$는 부분적으로 정확한 분류기이다. (b) 알고리듬의 내부 루프에서 $D$는 샘플과 데이터를 구별하도록 훈련되어 $G$ 업데이트 후 $D\ast{}(x) = \frac{pdata(x)}{pdata(x)+p_{g}(x)}$.로 수렴한다. $D$의 기울기는 $G(z)$가 데이터로 분류될 가능성이 더 높은 영역으로 흐르도록 안내했다. (d) 여러 단계 훈련 후 $G$와 $D$가 충분한 용량을 가지고 있다면 $p_{g} = p_{data}$로 인해 둘 다 개선될 수 없는 지점에 도달할 것이다. 판별기는 두 분포(예: $D(x) = \frac{1}{2}$)를 구별할 수 없다.

---

> **Algorithm 1** Minibatch stochastic gradient descent training of generative adversarial nets. The number of steps to apply to the discriminator, $k$, is a hyperparameter. We used $k = 1$, the least expensive option, in our experiments.
>> **알고리즘 1** 생성적 적대 네트워크의 미니 배치 확률적 경사 하강 훈련. 판별기 $k$에 적용할 단계 수는 하이퍼 파라미터이다. 우리는 실험에서 가장 저렴한 옵션인 $k = 1$을 사용했다.

---

![Algorithm 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2021-09-26-(GAN)Generative-Adversarial-Nets-translation/Algorithm-1.JPG)

$4.1\;Global\;Optimality\;of\;p_{g}=p_{data}$

> We first consider the optimal discriminator $D$ for any given generator $G$.
>> 우리는 먼저 주어진 생성기 $G$에 대한 최적의 판별기 $D$를 고려한다.

> **Proposition 1.** For $G$ fixed, the optimal discriminator $D$ is
>> **제안 1.** 고정된 $G$의 경우, 최적의 판별기 $D$는

$$D^{*}_{G}(x)=\frac{P_{data}(x)}{P_{data}(x)+P_{g}(x)}$$

> Proof. The training criterion for the discriminator $D$, given any generator $G$, is to maximize the quantity $V(G, D)$
>> 증명. 임의의 생성기 $G$가 주어지면 판별기 $D$에 대한 훈련 기준은 $V(G, D)$의 양을 최대화하는 것이다.

$$V(G,D)=\int_{x}P_{data}(x)\log{(D(x))dx}+\int_{z}p_{z}(z)\log{(1-D(g(z)))dz}$$

$$\;\;=\int_{x}p_{data}(x)\log{(D(x))}+p_{g}(x)\log{(1-D(x))dx}$$

> For any $(a, b)\in R^{2}$ \ {0, 0}, the function $y\to a \log{(y)} + b\log{(1 − y)}$ achieves its maximum in [0, 1] at $\frac{a}{a+b}$. The discriminator does not need to be defined outside of $Supp(p_{data})\cup Supp(p_{g})$, concluding the proof.
>> 임의의 $(a, b)\in R^{2}$ \{0,0}에 대해, 함수 $y\to a \log{(y)} + b\log{(1 - y)}$는 $\frac{a}{a+b}$에서 [0,1]에서 최대값을 달성한다. 판별기는 증명을 결론짓는 $Supp(p_{data})\cup Supp(p_{g})$ 외부에서 정의할 필요가 없다.

> Note that the training objective for $D$ can be interpreted as maximizing the log-likelihood for estimating the conditional probability $P(Y = y\mid x)$, where $Y$ indicates whether $x$ comes from $p_{data}$ (with $y = 1$) or from $p_{g}$ (with $y = 0$). The minimax game in Eq. 1 can now be reformulated as:
>> $D$에 대한 훈련 목표는 조건부 확률 $P(Y = y \mid  x)$를 추정하기 위한 로그 가능성을 최대화하는 것으로 해석될 수 있다. 여기서 $Y$는 $x$가 $p_{data}$($y = 1$인 경우) 또는 $p_{g}$($y = 0$인 경우)에서 오는지를 나타낸다. 미니맥스 게임은 이제 다음과 같이 재구성될 수 있다:

$$C(G)=\underset{D}{\max}V(G,D)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$=E_{x\sim p_{data}}[\log{D^{*}_{G}(x)}] +E_{z\sim p_{z}}[\log{(1-D^{*}_{G}(G(z)))]}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$=E_{x\sim p_{data}}[\log{D^{*}_{G}(x)}] +E_{z\sim p_{z}}[\log{(1-D^{*}_{G}(x)]}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$=E_{x\sim p_{data}}[\log{\frac{P_{data}(x)}{P_{data}(x)+P_{g}(x)}}] +E_{z\sim p_{z}}[\log{\frac{p_g(x)}{P_{data}(x)+P_{g}(x)}}$$

> **Theorem 1**. The global minimum of the virtual training criterion $C(G)$ is achieved if and only if $p_{g} = p_{data}$. At that point, $C(G)$ achieves the $value − \log{4}$.
>> **정식 1**. 가상 훈련 기준 $C(G)$의 전역 최소값은 $p_{g} = p_{data}$인 경우에만 달성된다. 이 시점에서 $C(G)$는 $value - \log{4}$를 달성한다.

> Proof. For $p_{g} = p_{data}, D^{\ast{}}_{G}(x) = \frac{1}{2}$ , (consider Eq. 2). Hence, by inspecting Eq. 4 at $D^{\ast{}}_{G}(x) = \frac{1}{2}$, we find $C(G) = log{\frac{1}{2}} + log{\frac{1}{2}} = − \log{4}$. To see that this is the best possible value of $C(G)$, reached only for $p_{g} = p_{data}$, observe that
>> 증명. $p_{g} = p_{data}의 경우, D^{*}_{G}(x) = \frac{1}{2}$, (예: 2). 따라서 $D^{frac}_{G}(x) = \frac{1}{2}$에서 등식 4를 검사함으로써 $C(G) = log\to frac{1}{2} + log\to frac{1}{2} = - \log{4}$를 찾을 수 있다. $p_{g} = p_{data}$에 대해서만 도달할 수 있는 최상의 $C(G)$ 값인지 확인하려면 다음을 관찰하십시오.

$$E_{x\sim{p_{data}}}[-\log{2}]+E_{x\sim{p_{g}}}[-\log{2}]=\log{4}$$

> and that by subtracting this expression from $C(G) = V (D^{\ast{}}_{G}, G)$, we obtain:
>> 그리고 $C(G) = V (D^{*}_{G}, G)$ 에서 이 식을 빼면 다음과 같은 것을 얻을 수 있다.

$$C(G)=-\log{4}+KL(P_{data}\parallel{\frac{P_{data}+P_{g}}{2}})+KL(P_{g}\parallel{\frac{P_{data}+P_{g}}{2}})$$

>> where $KL$ is the Kullback–Leibler divergence. We recognize in the previous expression the Jensen–Shannon divergence between the model’s distribution and the data generating process:
>> 여기서 $KL$은 쿨백-라이블러 발산이다. 우리는 이전 표현식에서 모델의 분포와 데이터 생성 과정 사이의 옌센-샤논 분산을 인식한다.

$$C(G)=-\log{4}+2\cdot{JSD}(P_{data}\parallel{P_{g}})$$

> Since the Jensen–Shannon divergence between two distributions is always non-negative, and zero iff they are equal, we have shown that $C^{*} = −\log{(4)}$ is the global minimum of $C(G)$ and that the only solution is $p_{g} = p_{data}$, i.e., the generative model perfectly replicating the data distribution.
>> 두 분포 사이의 Jensen-Shannon 분기는 항상 음이 아니므로, 만약 두 분포가 같다면 0이기 때문에, 우리는 $C^{*} = -\log{(4)}$가 $C(G)$의 전역 최소값이고 유일한 해결책은 $p_{g} = p_{data}$이며, 즉, 분포를 완벽하게 복제하는 생성 모델임을 보여주었다.

$4.2\;Convergence\;of\;Algorithm\;1$

> **Proposition 2**. If $G$ and $D$ have enough capacity, and at each step of Algorithm 1, the discriminator is allowed to reach its optimum given $G$, and $p_{g}$ is updated so as to improve the criterion
>> **제안 2***. $G$와 $D$가 충분한 용량을 가지고 있고 알고리듬 1의 각 단계에서 판별기가 주어진 $G$에 최적에 도달할 수 있도록 허용되며, $p_{g}$는 기준을 개선하기 위해 업데이트된다.

$$E_{x\sim{P_{data}}}[\log{D^{*}_{G}(x)}]+E_{x\sim{p_{g}}}[\log{(1-D^{*}{G}(x))}]$$

> then $p_{g}$ converges to $p_{data}$
>> 그러면 $p_{g}$가 $p_{data}$로 수렴됩니다.

> Proof. Consider $V (G, D) = U(p_{g}, D)$ as a function of $p_{g}$ as done in the above criterion. Note that $U(p_{g}, D)$ is convex in $p_{g}$. The subderivatives of a supremum of convex functions include the derivative of the function at the point where the maximum is attained. In other words, if $f(x) = \sup_{\alpha\in{A}}f_{\alpha}(x)$ and $f_{\alpha}(x)$ is convex in $x$ for every $\alpha$, then $\partial{f_{\beta}{(x)}}\in{\partial{f}}$ if $\beta = \arg\sup_{\alpha{\in{A}}}f_{\alpha}{(x)}$. This is equivalent to computing a gradient descent update for $p_{g}$ at the optimal $D$ given the corresponding $G$. $\sup_{D}U(p_{g}, D)$ is convex in $p_{g}$ with a unique global optima as proven in Thm 1, therefore with sufficiently small updates of $p_{g}$, $p_{g}$ converges to $p_{x}$, concluding the proof.
>> 증명. 위의 기준에서와 같이 $V(G, D) = U(p_{g}, D)$를 $p_{g}$의 함수로 간주한다. $U(p_{g}, D)$는 $p_{g}$에서 볼록하다. 볼록 함수의 최상위의 하위 미분들은 최대가 달성된 지점에서 함수의 미분들을 포함한다. 다시 말해서, 만약 $f(x) = \sup_{\alpha\in{A}}f_{\alpha}(x)$와 $f_{\alpha}(x)$가 모든  $\alpha$에 대해 $x$에서 볼록하다면, $\partial{f_{\beta}{(x)}}\in{\partial{f}}$는 $\beta = \arg\sup_{\alpha{\in{A}}}f_{\alpha}{(x)}$.이다. 이는 해당 $G$가 주어진 최적의 $D$에서 $p_{g}$에 대한 그레이디언트 강하 업데이트를 계산하는 것과 같다. $\sup_{D}U(p_{g}, D)$는 Tm1에서 입증된 고유한 전역 최적과 함께 $p_{g}$에서 볼록하므로 $p_{g}$의 작은 업데이트가 ${p_x}$로 수렴된다.

> In practice, adversarial nets represent a limited family of $p_{g}$ distributions via the function $G(z; \theta_{g})$, and we optimize $\theta_{g}$ rather than $p_{g}$ itself, so the proofs do not apply. However, the excellent performance of multilayer perceptrons in practice suggests that they are a reasonable model to use despite their lack of theoretical guarantees.
>> 실제로 적대적 네트는 $G(z; \theta_{g})$ 함수를 통해 제한된 $p_{g}$ 분포군을 나타낸다. 그리고 우리는 $p_{g}$ 자체보다 $\theta_{g}$를 최적화하므로 증명은 적용되지 않는다. 그러나 실제로 다층 퍼셉트론의 우수한 성능은 이론적 보증이 없음에도 불구하고 사용하기에 합리적인 모델임을 시사한다.

![Table 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2021-09-26-(GAN)Generative-Adversarial-Nets-translation/Table-1.JPG)

> Table 1: Parzen window-based log-likelihood estimates. The reported numbers on MNIST are the mean loglikelihood of samples on test set,with the standard error of the mean computed across examples. On TFD, we computed the standard error across folds of the dataset, with a different $\sigma$ chosen using the validation set of each fold. On TFD, $\sigma$ was cross validated on each fold and mean log-likelihood on each fold were computed. For MNIST we compare against other models of the real-valued (rather than binary) version of dataset.
>> 표 1: Parzen 창 기반 로그 우도 추정치 MNIST에 보고된 숫자는 테스트 세트에서 샘플의 평균 로그 우도이며, 예에서 평균의 표준 오차는 계산된다. TFD에서, 우리는 각 폴드의 유효성 검사 세트를 사용하여 다른 $\sigma$를 선택하여 데이터 세트의 폴드에 걸친 표준 오류를 계산했다. TFD에서, $\sigma$는 각 폴드에서 교차 검증되었고 각 폴드의 평균 로그 우도를 계산했다. MNIST의 경우 데이터 세트의 실제 값(이진수 대신) 버전의 다른 모델과 비교한다.

$5\;Experiments$

> We trained adversarial nets an a range of datasets including MNIST[21], the Toronto Face Database (TFD) [27], and CIFAR-10 [19]. The generator nets used a mixture of rectifier linear activations [17, 8] and sigmoid activations, while the discriminator net used maxout [9] activations. Dropout [16] was applied in training the discriminator net. While our theoretical framework permits the use of dropout and other noise at intermediate layers of the generator, we used noise as the input to only the bottommost layer of the generator network.
>> 우리는 MNIST[21], 토론토 얼굴 데이터베이스(TFD)[27] 및 CIFAR-10[19]을 포함한 광범위한 데이터 세트를 훈련했다. 발전기 네트는 정류기 선형 활성화[17, 8]와 시그모이드 활성화의 혼합물을 사용했고 판별기 네트는 maxout [9] 활성화를 사용했다. 탈락[16]은 판별자 망 훈련에 적용되었다. 우리의 이론적 프레임워크는 발전기의 중간 계층에서 드롭아웃 및 기타 노이즈의 사용을 허용하지만, 우리는 소음을 발전기 네트워크의 맨 아래 계층에만 대한 입력으로 사용했다.

> We estimate probability of the test set data under $p_{g}$ by fitting a Gaussian Parzen window to the samples generated with $G$ and reporting the log-likelihood under this distribution. The σ parameter of the Gaussians was obtained by cross validation on the validation set. This procedure was introduced in Breuleux et al. [7] and used for various generative models for which the exact likelihood is not tractable [24, 3, 4]. Results are reported in Table 1. This method of estimating the likelihood has somewhat high variance and does not perform well in high dimensional spaces but it is the best method available to our knowledge. Advances in generative models that can sample but not estimate likelihood directly motivate further research into how to evaluate such models. In Figures 2 and 3 we show samples drawn from the generator net after training. While we make no claim that these samples are better than samples generated by existing methods, we believe that these samples are at least competitive with the better generative models in the literature and highlight the potential of the adversarial framework.
>> 우리는 가우스 파르젠 창을 $G$로 생성된 샘플에 맞추고 이 분포에서 로그 우도를 보고하여 $p_{g}$ 아래의 테스트 세트 데이터의 확률을 추정한다. 가우시스의 α 매개변수는 검증 집합에 대한 교차 검증을 통해 얻어졌다. 이 절차는 Breuleux 외에서 소개되었다. [7] 정확한 가능성이 다루기 어려운 다양한 생성 모델에 사용됩니다 [24, 3, 4]. 결과는 표 1에 보고된다. 이 가능성을 추정하는 방법은 분산이 다소 높고 고차원 공간에서 잘 수행되지는 않지만 우리가 아는 가장 좋은 방법이다. 표본을 추출할 수 있지만 가능성을 추정할 수 없는 생성 모델의 발전은 이러한 모델을 평가하는 방법에 대한 추가 연구에 직접적인 동기를 부여한다. 그림 2와 3에는 교육 후 제너레이터 네트에서 추출한 샘플이 나와 있습니다. 우리는 이러한 샘플이 기존 방법에 의해 생성된 샘플보다 낫다고 주장하지는 않지만, 이러한 샘플이 적어도 문헌의 더 나은 생성 모델과 경쟁적이며 적대적 프레임워크의 잠재력을 강조한다고 믿는다.

$6\;Advantages\;and\;disadvantages$

> This new framework comes with advantages and disadvantages relative to previous modeling frameworks. The disadvantages are primarily that there is no explicit representation of $p_{g}(x)$, and that $D$ must be synchronized well with $G$ during training (in particular,$G$ must not be trained too much without updating D, in order to avoid “the Helvetica scenario” in which $G$ collapses too many values of $z$ to the same value of $x$ to have enough diversity to model pdata), much as the negative chains of a Boltzmann machine must be kept up to date between learning steps. The advantages are that Markov chains are never needed, only backprop is used to obtain gradients, no inference is needed during learning, and a wide variety of functions can be incorporated into the model. Table 2 summarizes the comparison of generative adversarial nets with other generative modeling approaches.
>> 이 새로운 프레임워크는 이전 모델링 프레임워크에 비해 장단점이 있다. 단점은 주로 $p_{g}(x)$의 명시적 표현이 없고 훈련 중에 $D$가 $G$와 잘 동기화되어야 한다는 것이다(특히 $G$가 $z$의 너무 많은 값을 $x$의 동일한 값으로 붕괴시키는 "헬베티카 시나리오"를 피하기 위해 $G$는 D를 업데이트하지 않고 너무 많이 훈련해서는 안 된다). 모델 pdata에 대한 다양성), 학습 단계 사이에 볼츠만 기계의 음의 체인을 최신 상태로 유지해야 하는 정도입니다. 장점은 마르코프 체인이 절대 필요하지 않고, 그레이디언트를 얻기 위해 백프로프만 사용되며, 학습 중에 추론이 필요하지 않으며, 매우 다양한 기능을 모델에 통합할 수 있다는 것이다. 표 2는 다른 생성 모델링 접근법과 생성적 적대적 네트의 비교를 요약한 것이다.

> The aforementioned advantages are primarily computational. Adversarial models may also gain some statistical advantage from the generator network not being updated directly with data examples, but only with gradients flowing through the discriminator. This means that components of the input are not copied directly into the generator’s parameters. Another advantage of adversarial networks is that they can represent very sharp, even degenerate distributions, while methods based on Markov chains require that the distribution be somewhat blurry in order for the chains to be able to mix between modes.
>> 앞서 언급한 장점들은 주로 계산적이다. 적대적 모델은 또한 데이터 예제로 직접 업데이트되지 않고 판별기를 통과하는 그레이디언트만으로 발전기 네트워크에서 통계적 이점을 얻을 수 있다. 즉, 입력의 구성 요소가 제너레이터의 매개 변수에 직접 복사되지 않습니다. 적대적 네트워크의 또 다른 장점은 매우 날카롭고 심지어 퇴화된 분포를 나타낼 수 있는 반면, 마르코프 체인에 기반한 방법은 체인이 모드 간에 혼합될 수 있도록 분포가 다소 흐릿해야 한다는 것이다.

$7\;Conclusions\;and\;future\;work$

> This framework admits many straightforward extensions:
>> 이 프레임워크는 많은 간단한 확장을 허용한다:

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2021-09-26-(GAN)Generative-Adversarial-Nets-translation/Figure-2.JPG)

> Figure 2: Visualization of samples from the model. Rightmost column shows the nearest training example of the neighboring sample, in order to demonstrate that the model has not memorized the training set. Samples are fair random draws, not cherry-picked. Unlike most other visualizations of deep generative models, these images show actual samples from the model distributions, not conditional means given samples of hidden units. Moreover, these samples are uncorrelated because the sampling process does not depend on Markov chain mixing. a) MNIST b) TFD c) CIFAR-10 (fully connected model) d) CIFAR-10 (convolutional discriminator and “deconvolutional” generator)
>> 그림 2: 모델의 샘플 시각화 오른쪽 끝 열은 모델이 훈련 세트를 기억하지 않았음을 보여주기 위해 인접 샘플의 가장 가까운 훈련 예를 보여준다. 표본은 무작위로 뽑은 것이지 체리를 뽑은 것이 아닙니다. 심층 생성 모델의 다른 대부분의 시각화와는 달리, 이러한 이미지는 숨겨진 단위의 샘플이 주어진 조건부 수단이 아니라 모델 분포의 실제 샘플을 보여준다. 더욱이, 이 샘플들은 샘플링 과정이 마르코프 연쇄 혼합에 의존하지 않기 때문에 상관관계가 없다. a) MNIST b) TFD c) CIFAR-10 (완전 연결된 모델) d) CIFAR-10 (컨볼루션 판별기 및 "디콘볼루션" 생성기)

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2021-09-26-(GAN)Generative-Adversarial-Nets-translation/Figure-3.JPG)

> Figure 3: Digits obtained by linearly interpolating between coordinates in $z$ space of the full model.
>> 그림 3: 전체 모델의 $z$ 공간에서 좌표 사이를 선형으로 보간하여 얻은 숫자.

> 1. A conditional generative model $p(x \mid  c)$ can be obtained by adding c as input to both $G$ and $D$.
>> 1. 조건부 생성 모델 $p(x \mid  c)$는 $G$와 $D$ 모두에 cas 입력을 추가하여 얻을 수 있다.
> 2. Learned approximate inference can be performed by training an auxiliary network to predict $z$ given $x$. This is similar to the inference net trained by the wake-sleep algorithm [15] but with the advantage that the inference net may be trained for a fixed generator net after the generator net has finished training.
>> 2. 학습된 근사 추론은 $x$가 주어진 $z$를 예측하도록 보조 네트워크를 훈련시킴으로써 수행될 수 있다. 이것은 웨이크 슬립 알고리듬[15]에 의해 훈련된 추론 네트와 유사하지만, 추론 네트가 훈련을 마친 후 고정 발전기 네트에 대해 훈련될 수 있다는 장점이 있다.
> 3. One can approximately model all conditionals $p(x_{s} \mid  x_{\not{s}})$ where S is a subset of the indices of $x$ by training a family of conditional models that share parameters. Essentially, one can use adversarial nets to implement a stochastic extension of the deterministic MP-DBM [10].
>> 3. 모든 조건부 $p(x_{s} \mid  x_{\not{s}})$를 대략적으로 모델링할 수 있다. 여기서 S는 매개 변수를 공유하는 조건부 모델 패밀리를 훈련시켜 $x$의 지수의 하위 집합이다. 본질적으로, 결정론적 MP-DBM의 확률적 확장을 구현하기 위해 적대적 네트워크를 사용할 수 있다[10].
> 4. Semi-supervised learning: features from the discriminator or inference net could improve performance of classifiers when limited labeled data is available.
>> 4. 준지도 학습: 판별기 또는 추론 네트워크의 기능은 제한된 레이블링된 데이터를 사용할 수 있을 때 분류기의 성능을 향상시킬 수 있다.
> 5. Efficiency improvements: training could be accelerated greatly by devising better methods for coordinating $G$ and $D$ or determining better distributions to sample $z$ from during training.
>> 5. 효율성 향상: $G$와 $D$를 조정하기 위한 더 나은 방법을 고안하거나 훈련 중에 $z$를 샘플링하기 위한 더 나은 분포를 결정함으로써 훈련을 크게 가속화할 수 있다.

> This paper has demonstrated the viability of the adversarial modeling framework, suggesting that these research directions could prove useful.
>> 이 논문은 적대적 모델링 프레임워크의 실행 가능성을 입증했으며, 이러한 연구 방향이 유용하게 입증될 수 있음을 시사했다.

![Table 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2021-09-26-(GAN)Generative-Adversarial-Nets-translation/Table-2.JPG)

> Table 2: Challenges in generative modeling: a summary of the difficulties encountered by different approaches to deep generative modeling for each of the major operations involving a model.
>> 표 2: 생성 모델링의 과제: 모델을 포함하는 각 주요 작업에 대한 심층 생성 모델링에 대한 다양한 접근 방식이 직면하는 어려움을 요약한다.

$Acknowledgments$

> We would like to acknowledge Patrice Marcotte, Olivier Delalleau, Kyunghyun Cho, Guillaume Alain and Jason Yosinski for helpful discussions. Yann Dauphin shared his Parzen window evaluation code with us. We would like to thank the developers of Pylearn2 [11] and Theano [6, 1], particularly Frederic Bastien who rushed a Theano feature specifically to benefit this project. Arnaud Bergeron provided much-needed support with $L^AT_EX$ typesetting. We would also like to thank CIFAR, and Canada Research Chairs for funding, and Compute Canada, and Calcul Quebec for providing computational resources. Ian Goodfellow is supported by the 2013 Google Fellowship in Deep Learning. Finally, we would like to thank Les Trois Brasseurs for stimulating our creativity.
>> Patrice Marcotte, Olivier Delalau, Kyunghyun Cho, Guillaume Alain, Jason Yosinski가 유익한 토론을 해주신 것에 대해 감사 드립니다. Yan Dauphin이 Parzen 창 평가 코드를 공유했습니다. Pylearn2[11]와 Theano[6,1]의 개발자들, 특히 이 프로젝트에 특별히 도움이 되기 위해 Theano 기능을 재촉한 Frederic Bastien에게 감사드립니다. Arnaud Bergeron은 $L^AT_EX$ 식자로 매우 필요한 지원을 제공했다. 또한 CIFAR, 캐나다 연구회의 자금조달, 캐나다 계산 및 퀘벡 계산에도 감사드립니다. Ian Goodfellow는 2013년 구글 딥러닝 펠로우십의 지원을 받고 있다. 마지막으로, 우리의 창의력을 자극해 준 Let Trois Braseurs에게 감사를 드리고 싶습니다.