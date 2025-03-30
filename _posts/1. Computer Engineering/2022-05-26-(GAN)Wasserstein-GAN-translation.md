---
layout: post 
title: "(GAN)Wasserstein GAN Translation"
categories: [1. Computer Engineering]
tags: [1.0. Paper Review, 1.2.2.1. Computer Vision]
---

### [CV Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/paper-of-GAN.html)

# $$Wasserstein\;GAN$$

### $\mathbf{1\;\;\;Introduction}$

> The problem this paper is concerned with is that of unsupervised learning. Mainly, what does it mean to learn a probability distribution? The classical answer to this is to learn a probability density. This is often done by defining a parametric family of densities $P_{\theta\in{R}^{d}}$ and finding the one that maximized $test$ the likelihood on our data:  
if we have real data examples $(x^{i})^{m}_{i=1}$ we would solve the problem
>> 이 논문이 다루는 문제는 비지도 학습에 관한 것이다. 주로, 확률 분포를 배운다는 것은 무엇을 의미합니까? 이에 대한 고전적인 대답은 확률 밀도를 배우는 것이다. 이것은 종종 밀도 $P_{\theta\in{R}^{d}}$의 파라메트릭 패밀리를 정의하고 데이터에 대한 가능성을 극대화한 것을 찾아냄으로써 이루어진다.  
만약 우리가 실제 데이터 예 $(x^{i})^{m}_{i=1}$를 가지고 있다면, 우리는 그 문제를 해결할 것이다.

$$\underset{\theta\in{R^{d}}}{\max}\frac{1}{m}\sum_{i=1}^{m} \log{P_{\theta}(x^{i})}$$

> If the real data distribution $P_{r}$ admits a density and $P_{\theta}$ is the distribution of the parametrized density $P_{\theta}$, then, asymptotically, this amounts to minimizing the Kullback-Leibler divergence $KL(P_{r}\parallel{P_{\theta}})$.
>> 실제 데이터 분포 $P_{r}$가 밀도를 허용하고 $P_{\theta}$가 매개 변수화된 밀도 $P_{\theta}$의 분포인 경우, 이는 점근적으로 쿨백-라이블러 발산 $KL(P_{r}\parallel{P_{\theta}})$를 최소화하는 것이다.

> For this to make sense, we need the model density $P_{\theta}$ to exist. This is not the case in the rather common situation where we are dealing with distributions supported by low dimensional manifolds. It is then unlikely that the model manifold and the true distribution’s support have a non-negligible intersection (see [1]), and this means that the KL distance is not defined (or simply infinite). The typical remedy is to add a noise term to the model distribution. This is why virtually all generative models described in the classical machine learning literature include a noise component. In the simplest case, one assumes a Gaussian noise with relatively high bandwidth in order to cover all the examples. It is well known, for instance, that in the case of image generation models, this noise degrades the quality of the samples and makes them blurry. For example, we can see in the recent paper [23] that the optimal standard deviation of the noise added to the model when maximizing likelihood is around 0.1 to each pixel in a generated image, when the pixels were already normalized to be in the range [0, 1]. This is a very high amount of noise, so much that when papers report the samples of their models, they don’t add the noise term on which they report likelihood numbers. In other words, the added noise term is clearly incorrect for the problem, but is needed to make the maximum likelihood approach work.
>> 이를 위해서는 모델 밀도 $P_{\theta}$가 존재해야 한다. 이것은 우리가 저차원 다양체에 의해 지원되는 분포를 다루는 다소 흔한 상황에서는 그렇지 않다. 따라서 모델 다양체와 실제 분포의 지지체가 무시할 수 없는 교집합을 가질 가능성은 낮으며 ([1] 참조), 이는 KL 거리가 정의되지 않는다는 것을 의미한다 (또는 단순히 무한하다. 일반적인 해결책은 모형 분포에 잡음 항을 추가하는 것입니다. 이것이 고전 기계 학습 문헌에 설명된 사실상 모든 생성 모델에 노이즈 성분이 포함된 이유이다. 가장 간단한 경우, 모든 예를 다루기 위해 상대적으로 높은 대역폭을 가진 가우스 노이즈를 가정한다. 예를 들어 이미지 생성 모델의 경우 이 소음이 샘플의 품질을 저하시키고 샘플을 흐리게 만든다는 것은 잘 알려져 있다. 예를 들어, 우리는 최근 논문 [23]에서 픽셀이 이미 [0, 1] 범위에 정규화된 경우, 우도 최대화 시 모델에 추가된 노이즈의 최적 표준 편차는 생성된 이미지의 각 픽셀에 대해 약 0.1이라는 것을 알 수 있다. 이것은 매우 높은 소음의 양입니다. 논문들이 그들의 모델의 샘플을 보도할 때, 그들은 가능성 수치를 보고하는 소음 항을 추가하지 않습니다. 다시 말해, 추가된 소음 항은 문제에 대해 명백히 부정확하지만, 최대우도 접근법이 작동하도록 하기 위해 필요하다.

> Rather than estimating the density of $P_{r}$ which may not exist, we can define a random variable $Z$ with a fixed distribution  $p(z)$  and pass it through a parametric function $g_{\theta}:Z\to{X}$ (typically a neural network of some kind) that directly generates samples following a certain distribution $P_{\theta}$. By varying $\theta$, we can change this distribution and make it close to the real data distribution $P_{r}$. This is useful in two ways. First of all, unlike densities, this approach can represent distributions confined to a low dimensional manifold. Second, the ability to easily generate  samples is often more useful than knowing the numerical value of the density (for example in image superresolution or semantic segmentation when considering the conditional distribution of the output image given the input image). In general, it is computationally difficult to generate samples given an arbitrary high dimensional density [16].
>> 존재하지 않을 수 있는 $P_{r}$의 밀도를 추정하는 대신, 우리는 고정된 분포 $p(z)$로 무작위 변수 $Z$를 정의하고 특정 분포 $P_{\theta}$에 따라 직접 샘플을 생성하는 매개 변수 함수 $g_{\theta}:Z\to{X}$(일반적으로 일종의 신경 네트워크)를 통해 이를 전달할 수 있다. $\theta$를 변화시킴으로써, 우리는 이 분포를 변경하고 실제 데이터 분포 $P_{r}$에 가깝게 만들 수 있다. 이것은 두 가지 면에서 유용하다. 무엇보다도, 밀도와 달리, 이 접근법은 저차원 다양체에 국한된 분포를 나타낼 수 있다. 둘째, 샘플을 쉽게 생성할 수 있는 능력은 종종 밀도의 수치 값을 아는 것보다 더 유용하다(예를 들어, 입력 이미지가 주어진 출력 이미지의 조건부 분포를 고려할 때 이미지 초해상도 또는 의미 분할에서). 일반적으로 임의의 고차원 밀도가 주어지면 샘플을 생성하는 것은 계산적으로 어렵다[16].

> Variational Auto-Encoders (VAEs) [9] and Generative Adversarial Networks (GANs) [4] are well known examples of this approach. Because VAEs focus on the approximate likelihood of the examples, they share the limitation of the standard models and need to fiddle with additional noise terms. GANs offer much more flexibility in the definition of the objective function, including Jensen-Shannon [4], and all f-divergences [17] as well as some exotic combinations [6]. On the other hand, training GANs is well known for being delicate and unstable, for reasons theoretically investigated in [1].
>> 변형 자동 인코더(VAE)[9] 및 생성 적대적 네트워크(GAN)[4]는 이 접근 방식의 잘 알려진 예이다. VAE는 사례의 대략적인 가능성에 중점을 두기 때문에 표준 모델의 한계를 공유하며 추가적인 노이즈 용어를 만지작거려야 한다. GAN은 Jensen-Shannon[4]과 모든 f-분산[17] 및 일부 이국적인 조합[6]을 포함하여 목적 함수의 정의에 훨씬 더 많은 유연성을 제공한다. 한편, GAN 훈련은 [1]에서 이론적으로 조사된 이유로 섬세하고 불안정한 것으로 잘 알려져 있다.

> In this paper, we direct our attention on the various ways to measure how close the model distribution and the real distribution are, or equivalently, on the various ways to define a distance or divergence $ρ(P_{\theta}, P_{r})$. The most fundamental difference between such distances is their impact on the convergence of sequences of probability distributions. A sequence of distributions $P_{t\in{ N}}$ converges if and only if there is a distribution $P_{\infty}$ such that $ρ(P_{t}, P_{\infty})$ tends to zero, something that depends on how exactly the distance $ρ$ is defined. Informally, a distance $ρ$ induces a weaker topology when it makes it easier for a sequence of distribution to converge.1 Section 2 clarifies how popular probability distances differ in that respect.
>> 본 논문에서, 우리는 거리 또는 발산 $ρ(P_{\theta}, P_{r})$를 정의하기 위한 다양한 방법에 대해 모델 분포와 실제 분포가 얼마나 가깝거나 동등하게 측정하기 위한 다양한 방법에 대한 주의를 기울인다. 이러한 거리 사이의 가장 근본적인 차이는 확률 분포 시퀀스의 수렴에 미치는 영향이다. 분포 $P_{t\in{N}}$의 수열은 분포 $P_{\infty}$가 있는 경우에만 수렴하며, $ρ(P_{t}, P_{\infty})$가 0이 되는 경향이 있다. 비공식적으로 거리 $θ$는 분포 시퀀스가 수렴하기 더 쉬울 때 약한 위상을 유도한다.1 섹션 2는 그러한 측면에서 인기 있는 확률 거리가 어떻게 다른지를 명확히 한다.

> In order to optimize the parameter $\theta$, it is of course desirable to define our model distribution $P_{\theta}$ in a manner that makes the mapping $\theta\to{P_{\theta}}$ continuous. Continuity means that when a sequence of parameters $\theta_{t}$ converges to $\theta$, the distributions $P_{\theta_{t}}$ also converge to $P_{\theta}$. However, it is essential to remember that the notion of the convergence of the distributions $P_{\theta_{t}}$ depends on the way we compute the distance between distributions. The weaker this distance, the easier it is to define a continuous mapping from $\theta$-space to $P_{\theta}$-space, since it’s easier for the distributions to converge. The main reason we care about the mapping $\theta\to{P_{\theta}}$ to be continuous is as follows. If $ρ$ is our notion of distance between two distributions, we would like to have a loss function $\theta\to{p}(P_{\theta}, P_{r})$ that is continuous, and this is equivalent to having the mapping $\theta\to{P_{\theta}}$ be continuous when using the distance between distributions ρ.
>> 매개 변수 $\theta$를 최적화하려면 매핑 $\theta\to{P_{\theta}}$를 연속적으로 만드는 방식으로 모델 분포 $P_{\theta}$를 정의하는 것이 물론 바람직하다. 연속성은 모수 $\theta_{t}$가 $\theta$로 수렴할 때 분포 $P_{\theta_{t}}$도 $P_{\theta}$로 수렴한다는 것을 의미한다. 그러나 분포 $P_{\theta_{t}}$의 수렴 개념은 분포 사이의 거리를 계산하는 방법에 따라 달라진다는 것을 기억해야 한다. 이 거리가 약할수록 분포가 수렴하기 쉽기 때문에 $P_{\theta}$-공간에서 $\theta$-공간으로의 연속적인 매핑을 정의하는 것이 더 쉽다. 우리가 지도 제작에 관심을 갖는 주된 이유인  $\theta\to{P_{\theta}}$는 다음과 같다. $ρ$가 두 분포 사이의 거리에 대한 개념이라면, 우리는 연속적인 손실 함수 $\theta\to{p}(P_{\theta}, P_{r})$를 가지기를 원하며, 이는 분포 사이의 거리를 사용할 때 매핑 $\theta\to{P_{\theta}}$가 연속적이 되는 것과 같다.

> The contributions of this paper are:

> * In Section 2, we provide a comprehensive theoretical analysis of how the Earth Mover (EM) distance behaves in comparison to popular probability distances and divergences used in the context of learning distributions.
>> * 섹션 2에서, 우리는 학습 분포의 맥락에서 사용되는 인기 있는 확률 거리와 분기와 비교하여 Earth Mover(EM) 거리가 어떻게 작용하는지에 대한 포괄적인 이론적 분석을 제공한다.
> * In Section 3, we define a form of GAN called Wasserstein-GAN that minimizes a reasonable and efficient approximation of the EM distance, and we theoretically show that the corresponding optimization problem is sound.
>> * 섹션 3에서는 전자파 거리의 합리적이고 효율적인 근사치를 최소화하는 와서스테인-GAN이라는 GAN의 형태를 정의하고, 해당 최적화 문제가 건전하다는 것을 이론적으로 보여준다.
> * In Section 4, we empirically show that WGANs cure the main training problems of GANs. In particular, training WGANs does not require maintaining a careful balance in training of the discriminator and the generator, and does not require a careful design of the network architecture either. The mode dropping phenomenon that is typical in GANs is also drastically reduced. One of the most compelling practical benefits of WGANs is the ability to continuously estimate the EM distance by training the discriminator to optimality. Plotting these learning curves is not only useful for debugging and hyperparameter searches, but also correlate remarkably well with the observed sample quality.
>> * 섹션 4에서, 우리는 WGAN이 GAN의 주요 훈련 문제를 해결한다는 것을 경험적으로 보여준다. 특히, WGAN을 훈련시키는 것은 판별기와 발전기의 훈련에서 신중한 균형을 유지할 필요가 없으며, 네트워크 아키텍처의 신중한 설계도 필요하지 않다. GAN에서 전형적으로 나타나는 모드 드롭 현상도 대폭 감소한다. WGAN의 가장 설득력 있는 실용적인 이점 중 하나는 판별기를 최적화하도록 훈련시켜 전자파 거리를 지속적으로 추정할 수 있는 능력이다. 이러한 학습 곡선을 그리는 것은 디버깅 및 초 매개 변수 검색에 유용할 뿐만 아니라 관찰된 샘플 품질과 현저하게 잘 상관된다.

### $\mathbf{2\;\;\;Different\;Distances}$

> We now introduce our notation. Let $X$ be a compact metric set (such as the space of images [0, 1]$^{d}$ ) and let $\sum$ denote the set of all the Borel subsets of $X$. Let $\;\mathrm{Prob}(X)$ denote the space of probability measures defined on $X$ . We can now define elementary distances and divergences between two distributions $P_{r}, P_{g}\in\mathrm{Prob}(X)$:
>> 이제 우리의 표기법을 소개하겠습니다. $X$가 콤팩트 메트릭 세트(예: 이미지 공간 [0, 1]$^{d}$)이고 $\sum$이 $X$의 모든 보렐 하위 세트 세트를 나타내도록 하자. $\mathrm{Prob}(X)$가 $X$에 정의된 확률 측정의 공간을 나타내도록 하자. 이제 두 분포 $P_{r}, P_{g}\in\mathrm{Prob}(X)$사이 의 기본 거리와 분산을 정의할 수 있다.

> * The Total Variation (TV) distance
>> * 총 변동(TV) 거리

$$\delta(P_{r},P_{g}) = \underset{A\in\sum}{\sup}\mid P_{r}(A)-P_{g}(A)\mid .$$

> * The Kullback-Leibler (KL) divergence
>> * 쿨백-라이블러(KL) 발산

$$KL(P_{r}\parallel P_{g}) = \int\log{(\frac{P_{r}(x)}{P_{g}(x)})}P_{r}(x)d\mu(x),$$

> where both $P_{r}$ and $P_{g}$ are assumed to be absolutely continuous, and therefore admit densities, with respect to a same measure µ defined on $X$ . 2 The KL divergence is famously assymetric and possibly infinite when there are points such that $P_{g}(x) = 0$ and $P_{r}(x) > 0$.
>> 여기서 둘 다 $P_{r}$ 및 $P_{g}$는 절대적으로 연속적인 것으로 가정되며, 따라서 $X$에 정의된 동일한 측정값과 관련하여 밀도를 허용한다.2 KL 분기는 $P_{g}(x) = 0$와 $P_{r}(x) > 0$와 같은 점이 있을 때 유명한 비대칭이며 무한할 수 있다.

> * The Jensen-Shannon (JS) divergence
>> * 옌센-샤논 (JS) 발산

$$JS(P_{r},P_{g})=KL(P_{r}\parallel{P_{m}})+KL(P_{g}\parallel{P_{m}}),$$

> where $P_{m}$ is the mixture $(P_{r} + P_{g})/2$. This divergence is symmetrical and always defined because we can choose $µ = P_{m}$.
>> 여기서 $P_{m}$는 혼합물 $(P_{r} + P_{g})/2$입니다. 이 분산은 대칭이며 항상 정의된다. 왜냐하면 우리는 $µ = P_{m}$.를 선택할 수 있기 때문이다.

> * The Earth-Mover (EM) distance or Wasserstein-1

$$W(P_{r},P_{g})=\underset{\gamma\in\prod(P_{r},P_{g})}{\inf}E_{(x,y)\sim{\gamma}}[\parallel{x-y}\parallel],$$

> where $Π(P_{r}, P_{g})$ denotes the set of all joint distributions $γ(x,y)$ whose marginals are respectively $P_{r}$ and $P_{g}$. Intuitively, $γ(x,y)$ indicates how much “mass” must be transported from x to y in order to transform the distributions $P_{r}$ into the distribution $P_{g}$. The EM distance then is the “cost” of the optimal transport plan.
>> 여기서 $Π(P_{r}, P_{g})$는 마진이 각각 $P_{r}$ and $P_{g}$인 모든 공동 분포 $((x,y)$의 집합을 나타낸다. 직관적으로 $θ(x,y)$는 분포 $P_{r}$를 분포 $P_{g}$.로 변환하기 위해 x에서 y로 얼마나 많은 "질량"을 전송해야 하는지를 나타낸다. 그러면 전자파 거리는 최적 전송 계획의 "비용"이다.

> The following example illustrates how apparently simple sequences of probability distributions converge under the EM distance but do not converge under the other distances and divergences defined above.
>> 다음 예는 확률 분포의 명백한 간단한 시퀀스가 전자파 거리 아래에서 수렴되지만 위에 정의된 다른 거리와 분산 아래에서 수렴되지 않는 방법을 보여준다.

> **Example 1** (Learning parallel lines). Let $Z\sim{U[0, 1]}$ the uniform distribution on the unit interval. Let $P_{0}$ be the distribution of $(0, Z) \in R 2$ (a 0 on the x-axis and the random variable $Z$ on the y-axis), uniform on a straight vertical line passing through the origin. Now let $g_{\theta}(z) = (\theta, z)$ with $\theta$ a single real parameter. It is easy to see that in this case,
>> **예 1**(병렬 학습) 단위 구간의 균일한 분포를 $Z\sim{U[0, 1]}$로 합니다. $P_{0}$ 를 $(0, Z) \in R 2$의 분포(x축의 0과 y축의 무작위 변수 $Z$)로 하고, 원점을 통과하는 직선 수직선에서 균일하게 한다. 이제 $\theta$를 가진 $g_{\theta}(z) = (\theta, z)$를 단일 실제 매개 변수로 설정한다. 이 경우엔 쉽게 알 수 있다.

$$\bullet{\;W(P_{0},P_{\theta}) = \mid \theta\mid },\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$\bullet{\;JS(P_{0},P_{\theta}) = \log{2}\;\mathrm{if}\;\theta\neq{0},\;0\;\mathrm{if}\;\theta={0},}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

$$\bullet{\;KL(P_{\theta}\parallel P_{0}) = \;KL(P_{0}\parallel P_{\theta})}= +\infty\;\mathrm{if}\;\theta\neq{0},\;0\;\mathrm{if}\;\theta={0},$$

$$\bullet{\;\mathrm{and}\;\delta(P_{0},P_{\theta}) =\;1\;\mathrm{if}\;\theta\neq{0},\;0\;\mathrm{if}\;\theta = 0.}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

> When $\theta_{t}\to{0}$, the sequence $P_{t\in{N}}$ converges to $P_{0}$ under the EM distance, but does not converge at all under either the JS, KL, reverse KL, or TV divergences. Figure 1 illustrates this for the case of the EM and JS distances.
>> $\theta_{t}\to{0}$일 때, 수열  $P_{t\in{N}}$는 전자파 거리에서는 $P_{0}$로 수렴하지만, JS, KL, 역KL, TV 발산에서는 전혀 수렴하지 않는다. 그림 1은 전자파 및 JS 거리의 경우에 대해 이를 보여준다.

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)Wasserstein-GAN-translation/Figure-1.JPG)

> **Example 1** gives us a case where we can learn a probability distribution over a low dimensional manifold by doing gradient descent on the EM distance. This cannot be done with the other distances and divergences because the resulting loss function is not even continuous. Although this simple example features distributions with disjoint supports, the same conclusion holds when the supports have a non empty intersection contained in a set of measure zero. This happens to be the case when two low dimensional manifolds intersect in general position [1].
>> **예 1**은 전자파 거리에서 그레이디언트 강하를 수행하여 저차원 매니폴드에 대한 확률 분포를 학습할 수 있는 사례를 제공한다. 이는 결과 손실 함수가 연속적이지도 않기 때문에 다른 거리와 발산으로는 수행될 수 없다. 이 간단한 예제는 분리된 지지대가 있는 분포를 특징으로 하지만, 지지대가 측정값 0 집합에 포함된 비어 있지 않은 교차점을 가질 때 동일한 결론이 유지된다. 이것은 두 개의 저차원 다양체가 일반적인 위치[1]에서 교차하는 경우에 발생한다.

> Since the Wasserstein distance is much weaker than the JS distance3 , we can now ask whether $W(P_{r}, P_{\theta})$ is a continuous loss function on $\theta$ under mild assumptions. This, and more, is true, as we now state and prove.
>> 와서스테인 거리는 JS 거리 3보다 훨씬 약하기 때문에, 이제 가벼운 가정 하에서 $W(P_{r}, P_{\theta})$가 $\theta$에 대한 연속 손실 함수인지 물어볼 수 있다. 우리가 지금 진술하고 증명하듯이, 이것뿐만 아니라 그 이상도 사실이다.

> **Theorem 1.** Let $P_{r}$ be a fixed distribution over $X$ . Let $Z$ be a random variable (e.g Gaussian) over another space $Z$. Let $g : Z × R d\to{X}$ be a function, that will be denoted $g_{\theta}(z)$ with z the first coordinate and $\theta$ the second. Let $P_{\theta}$ denote the distribution of $g_{\theta}(Z)$. Then,
>> **정식 1.** 를 $X$에 대한 고정 분포로 하자. $Z$를 다른 공간 $Z$에 대한 무작위 변수(예: 가우스)로 하자. $g : Z × R d\to{X}$를 첫 번째 좌표 z와 두 번째 좌표 $\theta$로 $g_{\theta}(z)$로 나타낼 함수라고 하자. $P_{\theta}$가 $g_{\theta}(Z)$의 분포를 나타내도록 하자. 그리고나서,

> 1. If $g$ is continuous in $\theta$, so is $W(P_{r},P_{\theta})$.
>> 1. $g$가 $\theta$에서 연속이면 $W(P_{r},P_{\theta})$도 연속적이다.
> 2. If $g$ is locally Lipschitz and satisfies regularity assumption 1, then $W(P_{r}, P_{\theta})$ is continuous everywhere, and differentiable almost everywhere.
>>  2. $g$가 국소적으로 립시츠이고 규칙성 가정 1을 만족한다면, $W(P_{r}, P_{\theta})$는 모든 곳에서 연속적이며 거의 모든 곳에서 미분 가능하다.
> 3. Statements 1-2 are false for the Jensen-Shannon divergence $JS(P_{r}, P_{\theta})$ and all the KLs.
>>  3. 진술 1-2는 옌센-샤논 발산 $JS(P_{r}, P_{\theta})$와 모든 KL에 대해 거짓이다.

> Proof. See Appendix C
>> 증명. 부록 C 참조

> The following corollary tells us that learning by minimizing the EM distance makes sense (at least in theory) with neural networks.
>> 다음 결과는 전자파 거리를 최소화하여 학습하는 것이 (적어도 이론적으로는) 신경망을 통해 타당하다는 것을 말해준다.

> *Corollary 1.* Let $g_{\theta}$ be any feedforward neural network4 parameterized by $\theta$, and  $p(z)$  a prior over $z$ such that $E_{z∼p(z)}[\parallel{z}\parallel] < \infty$ (e.g. Gaussian, uniform, etc.).
>> *상관 1.* $g_{\theta}$를 $\theta$에 의해 매개 변수화된 피드포워드 신경망 4로 하고 $p(z)$를 $z$보다 이전 값으로 $E_{z∼p(z)}[\parallel{z}\parallel] < \infty$(예: 가우스, 균일 등)로 한다.

> Then assumption 1 is satisfied and therefore $W(P_{r}, P_{\theta})$ is continuous everywhere and differentiable almost everywhere.
>> 그러면 가정 1이 만족되고 따라서 $W(P_{r}, P_{\theta})$는 모든 곳에서 연속적이며 거의 모든 곳에서 미분될 수 있다.

> Proof. See Appendix C
>> 증명. 부록 C 참조

> All this shows that EM is a much more sensible cost function for our problem than at least the Jensen-Shannon divergence. The following theorem describes the relative strength of the topologies induced by these distances and divergences, with KL the strongest, followed by JS and TV, and EM the weakest.
>> 이 모든 것은 EM이 적어도 Jensen-Shannon 발산보다 우리의 문제에 대해 훨씬 더 합리적인 비용 함수라는 것을 보여준다. 다음 정리는 KL이 가장 강하고, JS와 TV가 그 뒤를 이으며, EM이 가장 약한 위상의 상대적 강도를 설명한다.

> **Theorem 2.** Let $P$ be a distribution on a compact space $X$ and $(P_{n})n\in{ N}$ be a sequence of distributions on $X$ . Then, considering all limits as $n\to{\infty}$,
>> **정론 2.** $P$를 콤팩트 공간 $X$에 대한 분포로 하고, $(P_{n})n\in{ N}$를 $X$에 대한 분포의 시퀀스로 하자. 그러면, 모든 한계를 $n\to{\infty}$,로 간주한다.

> 1. The following statements are equivalent  
>>   1. 다음 문장은 동등합니다.  
> * $\delta(P_{n}, P)\to{0}$ with $\delta $ the total variation distance.  
>>  * $\delta $의 $\delta(P_{n}, P)\to{0}$ 총 변동 거리.
>   * $JS(P_{n}, P)\to{0}$ with JS the Jensen-Shannon divergence.  
>>  * JS에서 $JS(P_{n}, P)\to{0}$는 옌센-샤논 분산을 의미한다.
> 2. The following statements are equivalent  
>>  2. 다음 문장은 동등합니다.
>   * $W(P_{n}, P)\to{0}$.  
>   * $P_{n}\xrightarrow[]{D}P$ where $\xrightarrow[]{D}$ represents convergence in distribution for randomvariables.
>>  * $P_{n}\xrightarrow[]{D}P$ 에서 $\xrightarrow[]{D}$는 랜덤 변수에 대한 분포의 수렴을 나타냅니다.
> 3. $KL(P_{n}\parallel P)\to{0}$ or $KL(P\parallel P_{n})\to{0}$ imply the statements in (1).  
>>  3. $KL(P_{n}\parallel P)\to{0}$ 또는 $KL(P\parallel P_{n})\to{0}$는 (1)의 문구를 의미한다.
> 4. The statements in (1) imply the statements in (2).
>>  4. (1)의 진술은 (2)의 진술을 암시한다.

> Proof. See Appendix C
>> 증명. 부록 C 참조

> This highlights the fact that the KL, JS, and TV distances are not sensible cost functions when learning distributions supported by low dimensional manifolds. However the EM distance is sensible in that setup. This obviously leads us to the next section where we introduce a practical  approximation of optimizing the EM distance.
>> 이는 저차원 다양체가 지원하는 분포를 학습할 때 KL, JS 및 TV 거리가 합리적인 비용 함수가 아니라는 사실을 강조한다. 그러나 이 설정에서는 전자파 거리가 적절하다. 이는 분명히 전자파 거리 최적화의 실질적인 근사치를 소개하는 다음 섹션으로 이어진다.

### $\mathbf{3\;\;\;Wasserstein\;GAN}$

> Again, Theorem 2 points to the fact that $W(P_{r}, P_{\theta})$ might have nicer properties when optimized than $JS(P_{r}, P_{\theta})$. However, the infimum in (1) is highly intractable. On the other hand, the Kantorovich-Rubinstein duality [22] tells us that
>> 다시 정리 2는 $W(P_{r}, P_{\theta})$가 $JS(P_{r}, P_{\theta})$보다 최적화되었을 때 더 좋은 성질을 가질 수 있다는 사실을 가리킨다. 그러나 (1)의 임피엄은 매우 다루기 어렵다. 반면에, 칸토로비치-루빈스타인 이중성[22]은 우리에게 다음과 같이 말해준다.

$$W(P_{r},P_{\theta})=\underset{f_{L}\leq{1}}{\mathrm{sup}}E_{x\sim{P_{r}}}[f(x)]-E_{x\sim{P_{\theta}}}[f(x)]$$
    
> where the supremum is over all the 1-Lipschitz functions $f : X\to{R}$. Note that if we replace $\parallel{f}\parallel_{L} ≤ 1$ for $\parallel{f}\parallel_{L} ≤ K$ (consider K-Lipschitz for some constant K), then we end up with $K ·W(P_{r}, P_{g})$. Therefore, if we have a parameterized family of functions ${f_{w}}_{w}\in{W}$ that are all K-Lipschitz for some $K$, we could consider solving the problem
>> 여기서 수미는 모든 1-립시츠 함수 $f : X\to{R}$에 걸쳐 있다. 만약 우리가 $\parallel{f}\parallel_{L} ≤ 1$를 $\parallel{f}\parallel_{L} ≤ K$로 치환한다면(일부 상수 K에 대해 K-립시츠를 고려함), 우리는 $K ·W(P_{r}, P_{g})$로 끝난다. 따라서, 만약 우리가 $K$에 대해 모두 K-립시츠인 매개 변수화된 함수 ${f_{w}}_{w}\in{W}$가 있다면, 우리는 문제 해결을 고려할 수 있다.

$$\underset{w\in{W}}{\mathrm{max}}E_{x\sim{P_{r}}}[f_{w}(x)]-E_{z\sim{p(z)}}[f_{w}(g_{\theta}(z))]$$

> and if the supremum in (2) is attained for some $w\in{W}$ (a pretty strong assumption akin to what’s assumed when proving consistency of an estimator), this process would yield a calculation of $W(P_{r}, P_{\theta})$ up to a multiplicative constant. Furthermore, we could consider differentiating $W(P_{r}, P_{\theta})$  (again, up to a constant) by back-proping through equation (2) via estimating $E_{z}∼p(z)[\triangledown\theta{f_{w}}(g_{\theta}(z))]$. While this is all intuition, we now prove that this process is principled under the optimality assumption.
>> 그리고 만약 어떤 $w\in{W}$ (추정자의 일관성을 증명할 때 가정하는 것과 유사한 상당히 강력한 가정)에 대해 (2)의 최고값이 달성된다면, 이 과정은 곱셈 상수까지 $W(P_{r}, P_{\theta})$를 계산하게 될 것이다. 게다가, 우리는 $E_{z}∼p(z)[\triangledown\theta{f_{w}}(g_{\theta}(z))]$를 추정함으로써 방정식 (2)를 통해 역프로핑을 통해 $W(P_{r}, P_{\theta})$를 구별하는 것을 고려할 수 있다. 이것은 모두 직관적인 것이지만, 우리는 이제 이 과정이 최적성 가정 하에서 원칙적이라는 것을 증명한다.

> **Theorem 3.** Let $P_{r}$ be any distribution. Let $P_{\theta}$ be the distribution of $g_{\theta}(Z)$ with $Z$ a random variable with density $p$ and $g_{\theta}$ a function satisfying assumption 1. Then, there is a solution $f : X\to{R}$ to the problem
>> **정식 3.** $P_{r}$를 임의의 분포로 합니다. $P_{\theta}$는 밀도 $p$의 무작위 변수를 가진 $g_{\theta}(Z)$의 분포이고, $g_{\theta}$는 가정 1을 만족시키는 함수라고 하자. 그러면, 그 문제에 대한 해결책 $f : X\to{R}$가 있다.

$$\underset{f_{L}\leq{1}\mathrm{}}{\mathrm{max}}E_{x\sim{P_{r}}}[f(x)]-E_{x\sim{P_{\theta}}}[f(x)]$$

> and we have

$$\triangledown_{\theta}W(P_{r},P_{\theta}) = -E_{z\sim{p(z)}}[\triangledown_{\theta}f(g_{\theta}(z))]$$

> when both terms are well-defined.
>> 두 용어가 모두 잘 정의되어 있을 때

> Proof. See Appendix C
>> 증명. 부록 C 참조

> Now comes the question of finding the function f that solves the maximization problem in equation (2). To roughly approximate this, something that we can do is train a neural network parameterized with weights w lying in a compact space W and then backprop through $Ez∼p(z) [\triangledown\theta{f_{w}}(g_{\theta}(z))]$, as we would do with a typical GAN. Note that the fact that $W$ is compact implies that all the functions $f_{w}$ will be K-Lipschitz for some $K$ that only depends on $W$ and not the individual weights, therefore approximating (2) up to an irrelevant scaling factor and the capacity of the ‘critic’ fw. In order to have parameters $w$ lie in a compact space, something simple we can do is clamp the weights to a fixed box (say $W = [−0.01, 0.01]^{l}$ ) after each gradient update. The Wasserstein Generative Adversarial Network (WGAN) procedure is described in Algorithm 1.
>> 이제 방정식 (2)에서 최대화 문제를 해결하는 함수 f를 찾는 문제가 나온다. 이를 대략적으로 추정하기 위해, 우리가 할 수 있는 것은 일반적인 GAN에서 하는 것처럼 컴팩트 공간 W에 나부끼는 가중치로 매개 변수화된 신경망을 훈련시킨 다음 $Ez∼p(z) [\triangledown\theta{f_{w}}(g_{\theta}(z))]$를 통해 백프로핑하는 것이다. $W$가 콤팩트하다는 사실은 개별 가중치가 아닌 $W$에만 의존하는 일부 $K$에 대해 모든 함수 $f_{w}$가 K-Lipschitz가 된다는 것을 의미하므로, 관련 없는 스케일링 팩터와 '비판' fw의 용량까지 근사하게 된다. 매개 변수 $w$가 콤팩트 공간에 배치되도록 하려면 각 그레이디언트 업데이트 후 가중치를 고정 상자(예: $W = [−0.01, 0.01]^{l}$)에 클램프하는 것이 간단하다. WGAN(Wasserstein Generative Adversarial Network) 절차는 알고리듬 1에 설명되어 있다.

> Weight clipping is a clearly terrible way to enforce a Lipschitz constraint. If the clipping parameter is large, then it can take a long time for any weights to reach their limit, thereby making it harder to train the critic till optimality. If the clipping is small, this can easily lead to vanishing gradients when the number of layers is big, or batch normalization is not used (such as in RNNs). We experimented with simple variants (such as projecting the weights to a sphere) with little difference, and we stuck with weight clipping due to its simplicity and already good performance. However, we do leave the topic of enforcing Lipschitz constraints in a neural network setting for further investigation, and we actively encourage interested researchers to improve on this method.
>> 웨이트 클리핑은 립시츠 제약 조건을 강제하는 분명히 끔찍한 방법이다. 클리핑 매개 변수가 크면 가중치가 한계에 도달하는 데 오랜 시간이 걸릴 수 있으므로 최적화할 때까지 비평가를 훈련하는 것이 더 어려워진다. 클리핑이 작으면 레이어의 수가 크거나 배치 정규화가 사용되지 않을 때(예: RNN) 그레이디언트가 쉽게 사라질 수 있다. 우리는 거의 차이가 없는 간단한 변형(구체에 가중치를 투영하는 것과 같은)으로 실험했고, 단순성과 이미 우수한 성능으로 인해 가중치 클리핑을 고수했다. 그러나, 우리는 추가 조사를 위해 신경망 설정에서 립시츠 제약 조건을 적용하는 주제를 남겨두고, 관심 있는 연구자들이 이 방법을 개선하도록 적극 권장한다.

> Algorithm 1 WGAN, our proposed algorithm. All experiments in the paper used
the default values $α = 0.00005, c = 0.01, m = 64, ncritic = 5$.
>> Algorithm 1 WGAN, 우리가 제안한 알고리즘. 사용된 논문의 모든 실험
기본 값 $tv = 0.00005, c = 0.01, m = 64, ncritic = 5$입니다.

![Algorithm 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)Wasserstein-GAN-translation/Algorithm-1.JPG)

> The fact that the EM distance is continuous and differentiable a.e. means that we can (and should) train the critic till optimality. The argument is simple, the more we train the critic, the more reliable gradient of the Wasserstein we get, which is actually useful by the fact that Wasserstein is differentiable almost everywhere. For the JS, as the discriminator gets better the gradients get more reliable but the true gradient is 0 since the JS is locally saturated and we get vanishing gradients, as can be seen in Figure 1 of this paper and Theorem 2.4 of [1]. In Figure 2 we show a proof of concept of this, where we train a GAN discriminator and a WGAN critic till optimality. The discriminator learns very quickly to distinguish between fake and real, and as expected provides no reliable gradient information. The critic, however, can’t saturate, and converges to a linear function that gives remarkably clean gradients everywhere. The fact that we constrain the weights limits the possible growth of the function to be at most linear in different parts of the space, forcing the optimal critic to have this behaviour.
>> 전자파 거리가 연속적이고 미분 가능하다는 것은 최적화가 될 때까지 비평가를 훈련시킬 수 있다는 것을 의미한다. 논쟁은 간단합니다. 우리가 비평가들을 훈련시킬수록, 우리는 더 신뢰할 수 있는 바세르슈타인의 기울기를 얻게 되는데, 이것은 실제로 바세르슈타인이 거의 모든 곳에서 차별화 가능하다는 사실에 의해 유용합니다. JS의 경우 판별기가 향상될수록 그레이디언트의 신뢰성은 높아지지만, 본 논문의 그림 1과 [1]의 정리 2.4에서 볼 수 있듯이 JS는 국소 포화 상태이고 우리는 소멸 그레이디언트를 얻기 때문에 진정한 그레이디언트는 0이다. 그림 2에서 우리는 이것의 개념 증명을 보여주는데, 여기서 우리는 GAN 판별자와 WGAN 비평가를 최적화할 때까지 훈련시킨다. 판별기는 가짜와 진짜를 구별하기 위해 매우 빠르게 학습하며, 예상대로 신뢰할 수 있는 그레이디언트 정보를 제공하지 않는다. 그러나 비평가는 포화되지 않고 모든 곳에서 현저하게 깨끗한 구배를 제공하는 선형 함수로 수렴한다. 가중치를 제한한다는 사실은 함수의 가능한 성장을 공간의 다른 부분에서 최대 선형으로 제한하여 최적의 비평가가 이러한 행동을 갖도록 한다.

> Perhaps more importantly, the fact that we can train the critic till optimality makes it impossible to collapse modes when we do. This is due to the fact that mode collapse comes from the fact that the optimal generator for a fixed discriminator is a sum of deltas on the points the discriminator assigns the highest values, as observed by [4] and highlighted in [11].
>> 아마도 더 중요한 것은, 우리가 최적성까지 비평가를 훈련시킬 수 있다는 사실이 우리가 그럴 때 모드를 축소하는 것을 불가능하게 만든다는 것이다. 이는 고정 판별기에 대한 최적 발생기가 [4]에 의해 관찰되고 [11]에서 강조된 것처럼 판별기가 가장 높은 값을 할당하는 점의 델타 합이라는 사실에서 모드 붕괴가 발생하기 때문이다.

> In the following section we display the practical benefits of our new algorithm, and we provide an in-depth comparison of its behaviour and that of traditional GANs.
>> 다음 섹션에서는 새로운 알고리듬의 실질적인 이점을 표시하고, 기존 GAN의 동작과 동작을 심층적으로 비교한다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)Wasserstein-GAN-translation/Figure-2.JPG)

> Figure 2: Optimal discriminator and critic when learning to differentiate two Gaussians. As we can see, the discriminator of a minimax GAN saturates and results in vanishing gradients. Our WGAN critic provides very clean gradients on all parts of the space.
>> 그림 2: 두 가우시인을 구별하는 방법을 배울 때 최적의 판별자 및 비판자 우리가 볼 수 있듯이 미니맥스 GAN의 판별기는 포화되어 그레이디언트가 사라진다. 우리의 WGAN 비평가는 공간의 모든 부분에서 매우 깨끗한 그레이디언트를 제공한다.

### $\mathbf{4\;\;\;Empirical\;Results}$

> We run experiments on image generation using our Wasserstein-GAN algorithm and show that there are significant practical benefits to using it over the formulation used in standard GANs.
>> 우리는 우리의 Wasserstein-GAN 알고리듬을 사용하여 이미지 생성에 대한 실험을 실행하고 표준 GAN에 사용되는 공식에 비해 이를 사용하는 데 상당한 실질적인 이점이 있음을 보여준다.

> We claim two main benefits:
>> 다음과 같은 두 가지 주요 이점이 있습니다.

> * a meaningful loss metric that correlates with the generator’s convergence and sample quality
>>  * 발전기의 수렴 및 샘플 품질과 관련된 의미 있는 손실 메트릭
> * improved stability of the optimization process
>>  * 최적화 프로세스의 안정성 향상

### $\mathbf{4.1\;\;\;Experimental\;Procedure}$

> We run experiments on image generation. The target distribution to learn is the LSUN-Bedrooms dataset [24] – a collection of natural images of indoor bedrooms. Our baseline comparison is DCGAN [18], a GAN with a convolutional architecture trained with the standard GAN procedure using the $−\log{D}$ trick [4]. The generated samples are 3-channel images of 64x64 pixels in size. We use the hyper-parameters specified in Algorithm 1 for all of our experiments.
>> 우리는 이미지 생성에 대한 실험을 실행한다. 학습할 대상 분포는 실내 침실의 자연 이미지 모음인 LSUN-Bedrooms 데이터 세트 [24]이다. 우리의 기본 비교는 DCGAN[18]으로, $−\log{D}$ 트릭[4]을 사용하여 표준 GAN 절차로 훈련된 컨볼루션 아키텍처를 가진 GAN이다. 생성된 샘플은 64x64픽셀 크기의 3채널 이미지입니다. 우리는 모든 실험을 위해 알고리즘 1에 지정된 하이퍼 파라미터를 사용한다.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)Wasserstein-GAN-translation/Figure-3.JPG)

> Figure 3: Training curves and samples at different stages of training. We can see a clear correlation between lower error and better sample quality. Upper left: the generator is an MLP with 4 hidden layers and 512 units at each layer. The loss decreases constistently as training progresses and sample quality increases. Upper right: the generator is a standard DCGAN. The loss decreases quickly and sample quality increases as well. In both upper plots the critic is a DCGAN without the sigmoid so losses can be subjected to comparison. Lower half: both the generator and the discriminator are MLPs with substantially high learning rates (so training failed). Loss is constant and samples are constant as well. The training curves were passed through a median filter for visualization purposes.
>> 그림 3: 여러 단계의 훈련 곡선 및 샘플 우리는 낮은 오차와 더 나은 샘플 품질 사이의 명확한 상관관계를 볼 수 있다. 왼쪽 위: 제너레이터는 4개의 숨겨진 레이어와 각 레이어에 512개의 유닛이 있는 MLP입니다. 교육이 진행되어 샘플 품질이 증가함에 따라 손실은 지속적으로 감소합니다. 오른쪽 상단: 제너레이터는 표준 DCGAN입니다. 손실은 빠르게 감소하고 샘플 품질도 증가합니다. 두 상위 그림에서 비평가는 시그모이드가 없는 DCGAN이므로 손실을 비교할 수 있다. 하위 절반: 생성기와 판별기 모두 학습률이 상당히 높은 MLP이다(따라서 훈련에 실패함). 손실은 일정하고 표본도 일정합니다. 훈련 곡선은 시각화를 위해 중앙 필터를 통과했다.

### $\mathbf{4.2\;\;\;Meaningful\;loss\;metric}$

> Because the WGAN algorithm attempts to train the critic f (lines 2–8 in Algorithm 1) relatively well before each generator update (line 10 in Algorithm 1), the loss function at this point is an estimate of the EM distance, up to constant factors related to the way we constrain the Lipschitz constant of f.
>> WGAN 알고리듬은 각 발전기 업데이트(알고리즘 1의 라인 10) 전에 상대적으로 잘 임계치(알고리즘 1의 라인 2-8)를 훈련시키려고 시도하기 때문에, 이 지점의 손실 함수는 립시츠 상수를 off로 제한하는 방식과 관련된 상수 요인까지 전자파 거리의 추정치이다.

> Our first experiment illustrates how this estimate correlates well with the quality of the generated samples. Besides the convolutional DCGAN architecture, we also ran experiments where we replace the generator or both the generator and the critic by 4-layer ReLU-MLP with 512 hidden units.
>> 우리의 첫 번째 실험은 이 추정치가 생성된 샘플의 품질과 어떻게 잘 상관되는지 보여준다. 컨볼루션 DCGAN 아키텍처 외에도, 우리는 발전기 또는 발전기와 비평가 모두를 512개의 숨겨진 유닛으로 4계층 ReLU-MLP로 교체하는 실험을 실행했다.

> Figure 3 plots the evolution of the WGAN estimate (3) of the EM distance during WGAN training for all three architectures. The plots clearly show that these curves correlate well with the visual quality of the generated samples.
>> 그림 3은 세 가지 아키텍처 모두에 대한 WGAN 훈련 중 전자파 거리의 WGAN 추정치(3)의 진화를 보여준다. 그림을 보면 이러한 곡선이 생성된 표본의 시각적 품질과 잘 연관되어 있음을 알 수 있습니다.

> To our knowledge, this is the first time in GAN literature that such a property is shown, where the loss of the GAN shows properties of convergence. This property is extremely useful when doing research in adversarial networks as one does not need to stare at the generated samples to figure out failure modes and to gain information on which models are doing better over others.
>> 우리가 아는 한, GAN의 손실이 수렴의 속성을 보여주는 이러한 속성은 GAN 문헌에서 처음 보여진다. 이 속성은 고장 모드를 파악하고 어떤 모델이 다른 모델보다 더 잘하는지 정보를 얻기 위해 생성된 샘플을 응시할 필요가 없기 때문에 적대적 네트워크에서 연구할 때 매우 유용하다.

> However, we do not claim that this is a new method to quantitatively evaluate generative models yet. The constant scaling factor that depends on the critic’s architecture means it’s hard to compare models with different critics. Even more, in practice the fact that the critic doesn’t have infinite capacity makes it hard to know just how close to the EM distance our estimate really is. This being said, we have succesfully used the loss metric to validate our experiments repeatedly and without failure, and we see this as a huge improvement in training GANs which previously had no such facility.
>> 그러나 우리는 이것이 아직 생성 모델을 정량적으로 평가하기 위한 새로운 방법이라고 주장하지는 않는다. 비평가의 아키텍처에 따라 달라지는 일정한 스케일링 계수는 다른 비평가와 모델을 비교하기 어렵다는 것을 의미합니다. 더욱이 실제로 비평가가 무한한 용량을 가지고 있지 않다는 사실은 우리의 추정치가 실제로 전자파 거리에 얼마나 가까운지 알기 어렵게 만든다. 그럼에도 불구하고, 우리는 실패 없이 반복적으로 실험을 검증하기 위해 손실 메트릭을 성공적으로 사용했으며, 우리는 이것이 이전에 그러한 시설이 없었던 GAN을 훈련시키는 데 있어 큰 개선으로 본다.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)Wasserstein-GAN-translation/Figure-4.JPG)

> Figure 4: JS estimates for an MLP generator (upper left) and a DCGAN generator (upper right) trained with the standard GAN procedure. Both had a DCGAN discriminator. Both curves have increasing error. Samples get better for the DCGAN but the JS estimate increases or stays constant, pointing towards no significant correlation between sample quality and loss. Bottom: MLP with both generator and discriminator. The curve goes up and down regardless of sample quality. All training curves were passed through the same median filter as in Figure 3.
>> 그림 4: 표준 GAN 절차로 훈련된 MLP 발생기(왼쪽 위)와 DCGAN 발생기(오른쪽 위)에 대한 JS 추정치. 둘 다 DCGAN 판별기를 가지고 있었습니다. 두 곡선 모두 오차가 증가합니다. 샘플은 DCGAN에 대해 더 나아지지만 JS 추정치는 증가하거나 일정하게 유지되어 샘플 품질과 손실 사이에 유의한 상관관계가 없음을 나타낸다. 하단: 제너레이터 및 판별기가 모두 있는 MLP. 곡선은 표본 품질에 관계없이 오르락내리락합니다. 모든 훈련 곡선은 그림 3과 같은 중앙 필터를 통과했다.

> In contrast, Figure 4 plots the evolution of the GAN estimate of the JS distance during GAN training. More precisely, during GAN training, the discriminator is trained to maximize
>> 대조적으로, 그림 4는 GAN 훈련 중 JS 거리에 대한 GAN 추정치의 진화를 보여준다. 보다 정확하게, GAN 훈련 동안 판별기는 최대화하도록 훈련된다.

$$L(D,g_{\theta}) = E_{x\sim{P_{r}}}[\log{D(x)}]\mid +E_{x\sim{P_{\theta}}}[\log(1-D(x))]$$

> which is is a lower bound of $2JS(P_{r},P_{\theta})−2log{2}$. In the figure, we plot the quantity $\frac{1}{2}L(D, g_{\theta})+log{2}$, which is a lower bound of the JS distance. 
>> 이는 $2JS(P_{r},P_{\theta})−2log{2}$의 하한이다. 그림에서 우리는 JS 거리의 하한인 수량 $\frac{1}{2}L(D, g_{\theta})+log{2}$를 표시한다.

> This quantity clearly correlates poorly the sample quality. Note also that the JS estimate usually stays constant or goes up instead of going down. In fact it often remains very close to log $2\approx{0.69}$ which is the highest value taken by the JS distance. In other words, the JS distance saturates, the discriminator has zero loss, and the generated samples are in some cases meaningful (DCGAN generator, top right plot) and in other cases collapse to a single nonsensical image [4]. This last phenomenon has been theoretically explained in [1] and highlighted in [11]. 
>> 이 수량은 샘플 품질과 상관관계가 좋지 않습니다. 또한 JS 추정치는 일반적으로 일정하게 유지되거나 내려가지 않고 올라간다. 실제로 JS 거리가 취한 가장 높은 값인 로그 $2\approx{0.69}$에 매우 근접하게 유지되는 경우가 많다. 즉, JS 거리는 포화되고 판별기는 손실이 0이며 생성된 샘플은 어떤 경우에는 의미 있고(DCGAN 생성기, 오른쪽 상단 그림), 어떤 경우에는 단일 무의미한 이미지로 붕괴된다[4]. 이 마지막 현상은 이론적으로 [1]에서 설명되었고 [11]에서 강조되었다.

> When using the $− log D$ trick [4], the discriminator loss and the generator loss are different. Figure 8 in Appendix E reports the same plots for GAN training, but using the generator loss instead of the discriminator loss. This does not change the conclusions.
>> $− log D$ 트릭 [4]를 사용할 때 판별기 손실과 발전기 손실은 서로 다릅니다. 부록 E의 그림 8은 GAN 훈련에 대해 동일한 플롯을 보고하지만 판별기 손실 대신 발전기 손실을 사용한다. 이것은 결론을 바꾸지 않는다.

> Finally, as a negative result, we report that WGAN training becomes unstable at times when one uses a momentum based optimizer such as Adam [8] (with $β1 > 0$) on the critic, or when one uses high learning rates. Since the loss for the critic is nonstationary, momentum based methods seemed to perform worse. We identified momentum as a potential cause because, as the loss blew up and samples got worse, the cosine between the Adam step and the gradient usually turned negative. The only places where this cosine was negative was in these situations of instability. We therefore switched to RMSProp [21] which is known to perform well even on very nonstationary problems [13].
>> 마지막으로, 부정적인 결과로, 우리는 WGAN 훈련이 비평가에게 Adam [8]($β1 > 0$ 포함)과 같은 모멘텀 기반 최적기를 사용할 때 또는 높은 학습률을 사용할 때 불안정해진다고 보고한다. 비평가의 손실은 일정하지 않기 때문에, 모멘텀 기반 방법은 더 나쁜 성능을 발휘하는 것처럼 보였다. 손실이 폭발하고 샘플이 악화됨에 따라 Adam 단계와 그라데이션 사이의 코사인(coosine)이 보통 음으로 변하기 때문에 운동량을 잠재적 원인으로 확인했습니다. 이 코사인이 음성이었던 유일한 장소는 이러한 불안정한 상황에서였다. 따라서 매우 비정상적 문제에서도 우수한 성능을 발휘하는 것으로 알려진 RMSProp[21]로 전환했다[13].

### $\mathbf{4.3\;\;\;Improved\;stability}$

> One of the benefits of WGAN is that it allows us to train the critic till optimality. When the critic is trained to completion, it simply provides a loss to the generator that we can train as any other neural network. This tells us that we no longer need to balance generator and discriminator’s capacity properly. The better the critic, the higher quality the gradients we use to train the generator. 
>> WGAN의 이점 중 하나는 비평가를 최적까지 훈련시킬 수 있다는 것이다. 비평가가 완료하도록 훈련될 때, 그것은 단순히 우리가 다른 신경 네트워크처럼 훈련할 수 있는 발전기에 손실을 제공한다. 이는 더 이상 발전기와 판별기 용량의 균형을 제대로 맞출 필요가 없다는 것을 말해준다. 비평가가 좋을수록 발전기를 훈련시키는 데 사용하는 그레이디언트 품질이 높아진다.

> We observe that WGANs are much more robust than GANs when one varies the architectural choices for the generator. We illustrate this by running experiments on three generator architectures: (1) a convolutional DCGAN generator, (2) a convolutional DCGAN generator without batch normalization and with a constant number of filters, and (3) a 4-layer ReLU-MLP with 512 hidden units. The last two are known to perform very poorly with GANs. We keep the convolutional DCGAN architecture for the WGAN critic or the GAN discriminator. 
>> 우리는 WGAN이 발전기에 대한 아키텍처 선택을 변경할 때 GAN보다 훨씬 강력하다는 것을 관찰한다. 우리는 (1) 컨볼루션 DCGAN 생성기, (2) 배치 정규화가 없고 일정한 수의 필터가 있는 컨볼루션 DCGAN 생성기, (3) 512개의 숨겨진 장치가 있는 4층 ReLU-MLP의 세 가지 생성기 아키텍처에 대한 실험을 실행하여 이를 설명한다. 마지막 두 개는 GAN에서 매우 저조한 성능을 발휘하는 것으로 알려져 있다. 우리는 WGAN 비평가 또는 GAN 판별기를 위한 컨볼루션 DCGAN 아키텍처를 유지한다.

> Figures 5, 6, and 7 show samples generated for these three architectures using both the WGAN and GAN algorithms. We refer the reader to Appendix F for full sheets of generated samples. Samples were not cherry-picked.
>> 그림 5, 6, 7은 WGAN 및 GAN 알고리즘을 모두 사용하여 이러한 세 가지 아키텍처에 대해 생성된 샘플을 보여준다. 생성된 샘플의 전체 시트는 부록 F를 참조하십시오. 샘플은 체리 픽이 되지 않았다.

> **In no experiment did we see evidence of mode collapse for the WGAN algorithm.**
>> **어떤 실험에서도 WGAN 알고리듬에 대한 모드 붕괴의 증거를 발견하지 못했다.**

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)Wasserstein-GAN-translation/Figure-5.JPG)

> Figure 5: Algorithms trained with a DCGAN generator. Left: WGAN algorithm. Right: standard GAN formulation. Both algorithms produce high quality samples.
>> 그림 5: DCGAN 발생기로 훈련된 알고리즘 왼쪽: WGAN 알고리즘입니다. 오른쪽: 표준 GAN 공식입니다. 두 알고리듬 모두 고품질 샘플을 생성한다.

![Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)Wasserstein-GAN-translation/Figure-6.JPG)

> Figure 6: Algorithms trained with a generator without batch normalization and constant number of filters at every layer (as opposed to duplicating them every time as in [18]). Aside from taking out batch normalization, the number of parameters is therefore reduced by a bit more than an order of magnitude. Left: WGAN algorithm. Right: standard GAN formulation. As we can see the standard GAN failed to learn while the WGAN still was able to produce samples.
>> 그림 6: 배치 정규화 없이 모든 계층에서 일정한 수의 필터를 사용하는 발전기로 훈련된 알고리즘([18]에서와 같이 매번 복제하는 것과는 반대) 따라서 배치 정규화를 수행하는 것 외에도 매개 변수의 수가 크기 순서보다 약간 더 감소합니다. 왼쪽: WGAN 알고리즘입니다. 오른쪽: 표준 GAN 공식입니다. 우리가 볼 수 있듯이 표준 GAN은 학습에 실패했지만 WGAN은 여전히 샘플을 생산할 수 있었다.

![Figure 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-26-(GAN)Wasserstein-GAN-translation/Figure-7.JPG)

> Figure 7: Algorithms trained with an MLP generator with 4 layers and 512 units with ReLU nonlinearities. The number of parameters is similar to that of a DCGAN, but it lacks a strong inductive bias for image generation. Left: WGAN algorithm. Right: standard GAN formulation. The WGAN method still was able to produce samples, lower quality than the DCGAN, and of higher quality than the MLP of the standard GAN. Note the significant degree of mode collapse in the GAN MLP.
>> 그림 7: ReLU 비선형성을 가진 4개의 레이어와 512개의 유닛을 가진 MLP 발생기로 훈련된 알고리즘 매개 변수의 수는 DCGAN과 유사하지만 이미지 생성을 위한 강력한 유도 편향이 부족하다. 왼쪽: WGAN 알고리즘입니다. 오른쪽: 표준 GAN 공식입니다. WGAN 방법은 DCGAN보다 낮은 품질로 표준 GAN의 MLP보다 높은 품질의 샘플을 여전히 생산할 수 있었다. GAN MLP에서 모드 붕괴의 상당한 정도를 기록한다.

### $\mathbf{5\;\;\;Related\;Work}$

> There’s been a number of works on the so called Integral Probability Metrics (IPMs) [15]. Given $F$ a set of functions from $X$ to $R$, we can define
>> 이른바 적분 확률 메트릭스(IPM)에 대한 많은 연구가 있었다[15]. $X$에서 $R$까지의 함수 집합 $F$가 주어지면, 우리는 정의할 수 있다.

$$d_{F}(P_{r},P_{\theta}) =  \underset{f\in{F}}{\mathrm{sup}}E_{x\sim{P_{r}}}[f(x)]-E_{x\sim P_{\theta}}[f(x)]$$

> as an integral probability metric associated with the function class F. It is easily verified that if for every $f\in{F}$ we have $−f\in{F}$ (such as all examples we’ll consider), then $d_{F}$ is nonnegative, satisfies the triangular inequality, and is symmetric. Thus, $d_{F}$ is a pseudometric over $\mathrm{Prob}(X)$. 
>> 함수 클래스 F와 관련된 적분 확률 메트릭으로 사용됩니다. 만약 우리가 고려할 모든 예와 같이 모든 $f\in{F}$에 대해 $−f\in{F}$가 있다면, $d_{F}$  는 음이 아니며 삼각 부등식을 만족하며 대칭적이라는 것이 쉽게 확인된다. 따라서, $d_{F}$는 $\mathrm{Prob}(X)$에 대한 의사 측정이다.

> While IPMs might seem to share a similar formula, as we will see different classes of functions can yeald to radically different metrics.
>> IPM은 유사한 공식을 공유하는 것처럼 보일 수 있지만, 서로 다른 종류의 함수가 근본적으로 다른 메트릭을 나타낼 수 있음을 알 수 있다.

> * By the Kantorovich-Rubinstein duality [22], we know that $W(P_{r}, P_{\theta}) = d_{F} (P_{r}, P_{\theta})$ when $F$ is the set of 1-Lipschitz functions. Furthermore, if "F" is the set of KLipschitz functions, we get $K · W(P_{r}, P_{\theta}) = d_{F} (P_{r}, P_{\theta})$.
>>  * 칸토로비치-루빈슈타인 이중성[22]에 의해, 우리는 $F$가 1-립시츠 함수의 집합일 때 $W(P_{r}, P_{\theta}) = d_{F} (P_{r}, P_{\theta})$가 된다는 것을 안다. 게다가, 만약 "F"가 KLIPSchitz 함수의 집합이라면, 우리는 $K · W(P_{r}, P_{\theta}) = d_{F} (P_{r}, P_{\theta})$를 얻는다.
> * When $F$ is the set of all measurable functions bounded between -1 and 1 (or all continuous functions between -1 and 1), we retrieve $d_{F} (P_{r}, P_{\theta}) = \delta(P_{r}, P_{\theta})$ the total variation distance [15]. This already tells us that going from 1-Lipschitz to 1-Bounded functions drastically changes the topology of the space, and the regularity of $d_{F} (P_{r}, P_{\theta})$ as a loss function (as by Theorems 1 and 2).
>>  * $F$가 -1과 1 사이의 모든 측정 가능한 함수(또는 -1과 1 사이의 모든 연속 함수)의 집합일 때, 우리는 $d_{F} (P_{r}, P_{\theta}) = \delta(P_{r}, P_{\theta})$의 총 변동 거리[15]를 검색한다. 이것은 이미 1-립시츠 함수에서 1-경계 함수로 가는 것이 공간의 위상과 손실 함수로서 $d_{F} (P_{r}, P_{\theta})$의 규칙성을 극적으로 변화시킨다는 것을 말해준다.
> * Energy-based GANs (EBGANs) [25] can be thought of as the generative approach to the total variation distance. This connection is stated and proven in depth in Appendix $D$. At the core of the connection is that the discriminator will play the role of f maximizing equation (4) while its only restriction is being between 0 and $m$ for some constant m. This will yeald the same behaviour as being restricted to be between −1 and 1 up to a constant scaling factor irrelevant to optimization. Thus, when the discriminator approaches optimality the cost for the generator will aproximate the total variation distance $\delta(P_{r}, P_{\theta})$. Since the total variation distance displays the same regularity as the JS, it can be seen that EBGANs will suffer from the same problems of classical GANs regarding not being able to train the discriminator till optimality and thus limiting itself to very imperfect gradients.
>>  * 에너지 기반 GAN(EBGANs) [25]은 총 변동 거리에 대한 생성 접근법으로 생각할 수 있다. 이 연결은 부록 $D$에 자세히 설명되어 있고 입증되었다. 연결의 핵심은 판별기가 일부 상수 m에 대해 0과 $m$ 사이에 있는 동안 방정식을 최대화하는 f의 역할을 한다는 것이다. 이는 최적화와 무관한 일정한 스케일링 계수까지 -1과 1 사이로 제한되는 것과 동일한 동작을 의미한다. 따라서 판별기가 최적성에 접근할 때, 발전기의 비용은 총 변동 거리 $\delta(P_{r}, P_{\theta})$에 근사할 것이다. 총 변동 거리는 JS와 동일한 규칙성을 표시하므로, EBGAN이 최적성까지 판별기를 훈련시킬 수 없어 매우 불완전한 그레이디언트로 제한되는 것과 관련하여 고전적인 GAN의 동일한 문제를 겪을 것임을 알 수 있다.
> * Maximum Mean Discrepancy (MMD) [5] is a specific case of integral probability metrics when $F = {f \in H : \parallel{f}\parallel_{\infty} ≤ 1}$ for H some Reproducing Kernel Hilbert Space (RKHS) associated with a given kernel $k : X × X\to{R}$. As proved on [5] we know that MMD is a proper metric and not only a pseudometric when the kernel is universal. In the specific case where $H = L ^{2} (X , m)$ for m the normalized Lebesgue measure on $X$ , we know that ${f \in C_{b}(X ), \parallel{f}\parallel_{\infty} ≤ 1}$ will be contained in F, and therefore $d_{F} (P_{r}, P_{\theta}) ≤ \delta(P_{r}, P_{\theta})$ so the regularity of the MMD distance as a loss function will be at least as bad as the one of the total variation. Nevertheless this is a very extreme case, since we would need a very powerful kernel to approximate the whole $L^{2}$ . However, even Gaussian kernels are able to detect tiny noise patterns as recently evidenced by [20]. This points to the fact that especially with low bandwidth kernels, the distance might be close to a saturating regime similar as with total variation or the JS. This obviously doesn’t need to be the case for every kernel, and figuring out how and which different MMDs are closer to Wasserstein or total variation distances is an interesting topic of research. The great aspect of MMD is that via the kernel trick there is no need to train a separate network to maximize equation (4) for the ball of a RKHS. However, this has the disadvantage that evaluating the MMD distance has computational cost that grows quadratically with the amount of samples used to estimate the expectations in (4). This last point makes MMD have limited scalability, and is sometimes inapplicable to many real life applications because of it. There are estimates with linear computational cost for the MMD [5] which in a lot of cases makes MMD very useful, but they also have worse sample complexity.
>> 최대 평균 불일치 (MMD) [5]는 주어진 커널 $H = L ^{2} (X , m)$와 연관된 Home Reproducting 커널 힐베르트 공간 (RKHS)에 대해 $k : X × X\to{R}$일 때 적분 확률 메트릭의 특정한 경우이다. [5]에서 증명된 바와 같이, 우리는 MMD가 적절한 메트릭이며 커널이 보편적일 때 의사 메트릭만이 아니라는 것을 안다. ${f \in C_{b}(X ), \parallel{f}\parallel_{\infty} ≤ 1}$가 $X$에 대해 정규화된 르베그 측정을 형성하는 특정한 경우, 우리는 $d_{F} (P_{r}, P_{\theta}) ≤ \delta(P_{r}, P_{\theta})$가 F에 포함된다는 것을 알고, 따라서 손실 함수로서의 MMD 거리의 규칙성은 적어도 전체 변동 중 하나만큼 나쁠 것이다. 그럼에도 불구하고 전체 $L^{2}$를 근사화하기 위해 매우 강력한 커널이 필요하기 때문에 이것은 매우 극단적인 경우이다. 그러나 가우스 커널조차도 최근 [20]에서 증명된 것처럼 작은 노이즈 패턴을 감지할 수 있다. 이는 특히 낮은 대역폭 커널에서 거리가 총 변동 또는 JS와 유사한 포화 상태에 가까울 수 있다는 사실을 가리킨다. 이것은 분명히 모든 커널에 해당될 필요는 없으며, 어떻게 그리고 어떤 다른 MMD가 와서스테인 또는 총 변동 거리에 더 가까운지를 알아내는 것은 흥미로운 연구 주제이다. MMD의 가장 큰 측면은 커널 트릭을 통해 RKHS의 공에 대한 방정식(4)을 최대화하기 위해 별도의 네트워크를 훈련시킬 필요가 없다는 것이다. 그러나 이는 MMD 거리를 평가하는 데 (4)의 기대치를 추정하는 데 사용되는 샘플의 양에 따라 2차적으로 증가하는 계산 비용이 있다는 단점이 있다. 이 마지막 포인트는 MMD를 제한된 확장성을 가지도록 만들고, MMD 때문에 많은 실제 응용 프로그램에 적용할 수 없게 만든다. MMD에 대한 선형 계산 비용이 포함된 추정치[5]가 있는데, 이는 많은 경우 MMD를 매우 유용하게 만들지만 표본 복잡성이 더 나쁘다.

> * Generative Moment Matching Networks (GMMNs) [10, 2] are the generative counterpart of MMD. By backproping through the kernelized formula for equation (4), they directly optimize $d_{MMD}(P_{r}, P_{\theta})$  (the IPM when $F$ is as in the previous item). As mentioned, this has the advantage of not requiring a separate network to approximately maximize equation (4). However, GMMNs have enjoyed limited  applicability. Partial explanations for their unsuccess are the quadratic cost as a function of the number of samples and vanishing gradients for low-bandwidth kernels. Furthermore, it may be possible that some kernels used in practice are unsuitable for capturing very complex distances in high dimensional sample spaces such as natural images. This is properly justified by the fact that [19] shows that for the typical Gaussian MMD test to be reliable (as in it’s power as a statistical test approaching 1), we need the number of samples to grow linearly with the number of dimensions. Since the MMD computational cost grows quadratically with the number of samples in the batch used to estimate equation (4), this makes the cost of having a reliable estimator grow quadratically with the number of dimensions, which makes it very inapplicable for high dimensional problems. Indeed, for something as standard as 64x64 images, we would need minibatches of size at least 4096 (without taking into account the constants in the bounds of [19] which would make this number substantially larger) and a total cost per iteration of 40962 , over 5 orders of magnitude more than a GAN iteration when using the standard batch size of 64. That being said, these numbers can be a bit unfair to the MMD, in the sense that we are comparing empirical sample complexity of GANs with the theoretical sample complexity of MMDs, which tends to be worse. However, in the original GMMN paper [10] they indeed used a minibatch of size 1000, much larger than the standard 32 or 64 (even when this incurred in quadratic computational cost). While estimates that have linear computational cost as a function of the number of samples exist [5], they have worse sample complexity, and to the best of our knowledge they haven’t been yet applied in a generative context such as in GMMNs.
>> 생성 모멘트 매칭 네트워크(GMMN)[10, 2]는 MMD의 생성 대응물이다. 방정식(4)에 대한 커널화된 공식을 통해 백프로핑함으로써 "A"($F$가 이전 항목과 같은 경우 IPM)를 직접 최적화한다. 전술한 바와 같이, 이는 방정식(4)을 대략적으로 최대화하기 위해 별도의 네트워크가 필요하지 않다는 장점이 있다. 그러나 GMMN은 제한된 적용성을 누려왔다. 실패에 대한 부분적인 설명은 낮은 대역폭 커널에 대한 샘플 수와 사라지는 그레이디언트의 함수로서의 2차 비용이다. 또한 실제로 사용되는 일부 커널은 자연 이미지와 같은 고차원 샘플 공간에서 매우 복잡한 거리를 캡처하는 데 적합하지 않을 수 있다. 이는 [19]가 일반적인 가우스 MMD 테스트가 신뢰할 수 있으려면(1에 가까운 통계 테스트로서의 검정력에서와 같이) 표본 수가 차원 수에 따라 선형적으로 증가해야 한다는 사실을 보여준다는 사실에 의해 적절히 정당화된다. MMD 계산 비용은 방정식(4)을 추정하는 데 사용되는 배치의 샘플 수에 따라 2차적으로 증가하므로, 신뢰할 수 있는 추정기를 갖는 비용은 차원 수에 따라 2차적으로 증가하므로 고차원 문제에 매우 적용되지 않는다. 실제로 64x64 이미지와 같은 표준 이미지의 경우 최소 4096 크기의 미니 배치(이 숫자를 상당히 크게 만드는 [19]의 경계 상수를 고려하지 않음)와 40962의 반복당 총 비용이 필요합니다. 이는 6의 표준 배치 크기를 사용할 때 GAN 반복보다 5배 이상 큰 크기입니다.4. 그렇긴 하지만, 이러한 수치는 우리가 GAN의 경험적 샘플 복잡성을 MMD의 이론적 샘플 복잡성과 비교한다는 점에서 MMD에게 다소 불공평할 수 있다. 그러나 원래 GMMN 논문[10]에서 그들은 표준 32나 64보다 훨씬 큰 1000 크기의 미니 배치를 실제로 사용했다. (이것이 2차 계산 비용으로 발생한 경우에도 마찬가지였다.) 샘플 수의 함수로 선형 계산 비용을 갖는 추정치는 [5] 존재하지만, 샘플 복잡성이 더 나쁘고, 우리가 아는 한 GMMN과 같은 생성 컨텍스트에는 아직 적용되지 않았다.

> On another great line of research, the recent work of [14] has explored the use of Wasserstein distances in the context of learning for Restricted Boltzmann Machines for discrete spaces. The motivations at a first glance might seem quite different, since the manifold setting is restricted to continuous spaces and in finite discrete spaces the weak and strong topologies (the ones of W and JS respectively) coincide. However, in the end there is more in commmon than not about our motivations. We both want to compare distributions in a way that leverages the geometry of the underlying space, and Wasserstein allows us to do exactly that.
>> 또 다른 훌륭한 연구 라인에서, [14]의 최근 연구는 이산 공간에 대한 제한된 볼츠만 기계에 대한 학습의 맥락에서 와서스테인 거리의 사용을 탐구했다. 다양체 설정이 연속적인 공간과 무한 이산 공간에서 약한 위상 및 강한 위상(각각 W와 JS의 위상)이 일치하기 때문에 언뜻 보기에는 상당히 다르게 보일 수 있다. 하지만 결국 우리의 동기보다는 공통점이 더 많다. 우리는 둘 다 기초 공간의 기하학적 구조를 활용하는 방식으로 분포를 비교하고 싶어합니다. 그리고 와서스테인은 우리가 정확히 그렇게 할 수 있도록 해줍니다.

> Finally, the work of [3] shows new algorithms for calculating Wasserstein distances between  different distributions. We believe this direction is quite important, and perhaps could lead to new ways of evaluating generative models.
>> 마지막으로, [3]의 작업은 서로 다른 분포 사이의 와서스테인 거리를 계산하기 위한 새로운 알고리듬을 보여준다. 우리는 이 방향이 상당히 중요하며, 아마도 생성 모델을 평가하는 새로운 방법으로 이어질 수 있다고 믿는다.

### $\mathbf{6\;\;\;Conclusion}$

> We introduced an algorithm that we deemed WGAN, an alternative to traditional GAN training. In this new model, we showed that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter  searches. Furthermore, we showed that the corresponding optimization problem is sound, and provided extensive theoretical work highlighting the deep connections to other distances between distributions.
>> 우리는 전통적인 GAN 훈련의 대안인 WGAN으로 간주되는 알고리듬을 도입했다. 이 새로운 모델에서, 우리는 학습의 안정성을 향상시키고, 모드 붕괴와 같은 문제를 제거하고, 디버깅 및 하이퍼 파라미터 검색에 유용한 의미 있는 학습 곡선을 제공할 수 있음을 보여주었다. 또한, 우리는 해당 최적화 문제가 건전하다는 것을 보여주었고, 분포 사이의 다른 거리에 대한 깊은 연결을 강조하는 광범위한 이론적 작업을 제공했다.

### $\mathbf{Acknowledgments}$

> We would like to thank Mohamed Ishmael Belghazi, Emily Denton, Ian Goodfellow, Ishaan Gulrajani, Alex Lamb, David Lopez-Paz, Eric Martin, Maxime Oquab, Aditya Ramesh, Ronan Riochet, Uri Shalit, Pablo Sprechmann, Arthur Szlam, Ruohan Wang, for helpful comments and advice.
>> 모하메드 이스마엘 벨가지, 에밀리 덴튼, 이안 굿펠로, 이샨 굴라자니, 알렉스 램, 데이비드 로페즈-파즈, 에릭 마틴, 맥심 오캅, 아디트 라메시, 로난 리오체, 우리 샬리트, 파블로 스프레흐만, 루한, 그리고 도움이 되는 조언에 대해 감사드립니다.