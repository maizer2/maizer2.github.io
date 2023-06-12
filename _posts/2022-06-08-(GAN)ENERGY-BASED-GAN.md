---
layout: post 
title: "(GAN)ENERGY-BASED GENERATIVE ADVERSARIAL NETWORKS Translation"
categories: [1. Computer Engineering]
tags: [1.7. Paper Review, 1.2.2.5. GAN]
---

### [GAN Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Paper-of-GAN.html)

## $$\mathbf{ENERGY-BASED\;GENERATIVE\;ADVERSARIAL\;NETWORKS}$$

### $$\mathbf{Abstract}$$

> We introduce the “Energy-based Generative Adversarial Network” model (EBGAN) which views the discriminator as an energy function that attributes low energies to the regions near the data manifold and higher energies to other regions. Similar to the probabilistic GANs, a generator is seen as being trained to produce contrastive samples with minimal energies, while the discriminator is trained to assign high energies to these generated samples. Viewing the discriminator as an energy function allows to use a wide variety of architectures and loss  functionals in addition to the usual binary classifier with logistic output. Among them, we show one  instantiation of EBGAN framework as using an auto-encoder architecture, with the energy being the reconstruction error, in place of the discriminator. We show that this form of EBGAN exhibits more stable behavior than regular GANs during training. We also show that a single-scale architecture can be trained to generate high-resolution images.
>> 우리는 판별기를 데이터 매니폴드 근처의 영역에는 낮은 에너지를, 다른 영역에는 높은 에너지를 귀속시키는 에너지 함수로 보는 "에너지 기반 생성 적대적 네트워크" 모델(EBGAN)을 소개한다. 확률적 GAN과 유사하게, 발전기는 최소 에너지로 대조 샘플을 생성하도록 훈련되는 반면, 판별기는 이러한 생성된 샘플에 높은 에너지를 할당하도록 훈련된다. 판별기를 에너지 함수로 보면 로지스틱 출력이 있는 일반적인 이진 분류기 외에도 다양한 아키텍처와 손실 함수를 사용할 수 있다. 그 중, 우리는 판별기 대신 에너지가 재구성 오류인 자동 인코더 아키텍처를 사용하는 EBGAN 프레임워크의 한 가지 인스턴스화를 보여준다. 우리는 이러한 형태의 EBGAN이 훈련 중에 일반 GAN보다 더 안정적인 행동을 보인다는 것을 보여준다. 우리는 또한 고해상도 이미지를 생성하도록 단일 스케일 아키텍처를 훈련시킬 수 있음을 보여준다.

### $\mathbf{1\;Introduction}$

#### $\mathbf{1.1\;ENERGY-BASED\;MODEL}$

> The essence of the energy-based model (LeCun et al., 2006) is to build a function that maps each point of an input space to a single scalar, which is called “energy”. The learning phase is a datadriven process that shapes the energy surface in such a way that the desired configurations get assigned low energies, while the incorrect ones are given high energies. Supervised learning falls into this framework: for each $X$ in the training set, the energy of the pair $(X, Y)$ takes low values when $Y$ is the correct label and higher values for incorrect $Y$’s. Similarly, when modeling $X$ alone within an unsupervised learning setting, lower energy is attributed to the data manifold. The term contrastive sample is often used to refer to a data point causing an energy pull-up, such as the incorrect $Y$’s in supervised learning and points from low data density regions in unsupervised learning.
>> 에너지 기반 모델(LeCun et al., 2006)의 본질은 입력 공간의 각 점을 단일 스칼라에 매핑하는 함수를 구축하는 것인데, 이를 "에너지"라고 한다. 학습 단계는 원하는 구성에 낮은 에너지가 할당되고 잘못된 구성에 높은 에너지가 할당되는 방식으로 에너지 표면을 형성하는 데이터 기반 프로세스이다. 지도 학습은 이 프레임워크에 속한다. 훈련 세트의 각 $X$에 대해 $Y$가 올바른 레이블일 때 쌍 $(X, Y)$의 에너지는 낮은 값을 취하고 잘못된 $Y$의 경우 더 높은 값을 취한다. 마찬가지로, 감독되지 않은 학습 환경에서 X를 단독으로 모델링할 때, 낮은 에너지는 데이터 매니폴드에 기인한다. 대조 샘플이라는 용어는 종종 지도 학습의 잘못된 $Y$와 비지도 학습의 낮은 데이터 밀도 영역의 포인트와 같이 에너지 풀업을 유발하는 데이터 포인트를 지칭하는 데 사용된다.

#### $\mathbf{1.2\;GENERATIVE\;ADVERSARIAL\;NETWORKS}$

> Generative Adversarial Networks (GAN) (Goodfellow et al., 2014) have led to significant improvements in image generation (Denton et al., 2015; Radford et al., 2015; Im et al., 2016; Salimans et al., 2016), video prediction (Mathieu et al., 2015) and a number of other domains. The basic idea of GAN is to simultaneously train a discriminator and a generator. The discriminator is trained to distinguish real samples of a dataset from fake samples produced by the generator. The generator uses input from an easy-to-sample random source, and is trained to produce fake samples that the discriminator cannot distinguish from real data samples. During training, the generator receives the gradient of the output of the discriminator with respect to the fake sample. In the original formulation of GAN in Goodfellow et al. (2014), the discriminator produces a probability and, under certain conditions, convergence occurs when the distribution produced by the generator matches the data distribution. From a game theory point of view, the convergence of a GAN is reached when the generator and the discriminator reach a Nash equilibrium.
>> 생성적 적대 신경망(GAN, Goodfellow et al., 2014)은 이미지 생성(Denton et al., 2015; Radford et al., 2015; Im et al., 2016; Saliman et al., 2016), 비디오 예측(Mathieu et al., 2015) 및 기타 여러 도메인의 상당한 개선을 이끌었다. GAN의 기본 아이디어는 판별기와 발전기를 동시에 훈련시키는 것이다. 판별기는 데이터 세트의 실제 샘플을 생성기에 의해 생성된 가짜 샘플과 구별하도록 훈련된다. 생성기는 표본 추출이 쉬운 랜덤 소스의 입력을 사용하며 판별기가 실제 데이터 샘플과 구별할 수 없는 가짜 샘플을 생성하도록 훈련된다. 훈련 중에 생성기는 가짜 샘플에 대한 판별기 출력의 기울기를 수신한다. Goodfellow et al. (2014)의 GAN의 원래 공식에서 판별기는 확률을 생성하고 특정 조건에서 생성기에 의해 생성된 분포가 데이터 분포와 일치할 때 수렴이 발생한다. 게임 이론의 관점에서, GAN의 수렴은 생성자와 판별기가 내쉬 평형에 도달할 때 도달한다.

#### $\mathbf{1.3\;ENERGY-BASED\;GENERATIVE\;ADVERSARIAL\;NETWORKS}$

> In this work, we propose to view the discriminator as an energy function (or a contrast function) without explicit probabilistic interpretation. The energy function computed by the discriminator can be viewed as a trainable cost function for the generator. The discriminator is trained to assign low energy values to the regions of high data density, and higher energy values outside these regions. Conversely, the generator can be viewed as a trainable parameterized function that produces samples in regions of the space to which the discriminator assigns low energy. While it is often possible to convert energies into probabilities through a Gibbs distribution (LeCun et al., 2006), the absence of normalization in this energy-based form of GAN provides greater flexibility in the choice of architecture of the discriminator and the training procedure.
>> 본 연구에서, 우리는 판별기를 명시적 확률론적 해석 없이 에너지 함수(또는 대조 함수)로 볼 것을 제안한다. 판별기에 의해 계산된 에너지 함수는 발전기에 대한 훈련 가능한 비용 함수로 볼 수 있다. 판별기는 데이터 밀도가 높은 영역에 낮은 에너지 값을 할당하고 이러한 영역 외부에 더 높은 에너지 값을 할당하도록 훈련된다. 반대로, 발전기는 판별기가 낮은 에너지를 할당하는 공간의 영역에서 샘플을 생성하는 훈련 가능한 매개 변수화된 함수로 볼 수 있다. 깁스 분포(LeCun et al., 2006)를 통해 에너지를 확률로 변환하는 것이 종종 가능하지만, 이 에너지 기반 형태의 GAN에서 정규화의 부재는 판별기의 아키텍처 선택과 훈련 절차에서 더 큰 유연성을 제공한다.

> The probabilistic binary discriminator in the original formulation of GAN can be seen as one way among many to define the contrast function and loss functional, as described in LeCun et al. (2006) for the supervised and weakly supervised settings, and Ranzato et al. (2007) for unsupervised learning. We experimentally demonstrate this concept, in the setting where the discriminator is an autoencoder architecture, and the energy is the reconstruction error. More details of the interpretation of EBGAN are provided in the appendix B.
>> GAN의 원래 공식에서 확률적 이진 판별기는 LeCun 등에 설명된 바와 같이 대비 함수와 손실 함수를 정의하는 많은 방법 중 하나로 볼 수 있다. (2006)은 감독 및 약 감독 설정에 대해, 그리고 Ranzato 외. (2007년) 비지도 학습. 판별기가 자동 인코더 아키텍처이고 에너지가 재구성 오류인 설정에서 이 개념을 실험적으로 입증한다. EBGAN 해석에 대한 자세한 내용은 부록 B에 수록되어 있다.

> Our main contributions are summarized as follows:
    >> 우리의 주요 기여는 다음과 같이 요약된다.

* > An energy-based formulation for generative adversarial training.
    >> 생성적 적대 훈련을 위한 에너지 기반 공식.

* > A proof that under a simple hinge loss, when the system reaches convergence, the generator of EBGAN produces points that follow the underlying data distribution.
    >> 간단한 힌지 손실 하에서 시스템이 수렴에 도달하면 EBGAN의 생성자가 기본 데이터 분포를 따르는 점을 생성한다는 증거.

* > An EBGAN framework with the discriminator using an auto-encoder architecture in which the energy is the reconstruction error.
    >> 에너지가 재구성 오류인 자동 인코더 아키텍처를 사용하는 판별기가 있는 EBGAN 프레임워크.

* > A set of systematic experiments to explore hyper-parameters and architectural choices that produce good result for both EBGANs and probabilistic GANs.
    >> EBGAN과 확률적 GAN 모두에 대해 좋은 결과를 생성하는 하이퍼 파라미터와 아키텍처 선택을 탐색하기 위한 일련의 체계적인 실험

* > A demonstration that EBGAN framework can be used to generate reasonable-looking highresolution images from the ImageNet dataset at 256×256 pixel resolution, without a multiscale approach.
    >> EBGAN 프레임워크를 사용하여 멀티스케일 접근 방식 없이 256×256 픽셀 해상도의 ImageNet 데이터 세트에서 합리적으로 보이는 고해상도 이미지를 생성할 수 있음을 입증한다.

### $\mathbf{2\;THE\;EBGAN\;MODEL}$

> Let $p_{data}$ be the underlying probability density of the distribution that produces the dataset. The generator $G$ is trained to produce a sample $G(z)$, for instance an image, from a random vector $z$, which is sampled from a known distribution $p_{z}$, for instance $N(0, 1)$. The discriminator $D$ takes either real or generated images, and estimates the energy value $E ∈ R$ accordingly, as explained later. For simplicity, we assume that $D$ produces non-negative values, but the analysis would hold as long as the values are bounded below.
>> $p_{data}$를 데이터 세트를 생성하는 분포의 기본 확률 밀도로 설정한다. 생성기 $G$는 예를 들어 $N(0, 1)$과 같이 알려진 분포 $p_{z}$에서 샘플링되는 랜덤 벡터 $z$에서 샘플 $G(z)$를 생성하도록 훈련된다. 판별기 $D$는 실제 또는 생성된 이미지를 취하고, 나중에 설명되는 바와 같이 그에 따라 에너지 값 $E $ R$을 추정한다. 단순성을 위해 $D$는 음이 아닌 값을 생성한다고 가정하지만, 값이 아래에 제한되는 한 분석은 유지된다.

#### $\mathbf{2.1\;OBJECTIVE\;FUNCTIONAL}$

> The output of the discriminator goes through an objective functional in order to shape the energy function, attributing low energy to the real data samples and higher energy to the generated (“fake”) ones. In this work, we use a margin loss, but many other choices are possible as explained in LeCun et al. (2006). Similarly to what has been done with the probabilistic GAN (Goodfellow et al., 2014), we use a two different losses, one to train $D$ and the other to train $G$, in order to get better quality gradients when the generator is far from convergence.
>> 판별기의 출력은 에너지 함수를 형성하기 위해 목적 함수를 거치며, 낮은 에너지는 실제 데이터 샘플에, 더 높은 에너지는 생성된("가짜") 데이터에 귀속된다. 이 작업에서는 마진 손실을 사용하지만, LeCun 등에서 설명한 것처럼 다른 많은 선택이 가능하다. (2006). 확률론적 GAN(Goodfellow et al., 2014)에서 수행한 것과 유사하게, 우리는 생성기가 수렴과 거리가 먼 경우 품질 그레이디언트를 더 잘 얻기 위해 $D$를 훈련시키는 두 가지 다른 손실을 사용한다.

> Given a positive margin $m$, a data sample $x$ and a generated sample $G(z)$, the discriminator loss $L_{D}$ and the generator loss $L_{G}$ are formally defined by:
>> 양의 여유 $m$, 데이터 샘플 $x$ 및 생성된 샘플 $G(z)$가 주어지면 판별기 손실 $L_{D}$ 및 생성기 손실 $L_{G}$는 다음과 같이 공식 정의된다.

$$L_{D}(x,z)=D(x)+[m-D(G(z))]^{+}$$

$$L_{G}(z)=D(G(z))\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$$

> where $[·]^{+} = \max(0, \cdot)$. Minimizing $L_{G}$ with respect to the parameters of $G$ is similar to maximizing the second term of $L_{D}$ . It has the same minimum but non-zero gradients when $D(G(z)) ≥ m$.
>> $[·]^{+} = \max(0, \cdot)$에서 $G$의 매개 변수와 관련하여 $L_{G}$를 최소화하는 것은 $L_{D}$의 두 번째 항을 최대화하는 것과 유사하다. $D(G(z) $ m$일 때 최소는 같지만 0이 아닌 그레이디언트를 갖는다.

#### $\mathbf{2.2\;OPTIMALITY\;OF\;THE\;SOLUTION}$

> In this section, we present a theoretical analysis of the system presented in section 2.1. We show that if the system reaches a Nash equilibrium, then the generator $G$ produces samples that are indistinguishable from the distribution of the dataset. This section is done in a non-parametric setting, i.e. we assume that $D$ and $G$ have infinite capacity.
>> 이 섹션에서는 2.1절에 제시된 시스템의 이론적 분석을 제시한다. 시스템이 내시 평형에 도달하면 생성기 $G$가 데이터 세트의 분포와 구별할 수 없는 샘플을 생성한다는 것을 보여준다. 이 섹션은 비모수 설정에서 수행된다. 즉, $D$와 $G$의 용량이 무한하다고 가정한다.

> Given a generator $G$, let $p_{G}$ be the density distribution of $G(z)$ where $z ∼ p_{z}$. In other words, $p_{G}$ is the density distribution of the samples generated by $G$. We define $V(G,D)=\int_{x,z}L_{D}(x, z)p_{data}(x)p_{z}(z)dxdz$ and $U(G,D)=\int_{z}L_{G}(z)p_{z}(z)dz$. We train the discriminator $D$ to minimize the quantity $V$ and the generator $G$ to minimize the quantity $U$. A Nash equilibrium of the system is a pair $(G^{\ast} ,D^{\ast} )$ that satisfies:
>> 생성기 $G$가 주어지면 $p_{G}$를 $G(z)$의 밀도 분포로 한다. 여기서 $z ~ p_{z}$는 즉, $p_{G}$는 $G$에 의해 생성된 샘플의 밀도 분포이다. 우리는 $V(G,D)=\int_{x,z}L_{D}(x, z)p_{data}(x)p_{z}(z)dxdz$와 $U(G,D)=\int_{z}L_{G}(z)p_{z}(z)dz$를 정의한다. 우리는 판별기 $D$를 훈련하여 수량 $V$를 최소화하고 생성기 $G$를 훈련하여 수량 $U$를 최소화한다. 시스템의 내시 평형은 다음을 만족시키는 쌍 $(G^{\ast},D^{\ast})$이다.

$$V(G^{\ast} ,D^{\ast} )\leq{V(G^{\ast} ,D)}\;\;\;\;\;\;\;\forall{D}$$

$$U(G^{\ast} ,D^{\ast} )\leq{U(G,D^{\ast} )}\;\;\;\;\;\;\;\forall{G}$$

> **Theorem 1.**  If $(D^{\ast} , G^{\ast} )$ is a Nash equilibrium of the system, then $p_{G}\ast  = p_{data}$ almost everywhere, and $V(D^{\ast} , G^{\ast} ) = m$.
>> **정식 1.**  $(D^{\ast}, G^{\ast})$가 시스템의 내쉬 평형이라면, 거의 모든 곳에서 $p_{G}\ast = p_{data}$이고 $V(D^{\ast}, G^{\ast}) = m$이다.

Proof. First we observe that

$$V(G^{\ast} ,D) = \int_{x}{P_{data}(x)D(x)dx}+\int_{x}{P_{z}[z](m-D(G^{\ast} (z)))}^{+}dz$$

$$=\int_{x}({P_{data}(x)D(x)}+{P_{G^{\ast} }[x](m-D(x))}^{+})dx$$

> The analysis of the function $ϕ(y) = ay+b(m−y)^{+}$ (see lemma 1 in appendix A for details) shows: (a) $D^{\ast} (x)\leq{m}$ almost everywhere. To verify it, let us assume that there exists a set of measure non-zero such that $D^{\ast} (x)>m$. Let $\tilde{D}(x) = \min{(D^{\ast} (x), m)}$. Then $V(G^{\ast} ,\tilde{D}) < V (G^{\ast} ,D^{\ast} )$ which violates equation 3.
>> 함수 A의 분석(자세한 내용은 부록 $ϕ(y) = ay+b(m−y)^{+}$의 요약 1 참조)을 보면 다음과 같다: (a) 거의 모든 곳에서 $D^{\ast} (x)\leq{m}$. 이를 검증하기 위해, $D^{\ast} (x)>m$와 같이 0이 아닌 측정값의 집합이 존재한다고 가정하자. $\tilde{D}(x) = \min{(D^{\ast} (x), m)}$. 그렇다면 방정식 3을 위반하는 $V(G^{\ast} ,\tilde{D}) < V (G^{\ast} ,D^{\ast} )$를 놓아라.

> (b) The function $ϕ$ reaches its minimum in $m$ if $a < b$ and in 0 otherwise. So $V (G^{\ast}  , \tilde{D})$ reaches its minimum when we replace $D^{\ast} (x)$ by these values. We obtain
>(b) 함수 $α$는 $a < b$이면 $m$에서, 그렇지 않으면 0에서 최소값에 도달한다. 따라서 $V(G^{\ast}, \tilde{D})$는 $D^{\ast}(x)$를 이러한 값으로 대체할 때 최소값에 도달한다. 우리는 얻는다

$$V(G^{\ast} ,D^{\ast} ) = m\int_{x}One_{P_{data}<P_{G^{\ast}}(x)}P_{data}(x)dx+m\int_{x}{One_{P_{data}(x)\geq{P_{G^{\ast} }(x)}P_{G^{\ast} }}(x)}dx$$

$$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;= m\int_{x}({One_{P_{data}<P_{G^{\ast} }(x)}P_{data}(x)}+(1-One_{P_{data}(x)\geq{P_{G^{\ast} }(x)}})P_{G^{\ast} }(x))dx$$

$$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;= m\int_{x}P_{G^{\ast} }(x)dx+m\int_{x}{One_{P_{data}<P_{G^{\ast} }(x)}P_{data}(x)-P_{G^{\ast} }(x))}dx$$

$$=m+m\int_{x}{One_{P_{data}<P_{G^{\ast} }(x)}P_{data}(x)-P_{G^{\ast} }(x))}dx\;\;\;\;$$

> The second term in equation 10 is non-positive, so $V (G^{\ast} ,D^{\ast} ) ≤ m$. By putting the ideal generator that generates $p_{data}$ into the right side of equation 4, we get
>> 방정식 10의 두 번째 항은 양수이므로 $V (G^{\ast} ,D^{\ast} ) ≤ m$입니다. $p_{data}$를 생성하는 이상적인 생성자를 방정식 4의 오른쪽에 놓음으로써, 우리는 다음을 얻는다.

$$\int_{x}P_{G^{\ast} }(x)D^{\ast} (x)dx\leq{\int_{x}P_{data}(x)D^{\ast} (x)}dx.$$

Thus by (6),

$$\int_{x}P_{G^{\ast} }(x)D^{\ast} (x)dx+\int_{x}P_{G^{\ast} }[x](m-D^{\ast} (x))^{+}dx\leq{V(G^{\ast} ,D^{\ast} )}$$

> and since $D^{\ast} (x) ≤ m$, we get $m \leq{V (G^{\ast} ,D^{\ast} )}$.
>> 그리고 $D^{\ast}(x) m m$이므로 $m \leq{V(G^{\ast},D^{\ast})}$를 얻는다.

> Thus, $m ≤ V (G^{\ast} ,D^{\ast} ) \leq{m}$ i.e. $V (G^{\ast} ,D^{\ast} ) = m$. Using equation 10, we see that can only happen if $\int_{x}{One_{p_{data}}(x)}<p_{G}(x)dx = 0$, which is true if and only if $p_{G} = p_{data}$ almost everywhere (this is because $p_{data}$ and $p_{G}$ are probabilities densities, see lemma 2 in the appendix A for details).
>> 따라서 $m ≤ V (G^{\ast} ,D^{\ast} ) \leq{m}$. 즉, $V (G^{\ast} ,D^{\ast} ) = m$. 방정식 10을 사용하여, 우리는 $\int_{x}{One_{p_{data}}(x)}<p_{G}(x)dx = 0$가 진실인 경우에만 발생할 수 있음을 알 수 있다. 이는 $p_{G} = p_{data}$가 거의 모든 곳에서(이는 $p_{data}$와 $p_{G}$가 확률 밀도이기 때문이다. 자세한 내용은 부록 A의 렘마 2 참조).

> **Theorem 2** . A Nash equilibrium of this system exists and is characterized by (a) $p_{G}^{\ast } = p_{data}$ (almost everywhere) and (b) there exists a constant $γ ∈ [0, m]$ such that $D^{\ast} (x) = γ$ (almost everywhere).
>> > **정론 2**. 이 시스템의 내시 평형이 존재하며  (a) $p_{G}^{\ast } = p_{data}$ (거의 모든 곳에)와 (b) $D^{\ast} (x) = γ$(거의 모든 곳에)가 되도록 일정한 $γ [ [0, m]$이 존재한다.

> Proof. See appendix A.

#### $\mathbf{2.3\;USING\;AUTO-ENCODERS}$

> In our experiments, the discriminator $D$ is structured as an auto-encoder:
>> 우리의 실험에서 판별기 $D$는 자동 인코더로 구성된다.

$$D(x)=\parallel{Dec(Enc(x))-x}\parallel.$$

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-1.JPG)

> Figure 1: EBGAN architecture with an auto-encoder discriminator.
>> 그림 1: 자동 인코더 판별기가 있는 EBGAN 아키텍처.

> The diagram of the EBGAN model with an auto-encoder discriminator is depicted in figure 1. The choice of the auto-encoders for $D$ may seem arbitrary at the first glance, yet we postulate that it is conceptually more attractive than a binary logistic network:
>> 자동 인코더 판별기가 있는 EBGAN 모델의 다이어그램은 그림 1에 설명되어 있습니다. $D$에 대한 자동 인코더의 선택은 언뜻 보기에 임의적으로 보일 수 있지만, 우리는 그것이 이진 로지스틱 네트워크보다 개념적으로 더 매력적이라고 가정한다.

* > Rather than using a single bit of target information to train the model, the reconstruction-based output offers a diverse targets for the discriminator. With the binary logistic loss, only two targets are possible, so within a minibatch, the gradients corresponding to different samples are most likely far from orthogonal. This leads to inefficient training, and reducing the minibatch sizes is often not an option on current hardware. On the other hand, the reconstruction loss will likely produce very different gradient directions within the minibatch, allowing for larger minibatch size without loss of efficiency.
    >> 모델을 훈련시키기 위해 단일 비트의 대상 정보를 사용하는 대신, 재구성 기반 출력은 판별자에게 다양한 목표를 제공한다. 이항 로지스틱 손실의 경우 두 개의 목표값만 가능하므로 미니 배치 내에서 서로 다른 표본에 해당하는 그레이디언트는 직교에서 멀어질 가능성이 높습니다. 이는 비효율적인 교육으로 이어지며, 현재 하드웨어에서는 미니 배치 크기를 줄이는 것이 선택 사항이 아닌 경우가 많다. 반면, 재구성 손실은 미니 배치 내에서 매우 다른 기울기 방향을 생성하여 효율 손실 없이 미니 배치 크기를 더 크게 할 수 있다.

* > Auto-encoders have traditionally been used to represent energy-based model and arise naturally. When trained with some regularization terms (see section 2.3.1), auto-encoders have the ability to learn an energy manifold without supervision or negative examples. This means that even when an EBGAN auto-encoding model is trained to reconstruct a real sample, the discriminator contributes to discovering the data manifold by itself. To the contrary, without the presence of negative examples from the generator, a discriminator trained with binary logistic loss becomes pointless.
    >> 자동 인코더는 전통적으로 에너지 기반 모델을 나타내는 데 사용되었으며 자연스럽게 발생한다. 일부 정규화 용어로 훈련할 때(섹션 2.3.1 참조), 자동 인코더는 감독이나 부정적인 예 없이 에너지 매니폴드를 학습할 수 있다. 이는 EBGAN 자동 인코딩 모델이 실제 샘플을 재구성하도록 훈련될 때에도 판별기가 데이터 매니폴드를 스스로 발견하는 데 기여한다는 것을 의미한다. 반대로, 생성기의 부정적인 예가 없으면 이진 로지스틱 손실로 훈련된 판별기는 무의미해진다.

##### $\mathbf{2.3.1\;CONNECTION\;TO\;THE\;REGULARIZED\;AUTO-ENCODERS}$

> One common issue in training auto-encoders is that the model may learn little more than an identity function, meaning that it attributes zero energy to the whole space. In order to avoid this problem, the model must be pushed to give higher energy to points outside the data manifold. Theoretical and experimental results have addressed this issue by regularizing the latent representations (Vincent et al., 2010; Rifai et al., 2011; MarcAurelio Ranzato & Chopra, 2007; Kavukcuoglu et al., 2010). Such regularizers aim at restricting the reconstructing power of the auto-encoder so that it can only attribute low energy to a smaller portion of the input points.
>> > 자동 인코더 훈련에서 공통적인 문제 중 하나는 모델이 전체 공간에 0의 에너지를 부여한다는 의미인 식별 함수 이상을 학습할 수 있다는 것이다. 이 문제를 피하려면 모델을 눌러 데이터 매니폴드 외부의 점에 더 높은 에너지를 공급해야 합니다. 이론적 및 실험 결과는 잠재 표현을 정규화함으로써 이 문제를 해결했다(Vincent et al., 2010; Rifai et al., 2011; Marc Aurelio Ranzato & Chopra, 2007; Kavukcuoglu et al., 2010). 이러한 정규화기는 입력 포인트의 더 작은 부분에만 낮은 에너지를 돌릴 수 있도록 자동 인코더의 재구성 전력을 제한하는 것을 목표로 한다.

> We argue that the energy function (the discriminator) in the EBGAN framework is also seen as being regularized by having a generator producing the contrastive samples, to which the discriminator ought to give high reconstruction energies. We further argue that the EBGAN framework allows more flexibility from this perspective, because: (i)-the regularizer (generator) is fully trainable instead of being handcrafted; (ii)-the adversarial training paradigm enables a direct interaction between the duality of producing contrastive sample and learning the energy function.
>> > 우리는 EBGAN 프레임워크의 에너지 함수(판별기)가 또한 판별기가 높은 재구성 에너지를 제공해야 하는 대조 샘플을 생성하는 발전기를 가지고 있어 정규화된 것으로 보인다고 주장한다. 우리는 또한 EBGAN 프레임워크가 이러한 관점에서 더 많은 유연성을 허용한다고 주장한다. 왜냐하면 (i) 정규화기(발전기)는 수작업 대신 완전히 훈련할 수 있고, (ii) 적대적 훈련 패러다임은 대조적인 샘플 생산과 에너지 기능 학습의 이중성 사이에 직접적인 상호 작용을 가능하게 하기 때문이다.

#### $\mathbf{2.4\;REPELLING\;REGULARIZER}$

> We propose a “repelling regularizer” which fits well into the EBGAN auto-encoder model, purposely keeping the model from producing samples that are clustered in one or only few modes of $p_{data}$. Another technique “minibatch discrimination” was developed by Salimans et al. (2016) from the same philosophy.
>> 우리는 EBGAN 자동 인코더 모델에 잘 맞는 "반발 정규화기"를 제안하여 모델이 $p_{data}$의 하나 또는 몇 가지 모드로만 클러스터된 샘플을 생성하는 것을 의도적으로 방지한다. Salimans 외 연구진(2016)이 동일한 철학에서 개발한 또 다른 기술 "미니배치 차별"이 있다.

> Implementing the repelling regularizer involves a Pulling-away Term (PT) that runs at a representation level. Formally, let $S ∈ R^{s\times{N}}$ denotes a batch of sample representations taken from the encoder output layer. Let us define PT as:
>> 반발 정규화 구현에는 표현 수준에서 실행되는 PT(Pulling-away Term)가 포함됩니다. 형식적으로, $S ∈ R^{s\times{N}}$는 인코더 출력 계층에서 가져온 샘플 표현의 배치를 나타낸다. PT를 다음과 같이 정의하자.

$$f_{PT}(S)=\frac{1}{N(N-1)}\sum_{i}\sum_{j\neq{i}}(\frac{S_{i}^{T}S_{j}}{\parallel{S_{i}}\parallel\parallel{S_{j}}\parallel})^{2}.$$

> PT operates on a mini-batch and attempts to orthogonalize the pairwise sample representation. It is inspired by the prior work showing the representational power of the encoder in the auto-encoder alike model such as Rasmus et al. (2015) and Zhao et al. (2015). The rationale for choosing the cosine similarity instead of Euclidean distance is to make the term bounded below and invariant to scale. We use the notation “EBGAN-PT” to refer to the EBGAN auto-encoder model trained with this term. Note the PT is used in the generator loss but not in the discriminator loss
>> PT는 미니 배치에서 작동하며 쌍별 샘플 표현을 직교하려고 시도합니다. Rasmus 등(2015)과 Zhao 등(2015)과 같은 자동 인코더 유사 모델에서 인코더의 표현력을 보여주는 이전 연구에서 영감을 얻었다. 유클리드 거리 대신 코사인 유사성을 선택하는 근거는 항이 아래에 한정되고 크기에 따라 불변하도록 만드는 것이다. 우리는 "EBGAN-PT"라는 표기법을 사용하여 이 용어로 훈련된 EBGAN 자동 인코더 모델을 참조한다. PT는 제너레이터 손실에 사용되지만 판별기 손실에 사용되지는 않습니다.

### $\mathbf{3\;RELATED\;WORK}$

> Our work primarily casts GANs into an energy-based model scope. On this direction, the approaches studying contrastive samples are relevant to EBGAN, such as the use of noisy samples (Vincent et al., 2010) and noisy gradient descent methods like contrastive divergence (Carreira-Perpinan & Hinton, 2005). From the perspective of GANs, several papers were presented to improve the stability of GAN training, (Salimans et al., 2016; Denton et al., 2015; Radford et al., 2015; Im et al., 2016; Mathieu et al., 2015).
>> 우리의 연구는 주로 GAN을 에너지 기반 모델 범위에 캐스팅한다. 이 방향에서, 대조 샘플을 연구하는 접근법은 잡음이 많은 샘플의 사용(Vincent et al., 2010)과 대조적인 발산(Carreira-Perpinan & Hinton, 2005)과 같은 잡음이 많은 그레이디언트 강하 방법과 같은 EBGAN과 관련이 있다. GANs의 관점에서, GAN 훈련의 안정성을 개선하기 위해 여러 논문이 제시되었다(Salimans et al., 2016; Denton et al., 2015; Radford et al., 2015; Im et al., 2016; Mathieu et al., 2015).

> Kim & Bengio (2016) propose a probabilistic GAN and cast it into an energy-based density estimator by using the Gibbs distribution. Quite unlike EBGAN, this proposed framework doesn’t get rid of the computational challenging partition function, so the choice of the energy function is required to be integratable.
>> Kim & Bengio(2016)는 확률적 GAN을 제안하고 깁스 분포를 사용하여 에너지 기반 밀도 추정기에 캐스팅하였다. EBGAN과 달리, 이 제안된 프레임워크는 계산 도전 파티션 함수를 제거하지 않기 때문에 에너지 기능의 선택이 통합 가능해야 한다.

### $\mathbf{4\;Experiments}$

#### $\mathbf{4.1\;EXHAUSTIVE\;GRID\;SEARCH\;ON\;MNIST}$

> In this section, we study the training stability of EBGANs over GANs on a simple task of MNIST digit generation with fully-connected networks. We run an exhaustive grid search over a set of architectural choices and hyper-parameters for both frameworks.
>> 이 섹션에서는 완전히 연결된 네트워크를 통한 MNIST 숫자 생성의 간단한 작업에 대해 GAN을 통한 EBGAN의 훈련 안정성을 연구한다. 우리는 두 프레임워크에 대한 일련의 아키텍처 선택과 하이퍼 파라미터에 대해 철저한 그리드 검색을 실행한다.

> Formally, we specify the search grid in table 1. We impose the following restrictions on EBGAN models: (i)-using learning rate 0.001 and Adam (Kingma & Ba, 2014) for both $G$ and D; (ii)- nLayerD represents the total number of layers combining Enc and Dec. For simplicity, we fix Dec to be one layer and only tune the Enc #layers; (iii)-the margin is set to 10 and not being tuned. To analyze the results, we use the inception score (Salimans et al., 2016) as a numerical means reflecting the generation quality. Some slight modification of the formulation were made to make figure 2 visually more approachable while maintaining the score’s original meaning, $I' = E_{x}KL(p(y)\parallel{p(y\mid{x})})^{2}$ (more details in appendix C). Briefly, higher $I'$ score implies better generation quality.
>> 공식적으로, 우리는 표 1에 검색 그리드를 지정한다. 우리는 EBGAN 모델에 다음과 같은 제한을 가한다. (i) $G$와 D 모두에 대한 학습률 0.001과 Adam(Kingma & Ba, 2014); (ii)-nLayerD는 Enc와 Dec을 결합한 총 계층 수를 나타낸다. 단순성을 위해 Dec을 한 레이어로 고정하고 Enc #레이어만 튜닝한다. (iii) 여백은 10으로 설정되고 튜닝되지 않는다. 결과를 분석하기 위해, 우리는 세대 품질을 반영하는 수치 수단으로 초기 점수(Salimans et al., 2016)를 사용한다. 점수의 원래 의미인 $I' = E_{x}KL(p(y)\parallel{p(y\mid{x})})^{2}$(부록 C의 더 자세한 정보)를 유지하면서 그림 2를 시각적으로 더 쉽게 만들 수 있도록 공식을 약간 수정했다. 간단히 말해서, $I'$ 점수가 높으면 더 나은 생성 품질을 의미한다.

> **Histograms**  We plot the histogram of $I'$ scores in figure 2. We further separated out the optimization related setting from GAN’s grid (optimD, optimG and lr) and plot the histogram of each subgrid individually, together with the EBGAN I 0 scores as a reference, in figure 3. The number of experiments for GANs and EBGANs are both 512 in every subplot. The histograms evidently show that EBGANs are more reliably trained.
>> **Histograms** 우리는 그림 2에 $I'$ 점수의 히스토그램을 표시한다. 우리는 최적화 관련 설정을 GAN의 그리드(optim D, optim G 및 lr)에서 추가로 분리하고 각 하위 그리드의 히스토그램을 참조로 EBGAN I 0 점수와 함께 그림 3에 표시했다. GAN과 EBGAN에 대한 실험 횟수는 모든 하위 그림에서 모두 512회입니다. 히스토그램은 EBGAN이 더 안정적으로 훈련되었음을 분명히 보여준다.

![Table 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Table-1.JPG)

> Digits generated from the configurations presenting the best inception score are shown in figure 4.
>> 최상의 초기 점수를 나타내는 구성에서 생성된 숫자는 그림 4와 같다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-2.JPG)

> Figure 2: (Zooming in on pdf file is recommended.) Histogram of the inception scores from the grid search. The x-axis carries the inception score I and y-axis informs the portion of the models (in percentage) falling into certain bins. Left (a): general comparison of EBGANs against GANs; Middle (b): EBGANs and GANs both constrained by nLayer[GD]<=4; Right (c): EBGANs and GANs both constrained by nLayer[GD]<=3.
>> 그림 2: (pdf 파일을 확대/축소하는 것이 좋습니다.) 그리드 검색의 시작 점수에 대한 히스토그램입니다. X 축에는 시작 점수 I이 표시되며 Y 축에는 특정 빈에 떨어지는 모형의 부분(백분율)이 표시됩니다. 왼쪽 (a) : GANs에 대한 EBGANs의 일반적인 비교; 중간 (b) : EBGANs와 GANs는 모두 nLayer에 의해 제한된다.GD]<=4; 오른쪽 (c): EBGAN과 GAN은 모두 nLayer에 의해 제약된다.GD]<=3.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-3.JPG)

> Figure 3: (Zooming in on pdf file is recommended.) Histogram of the inception scores grouped by different optimization combinations, drawn from optimD, optimG and lr (See text).
>> 그림 3: (pdf 파일을 확대/축소하는 것이 좋습니다.) optim D, optim G 및 lr에서 추출한 서로 다른 최적화 조합으로 그룹화된 시작 점수의 히스토그램입니다(텍스트 참조).

#### $\mathbf{4.2\;SEMI-SUPERVISED\;LEARNING\;ON\;MNIST}$

> We explore the potential of using the EBGAN framework for semi-supervised learning on permutation-invariant MNIST, collectively on using 100, 200 and 1000 labels. We utilized a bottom layer-cost Ladder Network (LN) (Rasmus et al., 2015) with the EGBAN framework (EBGAN-LN). Ladder Network can be categorized as an energy-based model that is built with both feedforward and feedback hierarchies powered by stage-wise lateral connections coupling two pathways.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-4.JPG)

> Figure 4: Generation from the grid search on MNIST. Left(a): Best GAN model; Middle(b): Best EBGAN model. Right(c): Best EBGAN-PT model.
>> 그림 4: MNIST의 그리드 검색에서 생성 왼쪽(a): 최상의 GAN 모델; 중간(b): 최고의 EBGAN 모델. 오른쪽(c): 최상의 EBGAN-PT 모델.

> One technique we found crucial in enabling EBGAN framework for semi-supervised learning is to gradually decay the margin value m of the equation 1. The rationale behind is to let discriminator punish generator less when $p_{G}$ gets closer to the data manifold. One can think of the extreme case where the contrastive samples are exactly pinned on the data manifold, such that they are “not contrastive anymore”. This ultimate status happens when $m = 0$ and the EBGAN-LN model falls back to a normal Ladder Network. The undesirability of a non-decay dynamics for using the discriminator in the GAN or EBGAN framework is also indicated by Theorem 2: on convergence, the discriminator reflects a flat energy surface. However, we posit that the trajectory of learning a EBGAN-LN model does provide the LN (discriminator) more information by letting it see contrastive samples. Yet the optimal way to avoid the mentioned undesirability is to make sure m has been decayed to 0 when the Nash Equilibrium is reached. The margin decaying schedule is found by hyper-parameter search in our experiments (technical details in appendix D).
>> 준지도 학습을 위한 EBGAN 프레임워크를 활성화하는 데 중요한 한 가지 기술은 방정식 1의 여유 값 m을 점진적으로 감소시키는 것이다. 뒤의 근거는 $p_{G}$가 데이터 매니폴드에 가까워질 때 판별자가 생성자를 덜 처벌하도록 하는 것이다. 대조 표본이 데이터 매니폴드에 정확히 고정되어 "더 이상 대조적이지 않다"는 극단적인 경우를 생각할 수 있다. 이 궁극적인 상태는 $m = 0$이고 EBGAN-LN 모델이 정상적인 래더 네트워크로 다시 떨어질 때 발생한다. GAN 또는 EBGAN 프레임워크에서 판별기를 사용하기 위한 붕괴되지 않는 역학의 바람직하지 않은 점은 정리 2에 나타나 있다. 수렴 시 판별기는 평평한 에너지 표면을 반영한다. 그러나, 우리는 EBGAN-LN 모델을 학습하는 궤적은 LN(판별기)이 대조적인 샘플을 보게 함으로써 더 많은 정보를 제공한다고 가정한다. 그러나 언급된 바람직하지 않은 것을 피하는 가장 좋은 방법은 내쉬 균형에 도달할 때 m이 0으로 붕괴되었는지 확인하는 것이다. 여유 감소 일정은 실험의 하이퍼 매개 변수 검색에 의해 발견된다(부록 D의 기술 세부 사항).

> From table 2, it shows that positioning a bottom-layer-cost LN into an EBGAN framework profitably improves the performance of the LN itself. We postulate that within the scope of the EBGAN framework, iteratively feeding the adversarial contrastive samples produced by the generator to the energy function acts as an effective regularizer; the contrastive samples can be thought as an extension to the dataset that provides more information to the classifier. We notice there was a discrepancy between the reported results between Rasmus et al. (2015) and Pezeshki et al. (2015), so we report both results along with our own implementation of the Ladder Network running the same setting. The specific experimental setting and analysis are available in appendix D.
>> 표 2에서, 그것은 하위 계층 비용 LN을 EBGAN 프레임워크에 위치시키면 LN 자체의 성능이 수익성 있게 향상된다는 것을 보여준다. 우리는 EBGAN 프레임워크의 범위 내에서 발전기에 의해 생성된 적대적 대조 샘플을 에너지 함수에 반복적으로 공급하는 것이 효과적인 정규화기로 작용한다고 가정한다. 대조 샘플은 분류기에 더 많은 정보를 제공하는 데이터 세트의 확장으로 생각할 수 있다. 우리는 라스무스 외 연구진(2015)과 페제스키 외 연구진(2015) 사이에 보고된 결과 사이에 불일치가 있다는 것을 알아차렸다. 따라서 우리는 동일한 설정을 실행하는 래더 네트워크의 자체 구현과 함께 두 결과를 보고한다. 구체적인 실험 설정과 분석은 부록 D에서 확인할 수 있다.

> Table rate2: The comparison of LN bottom-layer-cost model and its EBGAN extension on PI-MNIST semi-supervised task. Note the results are error  (in %) and averaged over 15 different random seeds.
>> 표 2: PI-MNIST 준지도 작업에 대한 LN 하위 계층 비용 모델과 EBGAN 확장 비교. 결과는 오류율(%)이며 15개의 서로 다른 랜덤 시드에 대한 평균값입니다.

![Table 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Table-2.JPG)

#### $\mathbf{4.3\;LSUN\;\;CELEBA}$

> We apply the EBGAN framework with deep convolutional architecture to generate 64 × 64 RGB images, a more realistic task, using the LSUN bedroom dataset (Yu et al., 2015) and the large-scale face dataset CelebA under alignment (Liu et al., 2015). To compare EBGANs with DCGANs (Radford et al., 2015), we train a DCGAN model under the same configuration and show its generation side-by-side with the EBGAN model, in figures 5 and 6. The specific settings are listed in appendix C.
>> LSUN 침실 데이터 세트(Yu et al., 2015)와 정렬 중인 대규모 얼굴 데이터 세트 CelebA(Liu et al., 2015)를 사용하여 보다 현실적인 작업인 64 × 64 RGB 이미지를 생성하기 위해 심층 컨볼루션 아키텍처를 적용한 EBGAN 프레임워크를 적용한다. EBGAN을 DCGAN과 비교하기 위해(Radford et al., 2015), 우리는 동일한 구성에서 DCGAN 모델을 훈련시키고 그림 5와 6에서 EBGAN 모델과 그 생성을 나란히 보여준다. 특정 설정은 부록 C에 나와 있습니다.

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-5.JPG)

> Figure 5: Generation from the LSUN bedroom dataset. Left(a): DCGAN generation. Right(b): EBGAN-PT generation
>> 그림 5: LSUN 침실 데이터 세트에서 생성 왼쪽(a): DCGAN 생성. 오른쪽(b): EBGAN-PT 생성

![Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-6.JPG)

> Figure 6: Generation from the CelebA dataset. Left(a): DCGAN generation. Right(b): EBGAN-PT generation.
>> 그림 6: CelebA 데이터 세트에서 생성 왼쪽(a): DCGAN 생성. 오른쪽(b): EBGAN-PT 생성.

#### $\mathbf{4.4\;IMAGENET}$

> Finally, we trained EBGANs to generate high-resolution images on ImageNet (Russakovsky et al., 2015). Compared with the datasets we have experimented so far, ImageNet presents an extensively larger and wilder space, so modeling the data distribution by a generative model becomes very challenging. We devised an experiment to generate 128 × 128 images, trained on the full ImageNet-1k dataset, which contains roughly 1.3 million images from 1000 different categories. We also trained a network to generate images of size 256 × 256, on a dog-breed subset of ImageNet, using the wordNet IDs provided by Vinyals et al. (2016). The results are shown in figures 7 and 8. Despite the difficulty of generating images on a high-resolution level, we observe that EBGANs are able to learn about the fact that objects appear in the foreground, together with various background components resembling grass texture, sea under the horizon, mirrored mountain in the water, buildings, etc. In addition, our 256 × 256 dog-breed generations, although far from realistic, do reflect some knowledge about the appearances of dogs such as their body, furs and eye.
>> 마지막으로, ImageNet에서 고해상도 이미지를 생성하도록 EBGAN을 훈련시켰다(Russakovsky et al., 2015). 지금까지 실험한 데이터 세트와 비교하여 ImageNet은 광범위하게 더 크고 광활한 공간을 제공하므로 생성 모델에 의한 데이터 분포를 모델링하는 것이 매우 어려워진다. 우리는 1000개의 다른 범주에서 약 130만 개의 이미지를 포함하는 전체 ImageNet-1k 데이터 세트에 대해 훈련된 128 × 128개의 이미지를 생성하는 실험을 고안했다. 또한 Vinyals 등(2016)이 제공하는 Net ID라는 단어를 사용하여 ImageNet의 개 품종 하위 세트에서 256 × 256 크기의 이미지를 생성하도록 네트워크를 훈련시켰다. 결과는 그림 7과 8에 나와 있습니다. 고해상도 수준에서 이미지를 생성하는 것은 어렵지만, 우리는 EBGAN이 풀 텍스처, 수평선 아래의 바다, 물 속의 거울 산, 건물 등과 유사한 다양한 배경 구성 요소와 함께 물체가 전경에 나타난다는 사실을 배울 수 있다는 것을 관찰한다. 게다가, 우리의 256×256 견종 세대는 현실과는 거리가 멀지만, 그들의 몸, 털, 그리고 눈과 같은 개들의 외모에 대한 지식을 반영한다.

![Figure 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-7.JPG)

> Figure 7: ImageNet 128 × 128 generations using an EBGAN-PT.
>> 그림 7: EBGAN-PT를 사용하는 ImageNet 128 × 128 세대

![Figure 8](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-8.JPG)

> Figure 8: ImageNet 256 × 256 generations using an EBGAN-PT.
>> 그림 8: EBGAN-PT를 사용하는 ImageNet 256 × 256 세대

### $\mathbf{5\;OUTLOOK}$

> We bridge two classes of unsupervised learning methods – GANs and auto-encoders – and revisit the GAN framework from an alternative energy-based perspective. EBGANs show better convergence pattern and scalability to generate high-resolution images. A family of energy-based loss functionals presented in LeCun et al. (2006) can easily be incorporated into the EBGAN framework. For the future work, the conditional setting (Denton et al., 2015; Mathieu et al., 2015) is a promising setup to explore. We hope the future research will raise more attention on a broader view of GANs from the energy-based perspective.
>> 우리는 두 가지 등급의 비지도 학습 방법인 GAN과 자동 인코더를 연결하고 대체 에너지 기반 관점에서 GAN 프레임워크를 다시 살펴본다. EBGAN은 고해상도 이미지를 생성하기 위해 더 나은 수렴 패턴과 확장성을 보여준다. LeCun 등에 제시된 에너지 기반 손실 함수 제품군. (2006)은 EBGAN 프레임워크에 쉽게 통합될 수 있다. 향후 작업의 경우 조건부 설정(Denton et al., 2015; Mathieu et al., 2015)은 탐색할 수 있는 유망한 설정이다. 우리는 향후 연구가 에너지 기반 관점에서 GAN에 대한 더 넓은 시각에 대해 더 많은 관심을 불러일으킬 것으로 기대한다.

#### $\mathbf{Acknowledgments}$

> We thank Emily Denton, Soumith Chitala, Arthur Szlam, Marc’Aurelio Ranzato, Pablo Sprechmann, Ross Goroshin and Ruoyu Sun for fruitful discussions. We also thank Emily Denton and Tian Jiang for their help with the manuscript.
>> 에밀리 덴튼, 수미스 치탈라, 아서 슬람, 마르크 아우렐리오 란자토, 파블로 스프레히만, 로스 고로신, 뤼유 선에게 유익한 토론을 해주셔서 감사합니다. 우리는 또한 에밀리 덴튼과 티안 지앙에게 원고 작성에 도움을 준 것에 대해 감사를 표합니다.

### $\mathbf{F\;APPENDIX:\;MORE\;GENERATION}$

#### $\mathbf{LSUN\;AUGMENTED\;VERSION\;TRAINING}$

For LSUN bedroom dataset, aside from the experiment on the whole images, we also train an EBGAN auto-encoder model based on dataset augmentation by cropping patches. All the patches are of size 64×64 and cropped from 96×96 original images. The generation is shown in figure 11.
>> LSUN 침실 데이터 세트의 경우 전체 이미지에 대한 실험과는 별도로 패치를 자르는 방식으로 데이터 세트 증강을 기반으로 EBGAN 자동 인코더 모델을 교육한다. 모든 패치는 크기가 64×64이며 96×96 원본 이미지에서 잘라낸 것입니다. 세대는 그림 11에 나와 있습니다.
#### $\mathbf{COMPARISON\;OF\;EBGANS\;AND\;EBGAN-PTS}$

To further demonstrate how the pull-away term (PT) may influence EBGAN auto-encoder model training, we chose both the whole-image and augmented-patch version of the LSUN bedroom dataset, together with the CelebA dataset to make some further experimentation. The comparison of EBGAN and EBGAN-PT generation are showed in figure 12, figure 13 and figure 14. Note that all comparison pairs adopt identical architectural and hyper-parameter setting as in section 4.3. The cost weight on the PT is set to 0.1.
>> 풀 어웨이 용어(PT)가 EBGAN 자동 인코더 모델 훈련에 어떻게 영향을 미칠 수 있는지 추가로 입증하기 위해, 우리는 추가 실험을 하기 위해 LSUN 침실 데이터 세트의 전체 이미지 버전과 증강 패치 버전을 모두 선택했다. EBGAN과 EBGAN-PT 생성의 비교는 그림 12, 그림 13 및 그림 14에 나와 있습니다. 모든 비교 쌍은 섹션 4.3과 동일한 아키텍처 및 하이퍼 파라미터 설정을 채택한다. PT의 비용 가중치는 0.1로 설정됩니다.

![Figure 9](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-9.JPG)

> Figure 9: Generation from the EBGAN auto-encoder model trained with different m settings. From top to bottom, m is set to 1, 2, 4, 6, 8, 12, 16, 32 respectively. The rest setting is nLayerG=5, nLayerD=2, sizeG=1600, sizeD=1024, dropoutD=0, optimD=ADAM, optimG=ADAM, lr=0.001.
>> 그림 9: 다양한 m 설정으로 훈련된 EBGAN 자동 인코더 모델로부터의 생성. 위에서 아래로 m은 각각 1, 2, 4, 6, 8, 12, 16, 32로 설정된다. 나머지 설정은 nLayerG=5, nLayerD=2, sizeG=sumber, sizeD=sumber, dropoutD=0, optimD=입니다.ADAM, optimG=ADAM, lr=0.001.

![Figure 10](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-10.JPG)

> Figure 10: Generation from the EBGAN-LN model. The displayed generations are obtained by an identical experimental setting described in appendix D, with different random seeds. As we mentioned before, we used the unpadded version of the MNIST dataset (size 28×28) in the EBGANLN experiments.
>> 그림 10: EBGAN-LN 모델로부터의 생성 표시된 세대는 부록 D에 설명된 것과 동일한 실험 설정으로 다른 무작위 시드를 사용하여 얻는다. 앞서 언급했듯이, 우리는 EBGANLN 실험에서 추가되지 않은 버전의 MNIST 데이터 세트(크기 28×28)를 사용했다.

![Figure 11](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-11.JPG)

> Figure 11: Generation from augmented-patch version of the LSUN bedroom dataset. Left(a): DCGAN generation. Right(b): EBGAN-PT generation.
>> 그림 11: LSUN 침실 데이터 세트의 증강 패치 버전에서 생성. 왼쪽(a): DCGAN 생성. 오른쪽(b): EBGAN-PT 생성.

![Figure 12](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-12.JPG)

> Figure 12: Generation from whole-image version of the LSUN bedroom dataset. Left(a): EBGAN. Right(b): EBGAN-PT.
>> 그림 12: LSUN 침실 데이터 세트의 전체 이미지 버전에서 생성. 왼쪽(a): EBGAN. 오른쪽(b): EBGAN-PT.

![Figure 13](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-13.JPG)

> Figure 13: Generation from augmented-patch version of the LSUN bedroom dataset. Left(a): EBGAN. Right(b): EBGAN-PT.
>> 그림 13: LSUN 침실 데이터 세트의 증강 패치 버전에서 생성. 왼쪽(a): EBGAN. 오른쪽(b): EBGAN-PT.

![Figure 14](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-08-(GAN)ENERGY-BASED-GAN/Figure-14.JPG)

> Figure 14: Generation from the CelebA dataset. Left(a): EBGAN. Right(b): EBGAN-PT.
>> 그림 14: CelebA 데이터 세트에서 생성 왼쪽(a): EBGAN. 오른쪽(b): EBGAN-PT.