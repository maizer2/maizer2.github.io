---
layout: post 
title: "(GAN)A Style-Based Generator Architecture for Generative Adversarial Networks Translation"
categories: [1. Computer Engineering]
tags: [1.7. Paper Review, 1.2.2.5. GAN]
---

### [GAN Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/paper-of-GAN.html)

### [$$\mathbf{A\;Style-Based\;Generator\;Architecture\;for\;Generative\;Adversarial\;Networks}$$](https://arxiv.org/pdf/1812.04948.pdf)

### $\mathbf{Abstract}$

> We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. 
>> 우리는 스타일 전송(style transfer) 문헌에서 차용한 생성적 적대 네트워크를 위한 대체 생성기 아키텍처(alternative generator architecture)를 제안한다.

> The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. 
>> 새로운 아키텍처는 높은 수준의 속성(예: 사람 얼굴에서 훈련될 때 자세와 정체성)과 생성된 이미지(예: 주근깨, 머리카락)의 확률적 변동에 대해 자동으로 학습되고 감독되지 않은 분리로 이어지며, 이는 합성에 대한 직관적이고 규모별 제어를 가능하게 한다.

> The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. 
>> 새로운 발전기는 기존의 분포 품질 지표 측면에서 최첨단 기술을 개선하고, 보다 나은 보간 특성(interpolation properties)을 보여주며, 또한 변동의 잠재 요인을 더 잘 분리한다.

> To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. 
>> 보간 품질과 분리를 정량화하기 위해(To quantify), 우리는 모든 발전기 아키텍처에 적용할 수 있는 두 가지 새로운 자동화된 방법을 제안한다.

> Finally, we introduce a new, highly varied and high-quality dataset of human faces.
>> 마지막으로, 우리는 새롭고 매우 다양하고 고품질의 인간 얼굴 데이터 세트를 소개한다.

### $\mathbf{1.\;Introduction}$

> The resolution and quality of images produced by generative methods—especially generative adversarial networks (GAN)[<a href="#footnote_21_1" name="footnote_21_2">21</a>]— have seen rapid improvement recently[<a href="#footnote_28_1" name="footnote_28_2">28</a>, <a href="#footnote_41_1" name="footnote_41_2">41</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>]. 
>> 생성 방법, 특히 생성적 적대 네트워크(GAN)[<a href="#footnote_21_1" name="footnote_21_2">21</a>]에 의해 생성된 이미지의 해상도와 품질은 최근 급속하게 개선되었다[<a href="#footnote_28_1" name="footnote_28_2">28</a>, <a href="#footnote_41_1" name="footnote_41_2">41</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>].

> Yet the generators continue to operate as black boxes, and despite recent efforts [<a href="#footnote_2_1" name="footnote_2_2">2</a>], the understanding of various aspects of the image synthesis process, e.g., the origin of stochastic features, is still lacking. 
>> 그러나 발전기는 블랙박스로 계속 작동하며, 최근의 노력에도 불구하고[<a href="#footnote_2_1" name="footnote_2_2">2</a>], 확률적 특징의 기원 등 이미지 합성 과정의 다양한 측면에 대한 이해가 여전히 부족하다.

> The properties of the latent space are also poorly understood, and the commonly demonstrated latent space interpolations [<a href="#footnote_12_1" name="footnote_12_2">12</a>, <a href="#footnote_48_1" name="footnote_48_2">48</a>, <a href="#footnote_34_1" name="footnote_34_2">34</a>] provide no quantitative way to compare different generators against each other.
>> 잠재 공간의 특성도 잘 이해되지 않으며, 일반적으로 입증되는 잠재 공간 보간[<a href="#footnote_12_1" name="footnote_12_2">12</a>, <a href="#footnote_48_1" name="footnote_48_2">48</a>, <a href="#footnote_34_1" name="footnote_34_2">34</a>]은 서로 다른 생성자를 비교할 수 있는 정량적인 방법을 제공하지 않는다.

> Motivated by style transfer literature[<a href="#footnote_26_1" name="footnote_26_2">26</a>], we re-design the generator architecture in a way that exposes novel ways to control the image synthesis process. 
>> 스타일 전송 문헌[<a href="#footnote_26_1" name="footnote_26_2">26</a>]에 의해 동기 부여되어, 우리는 이미지 합성 프로세스를 제어하는 새로운 방법을 노출하는 방식으로 제너레이터 아키텍처를 다시 설계한다. 

> Our generator starts from a learned constant input and adjusts the “style” of the image at each convolution layer based on the latent code, therefore directly controlling the strength of image features at different scales. 
>> 우리의 생성기는 학습된 상수 입력에서 시작하여 잠재 코드를 기반으로 각 컨볼루션 레이어에서 이미지의 "스타일"을 조정하여 서로 다른 스케일에서 이미지 기능의 강도를 직접 제어한다.

> Combined with noise injected directly into the network, this architectural change leads to automatic, unsupervised separation of high-level attributes (e.g., pose, identity) from stochastic variation (e.g., freckles, hair) in the generated images, and enables intuitive scale-specific mixing and interpolation operations. 
>> 네트워크에 직접 주입된 노이즈와 결합된 이 아키텍처 변경은 생성된 이미지의 확률적 변화(예: 주근깨, 머리카락)로부터 높은 수준의 속성(예: 포즈, 정체성)을 자동적이고 감독되지 않은 분리로 이어지고 직관적인 스케일별 혼합 및 보간 작업을 가능하게 한다.

> We do not modify the discriminator or the loss function in any way, and our work is thus orthogonal to the ongoing discussion about GAN loss functions, regularization, and hyper-parameters[<a href="#footnote_23_1" name="footnote_23_2">23</a>, <a href="#footnote_41_1" name="footnote_41_2">41</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>, <a href="#footnote_37_1" name="footnote_37_2">37</a>, <a href="#footnote_40_1" name="footnote_40_2">4</a>, <a href="#footnote_33_1" name="footnote_33_2">33</a>].
>> 우리는 판별기 또는 손실 함수를 어떤 방식으로도 수정하지 않으며, 따라서 우리의 작업은 GAN 손실 함수, 정규화 및 초 매개 변수에 대한 진행 중인 논의와 직교한다[<a href="#footnote_23_1" name="footnote_23_2">23</a>, <a href="#footnote_41_1" name="footnote_41_2">41</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>, <a href="#footnote_37_1" name="footnote_37_2">37</a>, <a href="#footnote_40_1" name="footnote_40_2">4</a>, <a href="#footnote_33_1" name="footnote_33_2">33</a>].

> Our generator embeds the input latent code into an intermediate latent space, which has a profound effect on how the factors of variation are represented in the network. 
>> 우리의 생성기는 입력 잠재 코드를 중간 잠재 공간에 내장하는데, 이는 변동 요인이 네트워크에서 어떻게 표현되는지에 심오한 영향을 미친다.

> The input latent space must follow the probability density of the training data, and we argue that this leads to some degree of unavoidable entanglement. 
>> 입력 잠재 공간은 훈련 데이터의 확률 밀도(probability density)를 따라야 하며, 이로 인해 어느 정도 피할 수 없는 얽힘이 발생한다고 주장한다.

> Our intermediate latent space is free from that restriction and is therefore allowed to be disentangled. 
>> 우리의 중간 잠재 공간은 그러한 제약으로부터 자유로우므로 얽혀있는 것이 허용된다.

> As previous methods for estimating the degree of latent space disentanglement are not directly applicable in our case, we propose two new automated metrics— perceptual path length and linear separability— for quantifying these aspects of the generator. 
>> 잠재 공간 분리의 정도를 추정하는 이전의 방법은 우리의 경우에 직접 적용할 수 없으므로, 우리는 발전기의 이러한 측면을 정량화하기 위한 두 가지 새로운 자동화된 메트릭, 즉 지각 경로 길이와 선형 분리 가능성을 제안한다.

> Using these metrics, we show that compared to a traditional generator architecture, our generator admits a more linear, less entangled representation of different factors of variation.
>> 이러한 메트릭을 사용하여, 우리는 전통적인 발전기 아키텍처와 비교하여, 우리의 발전기가 다른 변동 요인의 더 선형적이고 덜 얽힌 표현을 허용한다는 것을 보여준다.

> Finally, we present a new dataset of human faces (FlickrFaces-HQ, FFHQ) that offers much higher quality and covers considerably wider variation than existing highresolution datasets (Appendix A). 
>> 마지막으로, 기존 고해상도 데이터 세트(부록 A)보다 훨씬 높은 품질을 제공하고 상당히 광범위한 변형을 다루는 새로운 인간 얼굴 데이터 세트(FlickrFaces-HQ, FFHQ)를 제시한다.

> We have made this dataset publicly available, along with our source code and pretrained networks. 
>> 우리는 이 데이터 세트를 소스 코드 및 사전 훈련된 네트워크와 함께 공개적으로 사용할 수 있도록 했다.

> The accompanying video can be found under the same link.
>> 동봉된 비디오는 같은 링크에서 찾을 수 있습니다.

### $\mathbf{2.\;Style-based\;generator}$

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Figure-1.JPG)

> Figure 1. While a traditional generator [<a href="#footnote_28_1" name="footnote_28_2">28</a>] feeds the latent code though the input layer only, we first map the input to an intermediate latent space $W$, which then controls the generator through adaptive instance normalization (AdaIN) at each convolution layer. Gaussian noise is added after each convolution, before evaluating the nonlinearity. Here “A” stands for a learned affine transform, and “B” applies learned per-channel scaling factors to the noise input. The mapping network f consists of 8 layers and the synthesis network g consists of 18 layers— two for each resolution $(4^{2}−1024^{2})$. The output of the last layer is converted to RGB using a separate 1 × 1 convolution, similar to Karras et al. [<a href="#footnote_28_1" name="footnote_28_2">28</a>]. Our generator has a total of 26.2M trainable parameters, compared to 23.1M in the traditional generator.
>> 그림 1 기존의 생성기 [<a href="#footnote_28_1" name="footnote_28_2">28</a>]는 입력 계층만을 통해 잠재 코드를 공급하지만, 우리는 먼저 입력을 중간 잠재 공간 $W$에 매핑한 다음 각 컨볼루션 계층에서 적응형 인스턴스 정규화(AdaIN)를 통해 생성기를 제어한다. 가우스 노이즈는 비선형성을 평가하기 전에 각 컨볼루션 후에 추가된다. 여기서 "A"는 학습된 아핀 변환을 의미하며, "B"는 학습된 채널당 스케일링 계수를 노이즈 입력에 적용합니다. 매핑 네트워크 f는 8개의 레이어로 구성되며 합성 네트워크 g는 각 해상도 $(4^{2}-1024^{2})$에 대해 2개씩 총 18개의 레이어로 구성된다. 마지막 레이어의 출력은 별도의 1×1 컨볼루션을 사용하여 RGB로 변환된다.[<a href="#footnote_28_1" name="footnote_28_2">28</a>]. 우리의 발전기는 기존 발전기의 23.1M와 비교하여 총 26.2M 훈련 가능한 매개 변수를 가지고 있다.

> Traditionally the latent code is provided to the generator through an input layer, i.e., the first layer of a feedforward network (Figure 1a). 
>> Traditional generator는 잠재 코드를 입력 계층에, 즉 피드포워드 네트워크의 첫 번째 계층을 통해 발전기에 제공된다(그림 1a).

> We depart from this design by omitting the input layer altogether and starting from a learned constant instead (Figure 1b, right). 
>> Style-based generator는 입력 레이어를 완전히 생략하고 학습된 상수(그림 1b, 오른쪽)에서 시작하여 이 설계에서 출발합니다.

> Given a latent code $z$ in the input latent space $Z$, a non-linear mapping network $f:Z\to{W}$ first produces $w\in{W}$(Figure 1b, left). 
>> 입력 잠재 공간 $Z$에 잠재 코드 $z$가 주어지면, 비선형 매핑 네트워크 $f:Z\to{W}$는 먼저 $w\in{W}$를 생성한다(그림 1b, 왼쪽).

> For simplicity, we set the dimensionality of both spaces to 512, and the mapping $f$ is implemented using an 8-layer MLP, a decision we will analyze in Section 4.1. 
>> 단순성을 위해 두 공간의 차원성을 512로 설정하고, 매핑 $f$는 섹션 4.1에서 분석할 결정인 8계층 MLP를 사용하여 구현된다. 

> Learned affine transformations then specialize $w$ to styles $y=(y_{s},y_{b})$ that control adaptive instance normalization(AdaIN)[<a href="#footnote_26_1" name="footnote_26_2">26</a>, <a href="#footnote_16_1" name="footnote_16_2">16</a>, <a href="#footnote_20_1" name="footnote_20_2">20</a>, <a href="#footnote_15_1" name="footnote_15_2">15</a>] operations after each convolution layer of the synthesis network $g$. 
>> 그런 다음 학습된 아핀 변환은 합성 네트워크 $g$의 각 컨볼루션 레이어 이후 적응형 인스턴스 정규화(AdaIN)[<a href="#footnote_26_1" name="footnote_26_2">26</a>, <a href="#footnote_16_1" name="footnote_16_2">16</a>, <a href="#footnote_20_1" name="footnote_20_2">20</a>, <a href="#footnote_15_1" name="footnote_15_2">15</a>] 연산을 제어하는 $y=(y_{s},y_{b})$ 스타일로 $w$를 특수화한다.

> The AdaIN operation is defined as
>> AdaIN연산은 다음과 같이 정의된다.

$$\mathrm{AdalN}(x_{i},y)=y_{s,i}\frac{x_{i}-\mu{(x_{i})}}{\sigma{(x_{i})}}+y_{b,i},$$

> where each feature map $x_{i}$ is normalized separately, and then scaled and biased using the corresponding scalar components from style $y$. Thus the dimensionality of $y$ is twice the number of feature maps on that layer.
>> 여기서 각 형상 맵 $x_{i}$는 별도로 정규화된 다음 스타일 $y$의 해당 스칼라 구성 요소를 사용하여 스케일링 및 편향된다. 따라서 $y$의 차원은 해당 레이어의 기능 맵 수의 두 배이다.

> Comparing our approach to style transfer, we compute the spatially invariant style $y$ from vector $w$ instead of an example image. 
>> 스타일 전송에 대한 우리의 접근 방식을 비교하여, 우리는 예제 이미지 대신 벡터 $w$에서 공간 불변 스타일 $y$를 계산한다. 

> We choose to reuse the word “style” for $y$ because similar network architectures are already used for feedforward style transfer[<a href="#footnote_26_1" name="footnote_26_2">26</a>], unsupervised image-toimage translation[<a href="#footnote_27_1" name="footnote_27_2">27</a>], and domain mixtures[<a href="#footnote_22_1" name="footnote_22_2">22</a>]. 
>> 유사한 네트워크 아키텍처가 피드포워드 스타일 전송[<a href="#footnote_26_1" name="footnote_26_2">26</a>], 감독되지 않은 이미지 간 변환[<a href="#footnote_27_1" name="footnote_27_2">27</a>] 및 도메인 혼합[<a href="#footnote_22_1" name="footnote_22_2">22</a>]에 이미 사용되고 있기 때문에 $y$에 대해 "스타일"이라는 단어를 재사용하기로 선택했다.

> Compared to more general feature transforms[<a href="#footnote_35_1" name="footnote_35_2">35</a>, <a href="#footnote_53_1" name="footnote_53_2">53</a>], AdaIN is particularly well suited for our purposes due to its efficiency and compact representation.
>>  보다 일반적인 형상 변환[<a href="#footnote_35_1" name="footnote_35_2">35</a>, <a href="#footnote_53_1" name="footnote_53_2">53</a>]과 비교했을 때, AdaIN은 효율성과 콤팩트한 표현으로 인해 우리의 목적에 특히 적합하다.

![Table 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Table-1.JPG)

> Table 1. Fréchet inception distance (FID) for various generator designs (lower is better). In this paper we calculate the FIDs using 50,000 images drawn randomly from the training set, and report the lowest distance encountered over the course of training.
>> 표 1. 다양한 발전기 설계에 대한 프레셰 개시 거리(FID). 본 논문에서는 훈련 세트에서 무작위로 추출한 50,000개의 이미지를 사용하여 FID를 계산하고, 훈련 과정에서 발생한 가장 낮은 거리를 보고한다.

> Finally, we provide our generator with a direct means to generate stochastic detail by introducing explicit noise inputs. 
>> 마지막으로, 우리는 명시적 노이즈 입력을 도입하여 확률적 세부 정보를 생성할 수 있는 직접적인 수단을 생성기에 제공한다. 

> These are single-channel images consisting of uncorrelated Gaussian noise, and we feed a dedicated noise image to each layer of the synthesis network.
>> 이것들은 상관없는 가우스 노이즈로 구성된 단일 채널 이미지이며, 우리는 합성 네트워크의 각 레이어에 전용 노이즈 이미지를 공급한다. 

> The noise image is broadcasted to all feature maps using learned per-feature scaling factors and then added to the output of the corresponding convolution, as illustrated in Figure 1b. 
>> 노이즈 이미지는 학습된 기능별 스케일링 계수를 사용하여 모든 기능 맵에 브로드캐스트된 다음 그림 1b와 같이 해당 컨볼루션의 출력에 추가됩니다. 

> The implications of adding the noise inputs are discussed in Sections 3.2 and 3.3.
>> 소음 입력 추가의 의미는 섹션 3.2 및 3.3에 설명되어 있습니다.

#### $\mathbf{2.1.\;Quality\;of\;generated\;images}$

> Before studying the properties of our generator, we demonstrate experimentally that the redesign does not compromise image quality but, in fact, improves it considerably. 
>> 발전기의 특성을 연구하기 전에, 우리는 재설계가 이미지 품질을 손상시키지 않지만, 실제로 상당히 개선된다는 것을 실험적으로 입증한다. 

> Table 1 gives Fréchet inception distances (FID)[<a href="#footnote_24_1" name="footnote_24_2">24</a>] for various generator architectures in CelebA-HQ[<a href="#footnote_28_1" name="footnote_28_2">28</a>] and our new FFHQ dataset (Appendix A). 
>> 표 1은 CellebA-HQ[<a href="#footnote_28_1" name="footnote_28_2">28</a>]의 다양한 발전기 아키텍처와 우리의 새로운 FFHQ 데이터 세트(부록 A)에 대한 프레셰 개시 거리(FID)[<a href="#footnote_24_1" name="footnote_24_2">24</a>]를 제공한다. 

> Results for other datasets are given in the supplement. 
>> 다른 데이터 세트에 대한 결과는 부록에 제시되어 있다.

> Our baseline configuration (a) is the Progressive GAN setup of Karras et al.[<a href="#footnote_28_1" name="footnote_28_2">28</a>], from which we inherit the networks and all hyperparameters except where stated otherwise. 
>> 우리의 기본 구성(a)은 Karras 등의 Progressive GAN 설정이며[<a href="#footnote_28_1" name="footnote_28_2">28</a>], 여기서 별도로 명시된 경우를 제외하고 네트워크와 모든 하이퍼 파라미터를 상속한다

> We first switch to an improved baseline (b) by using bilinear up/downsampling operations[<a href="#footnote_58_1" name="footnote_58_2">58</a>], longer training, and tuned hyperparameters.
>> 먼저 이중 선형 상향/하향 샘플링 작업[<a href="#footnote_58_1" name="footnote_58_2">58</a>], 더 긴 훈련 및 튜닝된 하이퍼 매개 변수를 사용하여 향상된 기준선(b)으로 전환한다.

> A detailed description of training setups and hyperparameters is included in the supplement. 
>> 교육 설정 및 하이퍼 파라미터에 대한 자세한 설명은 부록에 포함되어 있습니다. 

> We then improve this new baseline further by adding the mapping network and AdaIN operations (c), and make a surprising observation that the network no longer benefits from feeding the latent code into the first convolution layer. 
>> 그런 다음 매핑 네트워크와 AdaIN를 추가하여 이 새로운 기준선을 더욱 개선한다. 작업(c)에서, 그리고 네트워크가 더 이상 제1 컨볼루션 레이어에 잠재 코드를 공급함으로써 이익을 얻지 않는다는 놀라운 관찰을 한다. 

> We therefore simplify the architecture by removing the traditional input layer and starting the image synthesis from a learned 4 × 4 × 512 constant tensor(d). 
>> 따라서 기존의 입력 계층을 제거하고 학습된 4 × 4 × 512 상수 텐서(d)에서 이미지 합성을 시작하여 아키텍처를 단순화한다. 

> We find it quite remarkable that the synthesis network is able to produce meaningful results even though it receives input only through the styles that control the AdaIN operations.
>> 합성 네트워크가 AdaIN 연산을 제어하는 스타일을 통해서만 입력을 수신하더라도 의미 있는 결과를 낼 수 있다는 것은 매우 주목할 만하다.

> Finally, we introduce the noise inputs $(e)$. that improve the results further, as well as novel mixing regularization $(f)$ that decorrelates neighboring styles and enables more finegrained control over the generated imagery (Section 3.1).
>> 마지막으로 결과를 더욱 향상시키는 노이즈 입력 $(e)$뿐만 아니라 인접 스타일을 장식하고 생성된 이미지에 대한 보다 세밀한 제어를 가능하게 하는 새로운 혼합 정규화 $(f)$를 소개한다(섹션 3.1).

> We evaluate our methods using two different loss functions: for CelebA-HQ we rely on WGAN-GP[<a href="#footnote_23_1" name="footnote_23_2">23</a>], while FFHQ uses WGAN-GP for configuration a and nonsaturating loss <a href="#footnote_21_1" name="footnote_21_2">[21]</a> with $R_{1}$ regularization[<a href="#footnote_40_1" name="footnote_40_2">40</a>, <a href="#footnote_47_1" name="footnote_47_2">47</a>, <a href="#footnote_13_1" name="footnote_13_2">1</a>] for configurations $b–f$. We found these choices to give the best results. Our contributions do not modify the loss function.
>> 우리는 CelebA-HQ의 경우 WGAN-GP[<a href="#footnote_23_1" name="footnote_23_2">23</a>]에 의존하는 반면, FFHQ는 구성 a에 WGAN-GP를 사용하고 구성 $b–f$에 대해 $R_{1}$ 정규화[<a href="#footnote_40_1" name="footnote_40_2">40</a>, <a href="#footnote_47_1" name="footnote_47_2">47</a>, <a href="#footnote_13_1" name="footnote_13_2">1</a>]를 사용하는 비포화 손실<a href="#footnote_21_1" name="footnote_21_2">[21]</a>를 사용한다. 우리는 최상의 결과를 얻기 위해 이러한 선택들을 찾았다. 우리의 기여는 손실 함수를 수정하지 않는다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Figure-2.JPG)

> Figure 2. Uncurated set of images produced by our style-based generator (config $f$) with the FFHQ dataset. Here we used a variation of the truncation trick[<a href="#footnote_38_1" name="footnote_38_2">38</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>, <a href="#footnote_31_1" name="footnote_31_2">31</a>] with $\psi{}=0.7$ for resolutions $4^{2}−32^{2}$. Please see the accompanying video for more results.
>> 그림 2. 스타일 기반 생성기(config $f$)가 FFHQ 데이터 세트를 사용하여 생성한 미수정 이미지 세트. 여기서 우리는 해상도 $4^{2}-32^{2}$에 대해 $\psi{}=0.7$와 함께 절단 트릭k[<a href="#footnote_38_1" name="footnote_38_2">38</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>, <a href="#footnote_31_1" name="footnote_31_2">31</a>]의 변형을 사용했다. 자세한 결과를 보려면 동봉된 비디오를 참조하십시오.

> We observe that the style-based generator $(E)$ improves FIDs quite significantly over the traditional generator (b), almost 20%, corroborating the large-scale ImageNet measurements made in parallel work[<a href="#footnote_5_1" name="footnote_5_2">5</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>]. 
>> 스타일 기반 생성기 $(E)$는 기존 생성기(b)에 비해 FID를 상당히 향상시켜 거의 20%까지 향상시켜 병렬 작업에서 수행된 대규모 ImageNet 측정을 뒷받침한다는 것을 관찰한다[<a href="#footnote_5_1" name="footnote_5_2">5</a>,<a href="#footnote_4_1" name="footnote_4_2">4</a>].

> Figure 2 shows an uncurated set of novel images generated from the FFHQ dataset using our generator. 
>> 그림 2는 생성기를 사용하여 FFHQ 데이터 세트에서 생성된 미가공된 새로운 이미지 세트를 보여준다. 

> As confirmed by the FIDs, the average quality is high, and even accessories such as eyeglasses and hats get successfully synthesized. 
>> FID에서 확인되었듯이, 평균적인 품질은 높고, 안경이나 모자 같은 액세서리까지도 성공적으로 합성된다. 

> For this figure, we avoided sampling from the extreme regions of $w$ using the so-called truncation trick[<a href="#footnote_38_1" name="footnote_38_2">38</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>, <a href="#footnote_31_1" name="footnote_31_2">31</a>]— Appendix B details how the trick can be performed in $w$ instead of $Z$. 
>> 이 그림에서 우리는 소위 잘라내기 트릭[<a href="#footnote_38_1" name="footnote_38_2">38</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>, <a href="#footnote_31_1" name="footnote_31_2">31</a>]을 사용하여 $w$의 극한 영역에서 샘플링하는 것을 피했다- 부록 B는 $Z$ 대신 $w$에서 트릭을 수행할 수 있는 방법을 자세히 설명한다.

> Note that our generator allows applying the truncation selectively to low resolutions only, so that high-resolution details are not affected.
>> 당사 생성기는 고해상도 세부 정보에 영향을 미치지 않도록 낮은 해상도에만 절단을 선택적으로 적용할 수 있다.

> All FIDs in this paper are computed without the truncation trick, and we only use it for illustrative purposes in Figure 2 and the video. All images are generated in $1024^{2}$ resolution.
>> 이 논문의 모든 FID는 잘라내기 트릭 없이 계산되며, 우리는 그림 2와 비디오의 예시적인 목적으로만 사용한다. 모든 영상은 $1024^{2}$ 해상도로 생성됩니다.

#### $\mathbf{2.2.\;Prior\;art}$

> Much of the work on GAN architectures has focused on improving the discriminator by, e.g., using multiple discriminators[<a href="#footnote_17_1" name="footnote_17_2">17</a>, <a href="#footnote_43_1" name="footnote_43_2">43</a>, <a href="#footnote_10_1" name="footnote_10_2">10</a>], multiresolution discrimination[<a href="#footnote_55_1" name="footnote_55_2">55</a>, <a href="#footnote_51_1" name="footnote_51_2">51</a>], or self-attention[57]. 
>> GAN 아키텍처에 대한 많은 연구는 예를 들어 다중 판별기[<a href="#footnote_17_1" name="footnote_17_2">17</a>, <a href="#footnote_43_1" name="footnote_43_2">43</a>, <a href="#footnote_10_1" name="footnote_10_2">10</a>], 다중 해상도 판별기[<a href="#footnote_55_1" name="footnote_55_2">55</a>, <a href="#footnote_51_1" name="footnote_51_2">51</a>] 또는 자기 주의[57]를 사용하여 판별기 개선에 초점을 맞추고 있다. 

> The work on generator side has mostly focused on the exact distribution in the input latent space[<a href="#footnote_4_1" name="footnote_4_2">3</a>] or shaping the input latent space via Gaussian mixture models[<a href="#footnote_3_1" name="footnote_3_2">3</a>], clustering[<a href="#footnote_44_1" name="footnote_44_2">44</a>], or encouraging convexity[<a href="#footnote_48_1" name="footnote_48_2">48</a>].
>> 발생기 측면에 대한 연구는 주로 입력 잠재 공간[<a href="#footnote_4_1" name="footnote_4_2">3</a>]의 정확한 분포 또는 가우스 혼합 모델[<a href="#footnote_3_1" name="footnote_3_2">3</a>], 클러스터링[<a href="#footnote_44_1" name="footnote_44_2">44</a>] 또는 볼록성을 장려하는[<a href="#footnote_48_1" name="footnote_48_2">48</a>]을 통해 입력 잠재 공간을 형성하는 데 중점을 두었다.

> Recent conditional generators feed the class identifier through a separate embedding network to a large number of layers in the generator[<a href="#footnote_42_1" name="footnote_42_2">42</a>], while the latent is still provided though the input layer. A few authors have considered feeding parts of the latent code to multiple generator layers[8, 4]. 
>> 최근의 조건부 생성기는 별도의 임베딩 네트워크를 통해 생성기의 많은 레이어에 클래스 식별자를 공급하지만 [<a href="#footnote_42_1" name="footnote_42_2">42</a>]은 여전히 입력 레이어를 통해 제공된다. 몇몇 저자들은 잠재 코드의 일부를 여러 발전기 계층[8, 4]에 공급하는 것을 고려했다. 

> In parallel work, Chen et al.[<a href="#footnote_5_1" name="footnote_5_2">5</a>] “self modulate” the generator using AdaINs, similarly to our work, but do not consider an intermediate latent space or noise inputs.
>> 병렬 작업에서는 Chen 등이 있습니다.[<a href="#footnote_5_1" name="footnote_5_2">5</a>] 우리의 작업과 유사하게, AdaIN를 사용하여 발전기를 "자체 변조"하지만, 중간 잠재 공간이나 소음 입력을 고려하지 않는다.

### $\mathbf{3.\;Properties\;of\;the\;style-based\;generator}$

> Our generator architecture makes it possible to control the image synthesis via scale-specific modifications to the styles. 
>> 우리의 생성기 아키텍처는 스타일에 대한 스케일별 수정을 통해 이미지 합성을 제어할 수 있게 한다.

> We can view the mapping network and affine transformations as a way to draw samples for each style from a learned distribution, and the synthesis network as a way to generate a novel image based on a collection of styles. 
>> 우리는 매핑 네트워크와 아핀 변환을 학습된 분포에서 각 스타일에 대한 샘플을 추출하는 방법으로 볼 수 있고, 합성 네트워크를 스타일 모음을 기반으로 새로운 이미지를 생성하는 방법으로 볼 수 있다. 

> The effects of each style are localized in the network, i.e., modifying a specific subset of the styles can be expected to affect only certain aspects of the image.
>> 각 스타일의 효과는 네트워크에서 국한된다. 즉, 스타일의 특정 하위 집합을 수정하는 것은 이미지의 특정 측면에만 영향을 미칠 것으로 예상할 수 있다.

> To see the reason for this localization, let us consider how the AdaIN operation (Eq. 1) first normalizes each channel to zero mean and unit variance, and only then applies scales and biases based on the style. 
>> 이러한 현지화의 이유를 보려면 AdaIN 연산(식 1)이 먼저 각 채널을 0의 평균 및 단위 분산으로 정규화한 다음 스타일을 기반으로 척도 및 편향을 적용하는 방법을 고려해보자. 

> The new per-channel statistics, as dictated by the style, modify the relative importance of features for the subsequent convolution operation, but they do not depend on the original statistics because of the normalization. 
>> 스타일에 따라 새로운 채널별 통계는 후속 컨볼루션 연산을 위한 기능의 상대적 중요성을 수정하지만 정규화 때문에 원래 통계에 의존하지 않는다. 

> Thus each style controls only one convolution before being overridden by the next AdaIN operation.
>> 따라서 각 스타일은 다음 AdaIN 연산에 의해 재정의되기 전에 하나의 컨볼루션만 제어한다.
 
#### $\mathbf{3.1.\;Style\;mixing}$

> To further encourage the styles to localize, we employ mixing regularization, where a given percentage of images are generated using two random latent codes instead of one during training. 
>> 스타일의 현지화를 더욱 장려하기 위해, 우리는 혼합 정규화를 사용하며, 여기서 주어진 비율의 이미지는 훈련 중에 하나가 아닌 두 개의 무작위 잠재 코드를 사용하여 생성된다.

> When generating such an image, we simply switch from one latent code to another — an operation we refer to as style mixing — at a randomly selected point in the synthesis network. 
>> 이러한 이미지를 생성할 때, 우리는 합성 네트워크에서 무작위로 선택된 지점에서 하나의 잠재 코드에서 다른 코드(스타일 혼합이라고 하는 작업)로 전환하기만 한다.

> To be specific, we run two latent codes $z_{1}$, $z_{2}$ through the mapping network, and have the corresponding $w_{1}$, $w_{2}$ control the styles so that $w_{1}$ applies before the crossover point and $w_{2}$ after it. 
>>  구체적으로 말하면, 우리는 매핑 네트워크를 통해 두 개의 잠재 코드 $z_{1}$, $z_{2}$를 실행하고, 해당 $w_{1}$, $w_{2}$가 교차점 이전에 적용되고, 그 이후에 $w_{2}$가 적용되도록 스타일을 제어하도록 한다.

> This regularization technique prevents the network from assuming that adjacent styles are correlated. 
>> 이 정규화 기법은 네트워크가 인접한 스타일이 상관관계가 있다고 가정하는 것을 방지한다.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Figure-3.JPG)

> Figure 3. Two sets of images were generated from their respective latent codes (sources A and B); the rest of the images were generated by copying a specified subset of styles from source B and taking the rest from source A. Copying the styles corresponding to coarse spatial resolutions $(4^{2}–8^{2})$ brings high-level aspects such as pose, general hair style, face shape, and eyeglasses from source B, while all colors(eyes, hair, lighting) and finer facial features resemble A. If we instead copy the styles of middle resolutions $(16^{2}–32^{2})$ from B, we inherit smaller scale facial features, hair style, eyes open/closed from B, while the pose, general face shape, and eyeglasses from A are preserved. Finally, copying the fine styles $(64^{2}–1024^{2})$ from B brings mainly the color scheme and microstructure.
>> 그림 3. 두 세트의 이미지는 각각의 잠재 코드(소스 A와 소스 B)에서 생성되었으며, 나머지 이미지는 소스 B에서 특정 스타일의 하위 집합을 복사하고 소스 A에서 나머지를 가져와서 생성되었다. 거친 공간 해상도 $(4^{2}-8^{2})$ 에 해당하는 스타일을 복사하면 포즈, 일반적인 헤어 스타일, 얼굴 모양, 안경과 같은 높은 수준의 측면이 소스 B에서 제공되며, 모든 색상(눈, 머리, 조명)과 더 미세한 얼굴 특징은 A와 유사하다. 대신 중간 해상도 $(16^{2}–32^{2})$ 의 스타일을 B로부터 복사하면, 우리는 더 작은 규모의 얼굴 특징, 머리 모양, 눈을 뜨고 감는 것을 B로부터 물려받으며, 포즈, 일반적인 얼굴 모양, 그리고 A로부터 안경을 보존한다. 마지막으로, B에서 파인 스타일 $(64^{2}–1024^{2})$을 복사하면 주로 색 구성표와 미세 구조가 나타납니다.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Figure-4.JPG)

> Figure 4. Examples of stochastic variation. (a) Two generated images. (b) Zoom-in with different realizations of input noise. While the overall appearance is almost identical, individual hairs are placed very differently. (c) Standard deviation of each pixel over 100 different realizations, highlighting which parts of the images are affected by the noise. The main areas are the hair, silhouettes, and parts of background, but there is also interesting stochastic variation in the eye reflections. Global aspects such as identity and pose are unaffected by stochastic variation.
>> 그림 4. 확률적 변동의 예. (a) 생성된 두 개의 이미지. (b) 입력 노이즈의 다른 실현으로 줌인한다. 전체적인 외관은 거의 동일하지만, 각각의 털은 매우 다르게 배치된다. (c) 100가지 다른 실현에 걸쳐 각 픽셀의 표준 편차를 통해 이미지의 어떤 부분이 노이즈의 영향을 받는지 강조한다. 주요 부위는 머리카락, 실루엣, 배경 부분이지만 눈의 반사에도 흥미로운 확률적 변화가 있다. 정체성 및 포즈와 같은 전역적 측면은 확률적 변동의 영향을 받지 않는다.

![Table 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Table-2.JPG)

> Table 2. FIDs in FFHQ for networks trained by enabling the mixing regularization for different percentage of training examples. Here we stress test the trained networks by randomizing $1\cdots4$ latents and the crossover points between them. Mixing regularization improves the tolerance to these adverse operations significantly. Labels e and f refer to the configurations in Table 1.
>> 표 2. FFHQ의 FID는 다양한 비율의 훈련 예제에 대한 혼합 정규화를 활성화하여 훈련된 네트워크를 위한 것이다. 여기서는 $1\cdots4$ 잠재성과 이들 사이의 교차점을 랜덤화하여 훈련된 네트워크를 테스트한다. 혼합 정규화는 이러한 역작업에 대한 공차를 크게 향상시킵니다. 라벨 $e$ 및 $f$는 표 1의 구성을 참조합니다.

> Table 2 shows how enabling mixing regularization during training improves the localization considerably, indicated by improved FIDs in scenarios where multiple latents are mixed at test time. 
>> 표 2는 훈련 중에 혼합 정규화를 활성화하면 시험 시간에 여러 잠재력이 혼합되는 시나리오에서 개선된 FID로 나타나듯이 현지화가 상당히 개선되는 방법을 보여준다. 

> Figure 3 presents examples of images synthesized by mixing two latent codes at various scales. We can see that each subset of styles controls meaningful high-level attributes of the image.
>> 그림 3은 다양한 스케일로 두 개의 잠재 코드를 혼합하여 합성된 이미지의 예를 보여줍니다. 스타일의 각 하위 집합이 이미지의 의미 있는 고급 속성을 제어한다는 것을 알 수 있습니다.

#### $\mathbf{3.2.\;Stochastic\;variation}$

> There are many aspects in human portraits that can be regarded as stochastic, such as the exact placement of hairs, stubble, freckles, or skin pores. 
>> 인간의 초상화에는 머리카락, 그루터기, 주근깨, 피부 모공의 정확한 배치와 같이 확률적인 것으로 간주될 수 있는 많은 측면이 있다. 

> Any of these can be randomized without affecting our perception of the image as long as they follow the correct distribution.
>> 올바른 분포를 따르는 한 이미지에 대한 우리의 인식에 영향을 주지 않고 이들 중 어느 것도 무작위화할 수 있다.

> Let us consider how a traditional generator implements stochastic variation. 
>> 전통적인 생성기가 확률적 변동을 구현하는 방법을 고려해보자. 

> Given that the only input to the network is through the input layer, the network needs to invent a way to generate spatially-varying pseudorandom numbers from earlier activations whenever they are needed. 
>> 네트워크에 대한 유일한 입력이 입력 계층을 통해서라는 것을 고려할 때, 네트워크는 필요할 때마다 이전 활성화에서 공간적으로 변하는 의사 난수를 생성하는 방법을 발명할 필요가 있다. 

> This consumes network capacity and hiding the periodicity of generated signal is difficult — and not always successful, as evidenced by commonly seen repetitive patterns in generated images. 
>> 이것은 네트워크 용량을 소모하고 생성된 신호의 주기성을 숨기는 것은 어렵고, 생성된 이미지에서 흔히 볼 수 있는 반복 패턴에서 입증되듯이 항상 성공하지는 않는다. 

> Our architecture sidesteps these issues altogether by adding per-pixel noise after each convolution.
>> 우리의 아키텍처는 각 컨볼루션 후에 픽셀당 노이즈를 추가하여 이러한 문제를 완전히 회피한다.

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Figure-5.JPG)

> Figure 5. Effect of noise inputs at different layers of our generator. (a) Noise is applied to all layers. (b) No noise. (c) Noise in fine layers only $(64^{2}–1024^{2})$. (d) Noise in coarse layers only $(4^{2}–32^{2})$.We can see that the artificial omission of noise leads to featureless “painterly” look. Coarse noise causes large-scale curling of hairand appearance of larger background features, while the fine noise brings out the finer curls of hair, finer background detail, and skinpores.
>> 그림 5. 발전기의 다른 층에서의 노이즈 입력의 영향. (a) 노이즈는 모든 층에 적용된다. (b) 노이즈는 없다. (c) 미세한 층의 소음은 $(64^{2}–1024^{2})$뿐입니다. (d) 거친 층의 소음은 $(4^{2}–32^{2})$입니다.우리는 노이즈의 인위적인 누락이 특징 없는 "페인터리" 룩으로 이어진다는 것을 알 수 있다. 거친 소음은 머리카락의 대규모 컬링과 더 큰 배경 특징의 외관을 야기하는 반면, 미세한 소음은 머리카락의 더 미세한 컬, 더 미세한 배경 디테일, 그리고 피부 모공을 가져온다.

> Figure 4 shows stochastic realizations of the same underlying image, produced using our generator with different noise realizations. We can see that the noise affects only the stochastic aspects, leaving the overall composition and high-level aspects such as identity intact. Figure 5 further illustrates the effect of applying stochastic variation to different subsets of layers. Since these effects are best seen in animation, please consult the accompanying video for a demonstration of how changing the noise input of one layer leads to stochastic variation at a matching scale.
>> 그림 4는 노이즈 실현이 다른 발전기를 사용하여 생성된 동일한 기본 이미지의 확률적 실현을 보여준다. 노이즈가 확률적 측면에만 영향을 미쳐 전체 구성과 동일성과 같은 높은 수준의 측면은 그대로 둔다는 것을 알 수 있다. 그림 5는 다양한 계층 하위 집합에 확률적 변동을 적용하는 효과를 추가로 보여준다. 이러한 효과는 애니메이션에서 가장 잘 나타나기 때문에, 한 층의 노이즈 입력이 어떻게 일치하는 규모로 확률적 변동을 초래하는지 시연하려면 동봉된 비디오를 참조하십시오.

> We find it interesting that the effect of noise appears tightly localized in the network. We hypothesize that at any point in the generator, there is pressure to introduce new content as soon as possible, and the easiest way for our network to create stochastic variation is to rely on the noise provided. A fresh set of noise is available for every layer, and thus there is no incentive to generate the stochastic effects from earlier activations, leading to a localized effect.
>> 우리는 노이즈의 효과가 네트워크에서 밀접하게 국부적으로 나타난다는 것을 흥미롭게 생각한다. 우리는 발전기의 어느 지점에서든 가능한 한 빨리 새로운 콘텐츠를 도입해야 한다는 압력이 있으며, 우리 네트워크가 확률적 변화를 만드는 가장 쉬운 방법은 제공된 노이즈에 의존하는 것이라고 가정한다. 모든 계층에 대해 새로운 노이즈 세트를 사용할 수 있으므로 초기 활성화에서 확률적 효과를 생성하여 국부적 효과를 발생시킬 동기가 없다.

![Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Figure-6.JPG)

> Figure 6. Illustrative example with two factors of variation (image features, e.g., masculinity and hair length). (a) An example training set where some combination (e.g., long haired males) is missing. (b) This forces the mapping from $Z$ to image features to become curved so that the forbidden combination disappears in $Z$ to prevent the sampling of invalid combinations. (c) The learned mapping from $Z$ to $W$ is able to “undo” much of the warping.
>> 그림 6. 두 가지 변동 요인(예: 남성성 및 머리 길이)이 있는 예. (a) 일부 조합(예: 긴 머리 남성)이 누락된 예제 훈련 세트. (b) 이것은 $Z$에서 이미지 기능에 대한 매핑을 강제로 곡선이 되도록 하여 금지된 조합이 $Z$에서 사라지게 하여 잘못된 조합의 샘플링을 방지한다. (c) $Z$에서 $W$로 학습된 매핑은 워핑의 많은 부분을 "해제"할 수 있다.

#### $\mathbf{3.3.\;Separation\;of\;global\;effects\;from\;stochasticity}$

> The previous sections as well as the accompanying video demonstrate that while changes to the style have global effects (changing pose, identity, etc.), the noise affects only inconsequential stochastic variation (differently combed hair, beard, etc.). This observation is in line with style transfer literature, where it has been established that spatially invariant statistics (Gram matrix, channel-wise mean, variance, etc.) reliably encode the style of an image[19, 36] while spatially varying features encode a specific instance.
>> 이전 섹션과 함께 제공되는 비디오는 스타일의 변경이 전역적 영향(포즈, 정체성 변경 등)을 가지지만 노이즈는 중요하지 않은 확률적 변화(다르게 빗은 머리, 수염 등)에만 영향을 미친다는 것을 보여준다. 이러한 관찰은 스타일 전송 문헌과 일치하며, 공간적으로 다양한 기능이 특정 인스턴스를 인코딩하는 동안 공간적으로 불변 통계(Gram matrix, 채널별 평균, 분산 등)가 이미지의 스타일을 안정적으로 인코딩한다는 것이 확립되었다.

> In our style-based generator, the style affects the entire image because complete feature maps are scaled and biased with the same values. Therefore, global effects such as pose, lighting, or background style can be controlled coherently. Meanwhile, the noise is added independently to each pixel and is thus ideally suited for controlling stochastic variation. If the network tried to control, e.g., pose using the noise, that would lead to spatially inconsistent decisions that would then be penalized by the discriminator. Thus the network learns to use the global and local channels appropriately, without explicit guidance.
>> 스타일 기반 생성기에서 스타일은 전체 피쳐 맵이 동일한 값으로 조정되고 편향되기 때문에 전체 이미지에 영향을 미친다. 따라서 포즈, 조명 또는 배경 스타일과 같은 전역 효과를 일관성 있게 제어할 수 있습니다. 한편, 노이즈는 각 픽셀에 독립적으로 추가되므로 확률적 변동을 제어하는 데 이상적이다. 네트워크가 예를 들어 노이즈를 사용하여 자세를 제어하려고 할 경우, 이는 공간적으로 일관성이 없는 결정을 초래하여 판별자에 의해 불이익을 받게 될 것이다. 따라서 네트워크는 명시적 지침 없이 글로벌 및 로컬 채널을 적절하게 사용하는 방법을 학습한다.

### $\mathbf{4.\;Disentanglement\;studies}$

> There are various definitions for disentanglement[<a href="#footnote_50_1" name="footnote_50_2">50</a>, <a href="#footnote_46_1" name="footnote_46_2">46</a>, <a href="#footnote_1_1" name="footnote_1_2">1</a>, <a href="#footnote_6_1" name="footnote_6_2">6</a>, <a href="#footnote_18_1" name="footnote_18_2">18</a>], but a common goal is a latent space that consists of linear subspaces, each of which controls one factor of variation. However, the sampling probability of each combination of factors in $z$ needs to match the corresponding density in the training data. As illustrated in Figure 6, this precludes the factors from being fully disentangled with typical datasets and input latent distributions.2
>> 분리[<a href="#footnote_50_1" name="footnote_50_2">50</a>, <a href="#footnote_46_1" name="footnote_46_2">46</a>, <a href="#footnote_1_1" name="footnote_1_2">1</a>, <a href="#footnote_6_1" name="footnote_6_2">6</a>, <a href="#footnote_18_1" name="footnote_18_2">18</a>]에 대한 정의는 다양하지만, 공통 목표는 선형 부분공간으로 구성된 잠재 공간이며, 각 부분공간은 하나의 변동 요인을 제어한다. 그러나 $z$의 각 요인 조합의 샘플링 확률은 훈련 데이터의 해당 밀도와 일치해야 한다. 그림 6에 나타난 바와 같이, 이는 요인이 일반적인 데이터 세트 및 입력 잠재 분포와 완전히 분리되는 것을 방지한다.

> A major benefit of our generator architecture is that the intermediate latent space $w$ does not have to support sampling according to any fixed distribution; its sampling density is induced by the learned piecewise continuous mapping $f(z)$. This mapping can be adapted to “unwarp” $w$ so that the factors of variation become more linear. We posit that there is pressure for the generator to do so, as it should be easier to generate realistic images based on a disentangled representation than based on an entangled representation. As such, we expect the training to yield a less entangled $w$ in an unsupervised setting, i.e., when the factors of variation are not known in advance[<a href="#footnote_9_1" name="footnote_9_2">9</a>, <a href="#footnote_32_1" name="footnote_32_2">32</a>, <a href="#footnote_45_1" name="footnote_45_2">45</a>, <a href="#footnote_7_1" name="footnote_7_2">7</a>, 25, 30, 6].
>> 우리 발전기 아키텍처의 주요 이점은 중간 잠재 공간 $w$가 고정된 분포에 따라 샘플링을 지원할 필요가 없다는 것이다. 샘플링 밀도는 학습된 조각별 연속 매핑 $f(z)$에 의해 유도된다. 이 매핑은 변동 요인이 더 선형적이 되도록 $w$를 "뒤틀림 해제"하도록 조정할 수 있다. 우리는 얽힌 표현을 기반으로 하는 것보다 얽힌 표현을 기반으로 현실적인 이미지를 생성하는 것이 더 쉬워야 하기 때문에 생성기가 그렇게 해야 한다는 압력이 있다고 가정한다. 이와 같이, 우리는 훈련이 비지도 설정에서 덜 얽힌 $w$를 산출할 것으로 예상한다. 즉, 변동 요인을 미리 알 수 없을 때[<a href="#footnote_9_1" name="footnote_9_2">9</a>, <a href="#footnote_32_1" name="footnote_32_2">32</a>, <a href="#footnote_45_1" name="footnote_45_2">45</a>, <a href="#footnote_7_1" name="footnote_7_2">7</a>, 25, 30, 6]

> Unfortunately the metrics recently proposed for quantifying disentanglement[25, 30, 6, 18] require an encoder network that maps input images to latent codes. These metrics are ill-suited for our purposes since our baseline GAN lacks such an encoder. While it is possible to add an extra network for this purpose[<a href="#footnote_7_1" name="footnote_7_2">7</a>, <a href="#footnote_11_1" name="footnote_11_2">11</a>, <a href="#footnote_14_1" name="footnote_14_2">14</a>], we want to avoid investing effort into a component that is not a part of the actual solution. To this end, we describe two new ways of quantifying disentanglement, neither of which requires an encoder or known factors of variation, and are therefore computable for any image dataset and generator.
>> 불행하게도 분리를 정량화하기 위해 최근 제안된 메트릭[25, 30, 6, 18]에는 입력 이미지를 잠재 코드에 매핑하는 인코더 네트워크가 필요하다. 기본 GAN에는 이러한 인코더가 없기 때문에 이러한 메트릭은 우리의 목적에 적합하지 않다. 이러한 목적을 위해 추가 네트워크를 추가할 수 있지만[<a href="#footnote_7_1" name="footnote_7_2">7</a>, <a href="#footnote_11_1" name="footnote_11_2">11</a>, <a href="#footnote_14_1" name="footnote_14_2">14</a>] 실제 솔루션의 일부가 아닌 구성 요소에 대한 투자 노력은 피하고 싶다. 이를 위해 인코더나 알려진 변동 요인을 필요로 하지 않으므로 이미지 데이터 세트 및 생성기에 대해 계산할 수 있는 두 가지 새로운 분리 방법을 설명한다.

#### $\mathbf{4.1.\;Perceptual\;path\;length}$

> As noted by Laine[<A HREF="#FOOTNOTE_34_1" NAME="FOOTNOTE_34_2">34</A>], interpolation of latent-space vectors may yield surprisingly non-linear changes in the image. For example, features that are absent in either endpoint may appear in the middle of a linear interpolation path. This is a sign that the latent space is entangled and the factors of variation are not properly separated. To quantify this effect, we can measure how drastic changes the image undergoes as we perform interpolation in the latent space. Intuitively, a less curved latent space should result in perceptually smoother transition than a highly curved latent space.
>> Laine[<A HREF="#FOOTNOTE_34_1" NAME="FOOTNOTE_34_2">34</A>]이 지적한 바와 같이, 잠재 공간 벡터의 보간은 이미지에 놀랄 만큼 비선형적인 변화를 초래할 수 있다. 예를 들어, 두 끝점에 없는 피쳐는 선형 보간 경로 중간에 나타날 수 있습니다. 잠복공간이 얽혀 변동요인이 제대로 분리되지 않았다는 신호다. 이 효과를 정량화하기 위해 잠재 공간에서 보간을 수행할 때 이미지가 얼마나 급격한 변화를 겪는지 측정할 수 있다. 직관적으로, 덜 구부러진 잠재 공간은 고도로 구부러진 잠재 공간보다 지각적으로 더 부드러운 전이를 초래해야 한다.

> As a basis for our metric, we use a perceptually-based pairwise image distance[59] that is calculated as a weighted difference between two VGG16[54] embeddings, where the weights are fit so that the metric agrees with human perceptual similarity judgments. If we subdivide a latent space interpolation path into linear segments, we can define the total perceptual length of this segmented path as the sum of perceptual differences over each segment, as reported by the image distance metric. A natural definition for the perceptual path length would be the limit of this sum under infinitely fine subdivision, but in practice we approximate it using a small subdivision epsilon $\epsilon{}=10^{-4}$ . The average perceptual path length in latent space $Z$, over all possible endpoints, is therefore
>> 메트릭의 기준으로, 우리는 두 개의 VGG16[54] 임베딩 사이의 가중치 차이로 계산되는 지각 기반 쌍별 이미지 거리[59]를 사용하는데, 여기서 가중치는 측정값이 인간의 지각 유사성 판단과 일치하도록 적합하다. 잠재 공간 보간 경로를 선형 세그먼트로 세분화하면 이미지 거리 메트릭에 의해 보고되는 대로 이 세그먼트 경로의 총 지각 길이를 각 세그먼트에 대한 지각 차이의 합으로 정의할 수 있다. 지각 경로 길이에 대한 자연스러운 정의는 무한 세분화 하에서의 이 합계의 한계일 것이지만, 실제로 우리는 작은 세분화 엡실론 $\epsilon{}=10^{-4}$을 사용하여 근사한다. 따라서 잠재 공간 $Z$의 모든 가능한 엔드포인트에 대한 평균 지각 경로 길이는 다음과 같다.

$$$$

> where $z_{1},z_{2}\sim{P(z)},t\sim{U(0,1)}, G$ is the generator (i.e., $g\circ{f}$ for style-based networks), and $d(\cdot, \cdot)$ evaluates the perceptual distance between the resulting images. Here slerp denotes spherical interpolation[52], which is the most appropriate way of interpolating in our normalized input latent space[56]. To concentrate on the facial features instead of background, we crop the generated images to contain only the face prior to evaluating the pairwise image metric. As the metric $d$ is quadratic[59], we divide by $\epsilon{}^{2}$ . We compute the expectation by taking 100,000 samples.
>> 여기서 $z_{1},z_{2}\sim{P(z)},t\sim{U(0,1)},G$는 생성기(즉, 스타일 기반 네트워크의 경우 $g\circ{f}$), $d(\cdot, \cdot)$는 결과 이미지 간의 지각 거리를 평가한다. 여기서 slerp는 구면 보간[52]을 나타내며, 이는 정규화된 입력 잠재 공간에서 보간하는 가장 적절한 방법이다[56]. 배경 대신 얼굴 특징에 집중하기 위해 쌍별 이미지 메트릭을 평가하기 전에 생성된 이미지를 얼굴만 포함하도록 자른다. 메트릭 $d$가 2차[59]이므로 $\epsilon{}^{2}$로 나눈다. 100,000개의 샘플을 취하여 기대치를 계산한다.

> Computing the average perceptual path length in $w$ is carried out in a similar fashion:
>> $w$의 평균 지각 경로 길이를 계산하는 방법은 다음과 같다.

> where the only difference is that interpolation happens in $w$ space. Because vectors in $w$ are not normalized in any fashion, we use linear interpolation (lerp)
>> 여기서 유일한 차이점은 보간은 $w$ 공간에서 발생한다는 것이다. $w$의 벡터는 어떤 방식으로도 정규화되지 않기 때문에 선형 보간(lerp)을 사용한다.

![Table 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Table-3.JPG)

> Table 3 shows that this full-path length is substantially shorter for our style-based generator with noise inputs, indicating that $w$ is perceptually more linear than $Z$. Yet, this measurement is in fact slightly biased in favor of the input latent space $Z$. If $w$ is indeed a disentangled and “flattened” mapping of $Z$, it may contain regions that are not on the input manifold— and are thus badly reconstructed by the generator— even between points that are mapped from the input manifold, whereas the input latent space $z$ has no such regions by definition. It is therefore to be expected that if we restrict our measure to path endpoints, i.e., $t\in{(0,1)}$, we should obtain a smaller $l_{W}$ while $l_{Z}$ is not affected. This is indeed what we observe in Table 3
>> 표 3은 이 전체 경로 길이가 노이즈 입력이 있는 스타일 기반 생성기의 경우 상당히 짧다는 것을 보여주며, 이는 $w$가 $Z$보다 지각적으로 더 선형적이라는 것을 나타낸다. 그러나, 이 측정은 실제로 입력 잠재 공간 $Z$에 대해 약간 편향되어 있다. $w$가 실제로 $Z$의 분리되고 "평탄화된" 매핑인 경우, 입력 매니폴드에서 매핑된 포인트 간에도 입력 매니폴드에 없는 영역을 포함할 수 있으며, 따라서 생성기에 의해 잘못 재구성된 영역을 포함할 수 있지만, 입력 잠재 공간 $z$는 정의에 의해 그러한 영역이 없다. 따라서 경로 끝점, 즉 $t\in{(0,1)}$로 측정을 제한하면 $l_{Z}$는 영향을 받지 않지만 더 작은 $l_{W}$를 얻어야 한다. 이것이 실제로 우리가 표 3에서 관찰한 것이다.

![Table 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Table-4.JPG)

> Table 4. The effect of a mapping network in FFHQ. The number in method name indicates the depth of the mapping network. We see that FID, separability, and path length all benefit from having a mapping network, and this holds for both style-based and traditional generator architectures. Furthermore, a deeper mapping network generally performs better than a shallow one 
>> 표 4. FFHQ에서 매핑 네트워크의 효과입니다. 메서드 이름의 숫자는 매핑 네트워크의 깊이를 나타냅니다. 우리는 FID, 분리 가능성 및 경로 길이가 모두 매핑 네트워크를 갖는 것으로부터 이익을 얻으며 이는 스타일 기반과 전통적인 생성기 아키텍처 모두에 적용된다. 게다가, 더 깊은 매핑 네트워크는 일반적으로 얕은 네트워크보다 더 나은 성능을 발휘한다.

> Table 4 shows how path lengths are affected by the mapping network. We see that both traditional and style-based generators benefit from having a mapping network, and additional depth generally improves the perceptual path length as well as FIDs. It is interesting that while $l_{W}$ improves in the traditional generator, $l_{Z}$ becomes considerably worse, illustrating our claim that the input latent space can indeed be arbitrarily entangled in GANs
>> 표 4는 경로 길이가 매핑 네트워크의 영향을 받는 방식을 보여줍니다. 우리는 전통적인 생성기와 스타일 기반 생성기 모두 매핑 네트워크를 갖는 것으로부터 이익을 얻으며, 추가 깊이는 일반적으로 FID뿐만 아니라 지각 경로 길이를 향상시킨다. 흥미로운 점은 $l_{W}$가 기존 발전기에서 개선되지만 $l_{Z}$는 상당히 악화되어 입력 잠재 공간이 실제로 GAN에 임의로 얽힐 수 있다는 우리의 주장을 보여준다.


#### $\mathbf{4.2.\;Linear\;separability}$

> If a latent space is sufficiently disentangled, it should be possible to find direction vectors that consistently correspond to individual factors of variation. We propose another metric that quantifies this effect by measuring how well the latent-space points can be separated into two distinct sets via a linear hyperplane, so that each set corresponds to a specific binary attribute of the image
>> 잠재 공간이 충분히 풀린다면, 개별 변동 요인에 일관되게 대응하는 방향 벡터를 찾을 수 있을 것이다. 우리는 잠재 공간 포인트가 선형 초평면을 통해 두 개의 개별 세트로 얼마나 잘 분리될 수 있는지를 측정하여 이 효과를 정량화하는 다른 메트릭을 제안한다.

> In order to la>bel the generated images, we train auxiliary classification networks for a number of binary attributes, e.g., to distinguish male and female faces. In our tests, the classifiers had the same architecture as the discriminator we use (i.e., same as in[<a href="#footnote_28_1" name="footnote_28_2">28</a>]), and were trained using the CelebA-HQ dataset that retains the 40 attributes available in the original CelebA dataset. To measure the separability of one attribute, we generate 200,000 images with $z\sim{P(z)}$ and classify them using the auxiliary classification network. We then sort the samples according to classifier confidence and remove the least confident half, yielding 100,000 labeled latent-space vectors
>> 생성된 이미지에 레이블을 붙이기 위해, 우리는 남성 얼굴과 여성 얼굴을 구별하기 위해 여러 이진 속성에 대한 보조 분류 네트워크를 훈련시킨다. 우리의 테스트에서 분류기는 우리가 사용하는 판별기와 동일한 아키텍처를 가지고 있었으며(즉, [<a href="#footnote_28_1" name="footnote_28_2">28</a>]과 동일), 원래 CellebA 데이터 세트에서 사용할 수 있는 40개의 속성을 유지하는 CellebA-HQ 데이터 세트를 사용하여 훈련되었다. 한 속성의 분리 가능성을 측정하기 위해 $z\sim{P(z)}$로 200,000개의 이미지를 생성하고 보조 분류 네트워크를 사용하여 분류한다. 그런 다음 분류기 신뢰도에 따라 샘플을 정렬하고 신뢰도가 가장 낮은 절반을 제거하여 100,000개의 레이블이 지정된 잠재 공간 벡터를 산출한다.

> For each attribute, we fit a linear SVM to predict the label based on the latent-space point — $z$ for traditional and $w$ for style-based— and classify the points by this plane. We then compute the conditional entropy $H(Y\vert{}X)$ where $X$ are the classes predicted by the SVM and $y$ are the classes determined by the pre-trained classifier. This tells how much additional information is required to determine the true class of a sample, given that we know on which side of the hyperplane it lies. A low value suggests consistent latent space directions for the corresponding factor(s) of variation
>> 각 속성에 대해, 우리는 잠재 공간 포인트(기존의 경우 $z$, 스타일 기반의 경우 $w$)를 기반으로 레이블을 예측하고 이 평면을 기준으로 포인트를 분류하기 위해 선형 SVM을 적합시킨다. 그런 다음 조건부 엔트로피 $H(Y\vert{}X)$를 계산한다. 여기서 $X$는 SVM에 의해 예측된 클래스이고 $y$는 사전 훈련된 분류기에 의해 결정된 클래스이다. 초평면의 어느 쪽에 있는지 알 수 있기 때문에 샘플의 실제 클래스를 결정하는 데 얼마나 많은 추가 정보가 필요한지 알 수 있습니다. 낮은 값은 해당 변동 요인에 대해 일관된 잠재 공간 방향을 나타냅니다.

> We calculate the final separability score as $\exp(\sum_{i}H(Y_{i}|X_{i}))$, where $i$ enumerates the 40 attributes. Similar to the inception score[<a href="#footnote_49_1" name="footnote_49_2">49</a>], the exponentiation brings the values from logarithmic to linear domain so that they are easier to compare
>> $i$가 40개의 속성을 열거하는 $\exp(\sum_{i}H(Y_{i}\vert{}X_{i})$로 최종 분리 가능성 점수를 계산한다. 초기 점수[<a href="#footnote_49_1" name="footnote_49_2">49</a>]와 유사하게, 지수화는 값을 로그 도메인에서 선형 도메인으로 가져와 비교하기 쉽게 한다.

> Tables 3 and 4 show that $w$ is consistently better separable than $Z$, suggesting a less entangled representation Furthermore, increasing the depth of the mapping network improves both image quality and separability in $W$, which is in line with the hypothesis that the synthesis network inherently favors a disentangled input representation. Interestingly, adding a mapping network in front of a traditional generator results in severe loss of separability in $z$ but improves the situation in the intermediate latent space $W$, and the FID improves as well. This shows that even the traditional generator architecture performs better when we introduce an intermediate latent space that does not have to follow the distribution of the training data.
>> 표 3과 4는 $w$가 $Z$보다 일관되게 분리할 수 있다는 것을 보여주며, 덜 얽힌 표현을 제안한다. 더욱이, 매핑 네트워크의 깊이를 늘리면 $W$의 이미지 품질과 분리성이 향상된다. 이는 합성 네트워크가 본질적으로 분리된 입력 표현을 선호한다는 가설과 일치한다. 흥미롭게도, 전통적인 발전기 앞에 매핑 네트워크를 추가하면 $z$의 분리성이 심각하게 손실되지만 중간 잠재 공간 $W$의 상황이 개선되고 FID도 개선된다. 이는 전통적인 발전기 아키텍처조차도 훈련 데이터의 분포를 따를 필요가 없는 중간 잠재 공간을 도입할 때 더 나은 성능을 발휘한다는 것을 보여준다.

![Figure 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Figure-7.JPG)

> Figure 7. The FFHQ dataset offers a lot of variety in terms of age, ethnicity, viewpoint, lighting, and image background.
>> 그림 7. FFHQ 데이터 세트는 연령, 민족성, 관점, 조명 및 이미지 배경 측면에서 많은 다양성을 제공한다.

![Figure 8](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-23-(GAN)Style-GAN/Figure-8.JPG)

> Figure 8. The effect of truncation trick as a function of style scale $\psi$. When we fade $\psi\to{0}$, all faces converge to the “mean” face of FFHQ. This face is similar for all trained networks, and the interpolation towards it never seems to cause artifacts. By applying negative scaling to styles, we get the corresponding opposite or “anti-face”. It is interesting that various high-level attributes often flip between the opposites, including viewpoint, glasses, age, coloring, hair length, and often gender.
>> 그림 8. 스타일 스케일 $\psi$의 함수로서 잘림 트릭의 효과. $\psi\to{0}$를 페이드하면 모든 면은 FFHQ의 "평균" 면으로 수렴된다. 이 얼굴은 훈련된 모든 네트워크에서 유사하며, 그에 대한 보간은 결코 아티팩트를 유발하지 않는 것으로 보인다. 스타일에 네거티브 스케일링을 적용하면 그에 상응하는 반대 또는 "안티 페이스"를 얻을 수 있습니다. 관점, 안경, 나이, 색칠, 머리 길이, 그리고 종종 성별을 포함한 다양한 높은 수준의 속성들이 서로 엇갈리는 것이 흥미롭다.

### $\mathbf{5.\;Conclusion}$

> Based on both our results and parallel work by Chen et l.[<a href="#footnote_5_1" name="footnote_5_2">5</a>], it is becoming clear that the traditional GAN generator architecture is in every way inferior to a style-based design. This is true in terms of established quality metrics, and we further believe that our investigations to the separation of high-level attributes and stochastic effects, as well as the linearity of the intermediate latent space will prove fruitful in improving the understanding and controllability of GAN synthesis.
>> 우리의 결과와 Chen et.[<a href="#footnote_5_1" name="footnote_5_2">5</a>]의 병렬 작업을 바탕으로, 기존의 GAN 생성기 아키텍처가 스타일 기반 설계보다 모든 면에서 열등하다는 것이 분명해지고 있다. 이는 확립된 품질 지표 측면에서 사실이며, 중간 잠재 공간의 선형성뿐만 아니라 높은 수준의 속성과 확률적 효과의 분리에 대한 우리의 조사가 GAN 합성의 이해와 제어 가능성을 향상시키는 데 도움이 될 것으로 믿는다.

> We note that our average path length metric could easily be used as a regularizer during training, and perhaps some variant of the linear separability metric could act as one, too. In general, we expect that methods for directly shaping the intermediate latent space during training will provide interesting avenues for future work.
>> 우리의 평균 경로 길이 메트릭은 훈련 중에 정규화기로 쉽게 사용될 수 있으며, 선형 분리 가능성 메트릭의 일부 변형도 하나로 작용할 수 있다. 일반적으로, 우리는 훈련 중에 중간 잠재 공간을 직접 형성하는 방법이 향후 작업에 흥미로운 방법을 제공할 것으로 기대한다.

### $\mathbf{6.\;Acknowledgements}$

> We thank Jaakko Lehtinen, David Luebke, and Tuomas Kynkäänniemi for in-depth discussions and helpful comments; Janne Hellsten, Tero Kuosmanen, and Pekka Jänis for compute infrastructure and help with the code release.
>> Jaakko Lhtinen, David Luebke, Tuomas Kynkäniemi는 심도 있는 토론과 유용한 의견을, Janne Hellsten, Tero Kuosmanen 및 Pecka Jänis는 컴퓨팅 인프라와 코드 릴리스에 도움을 주었다.

### $\mathbf{A.\;The\;FFHQ\;dataset}$

> We have collected a new dataset of human faces, FlickrFaces-HQ (FFHQ), consisting of 70,000 high-quality images at $1024^{2}$ resolution (Figure 7). The dataset includes vastly more variation than CelebA-HQ[<a href="#footnote_28_1" name="footnote_28_2">28</a>] in terms of age, ethnicity and image background, and also has much better coverage of accessories such as eyeglasses, sunglasses, hats, etc. The images were crawled from Flickr (thus inheriting all the biases of that website) and automatically aligned[<a href="#footnote_29_1" name="footnote_29_2">29</a>] and cropped. Only images under permissive licenses were collected. Various automatic filters were used to prune the set, and finally Mechanical Turk allowed us to remove the occasional statues, paintings, or photos of photos. We have made the dataset publicly available at [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)
>> $1024^{2}$ 해상도의 70,000개의 고품질 이미지로 구성된 새로운 인간 얼굴 데이터 세트인 FlickrFaces-HQ(FFHQ)를 수집했다(그림 7). 이 데이터 세트는 나이, 민족성 및 이미지 배경 측면에서 CelevA-HQ[<a href="#footnote_28_1" name="footnote_28_2">28</a>]보다 훨씬 더 많은 변형을 포함하고 있으며 안경, 선글라스, 모자 등과 같은 액세서리에 대한 커버리지도 훨씬 좋다. 이미지는 플리커(Flickr)에서 크롤링(따라서 해당 웹 사이트의 모든 편견을 상속)되었으며 [<a href="#footnote_29_1" name="footnote_29_2">29</a>] 자동으로 정렬되고 잘렸습니다. 허용 라이센스에 따른 이미지만 수집되었습니다. 세트를 가지치기 위해 다양한 자동 필터가 사용되었고, 마침내 메카니컬 터크는 우리가 가끔 조각, 그림 또는 사진의 사진을 제거할 수 있게 해주었다. 우리는 데이터 세트를 [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)에서 공개적으로 사용할 수 있도록 했다.

### $\mathbf{B.\;Truncation\;trick\;in}\;W$

> If we consider the distribution of training data, it is clear that areas of low density are poorly represented and thus likely to be difficult for the generator to learn. This is a significant open problem in all generative modeling techniques. However, it is known that drawing latent vectors from a truncated[<a href="#footnote_38_1" name="footnote_38_2">38</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>] or otherwise shrunk[<a href="#footnote_31_1" name="footnote_31_2">31</a>] sampling space tends to improve average image quality, although some amount of variation is lost.
>> 훈련 데이터의 분포를 고려할 때, 낮은 밀도의 영역은 잘 표현되지 않아 발전기가 학습하기 어려울 가능성이 높다는 것은 분명하다. 이것은 모든 생성 모델링 기법에서 중요한 미해결 문제이다. 그러나 잘린 [<a href="#footnote_38_1" name="footnote_38_2">38</a>, <a href="#footnote_4_1" name="footnote_4_2">4</a>] 또는 축소된 [<a href="#footnote_31_1" name="footnote_31_2">31</a>] 샘플링 공간에서 잠재 벡터를 그리는 것은 일정량의 변동은 손실되지만 평균 이미지 품질을 향상시키는 경향이 있는 것으로 알려져 있다.

> We can follow a similar strategy. To begin, we compute the center of mass of $W$ as $\bar{w}=E_{z}\sim{P(z)}[f(z)]$. In case of FFHQ this point represents a sort of an average face (Figure 8, $\psi{}=0$). We can then scale the deviation of a given $w$ from the center as $w'=\bar{w}+\psi{}(w-\bar{w})$, where $\psi<1$. While Brock et al.[<a href="#footnote_4_1" name="footnote_4_2">4</a>] observe that only a subset of networks is amenable to such truncation even when orthogonal regularization is used, truncation in $W$ space seems to work reliably even without changes to the loss function.
>> 우리는 비슷한 전략을 따를 수 있다. 먼저 $W$의 질량 중심을 $\bar{w}=E_{z}\sim{P(z)}[f(z)]$로 계산한다. FFHQ의 경우 이 점은 일종의 평균 면을 나타냅니다(그림 8, $\psi{}=0$). 그런 다음 중심에서 주어진 $w$의 편차를 $w'=\bar{w}+\psi{}(w-\bar{w})$로 스케일링할 수 있으며, 여기서 $\psi<1$이다. 브록 외[<a href="#footnote_4_1" name="footnote_4_2">4</a>] 직교 정규화가 사용되는 경우에도 네트워크의 일부만 이러한 절단을 수정할 수 있다는 것을 관찰한다. $W$ 공간의 절단은 손실 함수에 대한 변경 없이도 안정적으로 작동하는 것으로 보인다.

---

#### $\mathbf{References}$

<a href="#footnote_1_2" name="footnote_1_1">[1]</a> [A. Achille and S. Soatto. On the emergence of invariance and disentangling in deep representations. CoRR, abs/1706.01350, 2017]. 6

<a href="#footnote_2_2" name="footnote_2_1">[2]</a> [D. Bau, J. Zhu, H. Strobelt, B. Zhou, J. B. Tenenbaum, W. T. Freeman, and A. Torralba. GAN dissection: Visualizing and understanding generative adversarial networks. In Proc. ICLR, 2019]. 1

<a href="#footnote_3_2" name="footnote_3_1">[3]</a>  M. Ben-Yosef and D. Weinshall. Gaussian mixture generative adversarial networks for diverse datasets, and the unsupervised clustering of images. CoRR, abs/1808.10356, 2018. 3

<a href="#footnote_4_2" name="footnote_4_1">[4]</a> A. Brock, J. Donahue, and K. Simonyan. Large scale GAN training for high fidelity natural image synthesis. CoRR, abs/1809.11096, 2018. 1, 3, 8

<a href="#footnote_5_2" name="footnote_5_1">[5]</a> T. Chen, M. Lucic, N. Houlsby, and S. Gelly. On self modulation for generative adversarial networks. CoRR, abs/1810.01365, 2018. 3, 8

<a href="#footnote_6_2" name="footnote_6_1">[6]</a> T. Q. Chen, X. Li, R. B. Grosse, and D. K. Duvenaud. Isolating sources of disentanglement in variational autoencoders. CoRR, abs/1802.04942, 2018. 6

<a href="#footnote_7_2" name="footnote_7_1">[7]</a> X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel. InfoGAN: interpretable representation learning by information maximizing generative adversarial nets. CoRR, abs/1606.03657, 2016. 6

<a href="#footnote_8_2" name="footnote_8_1">[8]</a> E. L. Denton, S. Chintala, A. Szlam, and R. Fergus. Deep generative image models using a Laplacian pyramid of adversarial networks. CoRR, abs/1506.05751, 2015. 3

<a href="#footnote_9_2" name="footnote_9_1">[9]</a> G. Desjardins, A. Courville, and Y. Bengio. Disentangling factors of variation via generative entangling. CoRR, abs/1210.5474, 2012. 6

<a href="#footnote_10_2" name="footnote_10_1">[10]</a> T. Doan, J. Monteiro, I. Albuquerque, B. Mazoure, A. Durand, J. Pineau, and R. D. Hjelm. Online adaptative curriculum learning for GANs. CoRR, abs/1808.00020, 2018. 3

<a href="#footnote_11_2" name="footnote_11_1">[11]</a> J. Donahue, P. Krähenbühl, and T. Darrell. Adversarial feature learning. CoRR, abs/1605.09782, 2016. 6

<a href="#footnote_12_2" name="footnote_12_1">[12]</a> A. Dosovitskiy, J. T. Springenberg, and T. Brox. Learning to generate chairs with convolutional neural networks. CoRR, abs/1411.5928, 2014. 1

<a href="#footnote_13_2" name="footnote_13_1">[13]</a> H. Drucker and Y. L. Cun. Improving generalization performance using double backpropagation. IEEE Transactions on Neural Networks, 3(6):991–997, 1992. 3

<a href="#footnote_14_2" name="footnote_14_1">[14]</a> V. Dumoulin, I. Belghazi, B. Poole, A. Lamb, M. Arjovsky, O. Mastropietro, and A. Courville. Adversarially learned inference. In Proc. ICLR, 2017. 6

<a href="#footnote_15_2" name="footnote_15_1">[15]</a> V. Dumoulin, E. Perez, N. Schucher, F. Strub, H. d. Vries, A. Courville, and Y. Bengio. Feature-wise transformations. Distill, 2018. https://distill.pub/2018/feature-wisetransformations. 2

<a href="#footnote_16_2" name="footnote_16_1">[16]</a> V. Dumoulin, J. Shlens, and M. Kudlur. A learned representation for artistic style. CoRR, abs/1610.07629, 2016. 2

<a href="#footnote_17_2" name="footnote_17_1">[17]</a> I. P. Durugkar, I. Gemp, and S. Mahadevan. Generative multiadversarial networks. CoRR, abs/1611.01673, 2016. 3

<a href="#footnote_18_2" name="footnote_18_1">[18]</a> C. Eastwood and C. K. I. Williams. A framework for the quantitative evaluation of disentangled representations. In Proc. ICLR, 2018. 6

<a href="#footnote_19_2" name="footnote_19_1">[19]</a> L. A. Gaty_{s}, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. In Proc. CVPR, 2016. 6

<a href="#footnote_20_2" name="footnote_20_1">[20]</a> G. Ghiasi, H. Lee, M. Kudlur, V. Dumoulin, and J. Shlens. Exploring the structure of a real-time, arbitrary neural artistic stylization network. CoRR, abs/1705.06830, 2017. 2

<a href="#footnote_21_2" name="footnote_21_1">[21]</a>  I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. WardeFarley, S. Ozair, A. Courville, and Y. Bengio. Generative Adversarial Networks. In NIPS, 2014. 1, 3

<a href="#footnote_22_2" name="footnote_22_1">[22]</a> W.-S. Z. Guang-Yuan Hao, Hong-Xing Yu. MIXGAN: learning concepts from different domains for mixture generation. CoRR, abs/1807.01659, 2018. 2

<a href="#footnote_23_2" name="footnote_23_1">[23]</a> I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville. Improved training of Wasserstein GANs. CoRR, abs/1704.00028, 2017. 1, 2

<a href="#footnote_24_2" name="footnote_24_1">[24]</a>M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Proc. NIPS, pages 6626–6637, 2017.  2

<a href="#footnote_25_2" name="footnote_25_1">[25]</a> I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In Proc. ICLR, 2017. 6

<a href="#footnote_26_2" name="footnote_26_1">[26]</a> X. Huang and S. J. Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. CoRR, abs/1703.06868, 2017. 1, 2

<a href="#footnote_27_2" name="footnote_27_1">[27]</a> X. Huang, M. Liu, S. J. Belongie, and J. Kautz. Multimodal unsupervised image-to-image translation. CoRR, abs/1804.04732, 2018. 2

<a href="#footnote_28_2" name="footnote_28_1">[28]</a> T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. CoRR, abs/1710.10196, 2017. 1, 2, 7 ,8

<a href="#footnote_29_2" name="footnote_29_1">[29]</a> V. Kazemi and J. Sullivan. One millisecond face alignment with an ensemble of regression trees. In Proc. CVPR, 2014. 8

<a href="#footnote_30_2" name="footnote_30_1">[30]</a> H. Kim and A. Mnih. Disentangling by factorising. In Proc. ICML, 2018. 6

<a href="#footnote_31_2" name="footnote_31_1">[31]</a> D. P. Kingma and P. Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. CoRR, abs/1807.03039, 2018. 3, 8

<a href="#footnote_32_2" name="footnote_32_1">[32]</a> D. P. Kingma and M. Welling. Auto-encoding variational bayes. In ICLR, 2014. 6

<a href="#footnote_33_2" name="footnote_33_1">[33]</a> K. Kurach, M. Lucic, X. Zhai, M. Michalski, and S. Gelly. The gan landscape: Losses, architectures, regularization, and normalization. CoRR, abs/1807.04720, 2018. 1

<a href="#footnote_34_2" name="footnote_34_1">[34]</a> S. Laine. Feature-based metrics for exploring the latent space of generative models. ICLR workshop poster, 2018. 1, 6

<a href="#footnote_35_2" name="footnote_35_1">[35]</a> Y. Li, C. Fang, J. Yang, Z. Wang, X. Lu, and M.-H. Yang. Universal style transfer via feature transforms. In Proc. NIPS, 2017 2

<a href="#footnote_36_2" name="footnote_36_1">[36]</a> Y. Li, N. Wang, J. Liu, and X. Hou. Demy_{s}tifying neural style transfer. CoRR, abs/1701.01036, 2017. 6

<a href="#footnote_37_2" name="footnote_37_1">[37]</a> M. Lucic, K. Kurach, M. Michalski, S. Gelly, and O. Bousquet. Are GANs created equal? a large-scale study. CoRR, abs/1711.10337, 2017. 1

<a href="#footnote_38_2" name="footnote_38_1">[38]</a> M. Marchesi. Megapixel size image creation using generative adversarial networks. CoRR, abs/1706.00082, 2017. 3, 8

<a href="#footnote_39_2" name="footnote_39_1">[39]</a> L. Matthey, I. Higgins, D. Hassabis, and A. Lerchner. dsprites: Disentanglement testing sprites dataset. https://github.com/deepmind/dsprites-dataset/, 2017. 6

<a href="#footnote_40_2" name="footnote_40_1">[40]</a> L. Mescheder, A. Geiger, and S. Nowozin. Which training methods for GANs do actually converge? CoRR, abs/1801.04406, 2018. 1, 3

<a href="#footnote_41_2" name="footnote_41_1">[41]</a> T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. CoRR, abs/1802.05957, 2018. 1

<a href="#footnote_42_2" name="footnote_42_1">[42]</a> T. Miyato and M. Koyama. cGANs with projection discriminator. CoRR, abs/1802.05637, 2018. 3

<a href="#footnote_43_2" name="footnote_43_1">[43]</a> G. Mordido, H. Yang, and C. Meinel. Dropout-gan: Learning from a dynamic ensemble of discriminators. CoRR, abs/1807.11346, 2018. 3

<a href="#footnote_44_2" name="footnote_44_1">[44]</a> S. Mukherjee, H. Asnani, E. Lin, and S. Kannan. ClusterGAN : Latent space clustering in generative adversarial networks. CoRR, abs/1809.03627, 2018. 3

<a href="#footnote_45_2" name="footnote_45_1">[45]</a> D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proc. ICML, 2014. 6

<a href="#footnote_46_2" name="footnote_46_1">[46]</a> K. Ridgeway. A survey of inductive biases for factorial representation-learning. CoRR, abs/1612.05299, 2016. 6

<a href="#footnote_47_2" name="footnote_47_1">[47]</a> A. S. Ross and F. Doshi-Velez. Improving the adversarial robustness and interpretability of deep neural networks by regularizing their input gradients. CoRR, abs/1711.09404, 2017. 3

<a href="#footnote_48_2" name="footnote_48_1">[48]</a> T. Sainburg, M. Thielk, B. Theilman, B. Migliori, and T. Gentner. Generative adversarial interpolative autoencoding: adversarial training on latent space interpolations encourage convex latent distributions. CoRR, abs/1807.06650, 2018. 1, 3

<a href="#footnote_49_2" name="footnote_49_1">[49]</a> T. Salimans, I. J. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training GANs. In NIPS, 2016. 7

<a href="#footnote_50_2" name="footnote_50_1">[50]</a> J. Schmidhuber. Learning factorial codes by predictability minimization. Neural Computation, 4(6):863–879, 1992. 6

<a href="#footnote_51_2" name="footnote_51_1">[51]</a> R. Sharma, S. Barratt, S. Ermon, and V. Pande. Improved training with curriculum gans. CoRR, abs/1807.09295, 2018. 3

<a href="#footnote_52_2" name="footnote_52_1">[52]</a> K. Shoemake. Animating rotation with quaternion curves. In Proc. SIGGRAPH ’85, 1985. 7

<a href="#footnote_53_2" name="footnote_53_1">[53]</a> A. Siarohin, E. Sangineto, and N. Sebe. Whitening and coloring transform for GANs. CoRR, abs/1806.00420, 2018. 2

<a href="#footnote_54_2" name="footnote_54_1">[54]</a> K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014. 6

<a href="#footnote_55_2" name="footnote_55_1">[55]</a> T. Wang, M. Liu, J. Zhu, A. Tao, J. Kautz, and B. Catanzaro. High-resolution image synthesis and semantic manipulation with conditional GANs. CoRR, abs/1711.11585, 2017. 3

<a href="#footnote_56_2" name="footnote_56_1">[56]</a> T. White. Sampling generative networks: Notes on a few effective techniques. CoRR, abs/1609.04468, 2016. 7

<a href="#footnote_57_2" name="footnote_57_1">[57]</a> H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena. Self-attention generative adversarial networks. CoRR, abs/1805.08318, 2018. 3

<a href="#footnote_58_2" name="footnote_58_1">[58]</a> R. Zhang. Making convolutional networks shift-invariant again, 2019. 2

<a href="#footnote_59_2" name="footnote_59_1">[59]</a> R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proc. CVPR, 2018. 6, 7
