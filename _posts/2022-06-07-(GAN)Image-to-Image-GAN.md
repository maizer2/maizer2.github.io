---
layout: post 
title: "(GAN)Image-to-Image Translation with Conditional Adversarial Networks Translation"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review, 1.2.2.5. GAN]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

### $$\mathbf{Image-to-Image\;Translation\;with\;Conditional\;Adversarial\;Networks}$$

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-1.JPG)

> Figure 1: Many problems in image processing, graphics, and vision involve translating an input image into a corresponding output image. These problems are often treated with application-specific algorithms, even though the setting is always the same: map pixels to pixels. Conditional adversarial nets are a general-purpose solution that appears to work well on a wide variety of these problems. Here we show results of the method on several. In each case we use the same architecture and objective, and simply train on different data.
>> 그림 1: 이미지 처리, 그래픽 및 비전의 많은 문제는 입력 이미지를 해당 출력 이미지로 변환하는 것입니다. 이러한 문제는 픽셀을 픽셀로 매핑하는 설정이 항상 동일함에도 불구하고 종종 애플리케이션별 알고리즘으로 처리된다. 조건부 적대적 네트워크는 이러한 다양한 문제에서 잘 작동하는 것으로 보이는 범용 솔루션이다. 여기서는 몇 가지 방법에 대한 결과를 보여 준다. 각각의 경우에 우리는 동일한 아키텍처와 목표를 사용하며, 단순히 다른 데이터에 대해 훈련한다.

#### $$\mathbf{Abstract}$$

> We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Moreover, since the release of the pix2pix software associated with this paper, hundreds of twitter users have posted their own artistic experiments using our system. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without handengineering our loss functions either. 
>> 우리는 이미지 간 변환 문제에 대한 범용 솔루션으로 조건부 적대적 네트워크를 조사한다. 이러한 네트워크는 입력 이미지에서 출력 이미지로의 매핑을 학습할 뿐만 아니라 이 매핑을 훈련시키기 위한 손실 함수를 학습한다. 이를 통해 전통적으로 매우 다른 손실 공식을 필요로 하는 문제에 동일한 일반적인 접근법을 적용할 수 있다. 우리는 이 접근 방식이 레이블 맵의 사진을 합성하고, 에지 맵의 객체를 재구성하고, 이미지를 색칠하는 데 효과적이라는 것을 보여준다. 또한 본 논문과 관련된 pix2pix 소프트웨어가 출시된 이후 수백 명의 트위터 사용자가 우리 시스템을 사용하여 자신만의 예술적 실험을 게시했다. 공동체로서 우리는 더 이상 매핑 기능을 수작업으로 설계하지 않으며, 이 연구는 손실 함수를 수작업으로 설계하지 않고도 합리적인 결과를 얻을 수 있음을 시사한다.

### $1\;\mathbf{Introduction}$

> Many problems in image processing, computer graphics, and computer vision can be posed as “translating” an input image into a corresponding output image. Just as a concept may be expressed in either English or French, a scene may be rendered as an RGB image, a gradient field, an edge map, a semantic label map, etc. In analogy to automatic language translation, we define automatic image-to-image translation as the problem of translating one possible representation of a scene into another, given sufficient training data (see Figure 1). Traditionally, each of these tasks has been tackled with separate, special-purpose machinery (e.g., [14, 23, 18, 8, 10, 50, 30, 36, 16, 55, 58]), despite the fact that the setting is always the same: predict pixels from pixels. Our goal in this paper is to develop a common framework for all these problems.
>> 이미지 처리, 컴퓨터 그래픽스, 컴퓨터 비전의 많은 문제는 입력 이미지를 해당 출력 이미지로 "변환"하는 것으로 상정될 수 있다. 개념이 영어나 프랑스어로 표현되는 것처럼 장면은 RGB 이미지, 그라데이션 필드, 에지 맵, 의미 라벨 맵 등으로 렌더링될 수 있다. 자동 언어 번역과 유사하게, 우리는 충분한 훈련 데이터가 주어지면 장면의 가능한 표현 중 하나를 다른 것으로 번역하는 문제로 자동 이미지 대 이미지 번역을 정의한다(그림 1 참조). 전통적으로 이러한 각 작업은 설정이 항상 동일함에도 불구하고 별도의 특수 목적 기계(예: [14, 23, 18, 8, 10, 50, 30, 36, 16, 55, 58])로 처리되었다. 이 논문에서 우리의 목표는 이러한 모든 문제에 대한 공통 프레임워크를 개발하는 것이다.

> The community has already taken significant steps in this direction, with convolutional neural nets (CNNs) becoming the common workhorse behind a wide variety of image prediction problems. CNNs learn to minimize a loss function – an objective that scores the quality of results – and although the learning process is automatic, a lot of manual effort still goes into designing effective losses. In other words, we still have to tell the CNN what we wish it to minimize. But, just like King Midas, we must be careful what we wish for! If we take a naive approach, and ask the CNN to minimize Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results [40, 58]. This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring. Coming up with loss functions that force the CNN to do what we really want – e.g., output sharp, realistic images – is an open problem and generally requires expert knowledge.
>> 컨볼루션 신경망(CNN)이 다양한 이미지 예측 문제 뒤에 있는 일반적인 업무 중심지가 되면서 커뮤니티는 이미 이러한 방향으로 중요한 단계를 밟았다. CNN은 결과의 품질을 점수화하는 목표인 손실 함수를 최소화하는 방법을 배우고, 학습 프로세스는 자동이지만 여전히 효과적인 손실을 설계하는 데 많은 수작업이 사용된다. 다시 말해, 우리는 여전히 CNN에 그것이 최소화되기를 바라는 것을 말해야 한다. 하지만, 마이다스 왕처럼, 우리는 우리가 바라는 것을 조심해야 합니다! 우리가 순진한 접근법을 취하고 CNN에 예측된 진실 픽셀과 지상 진실 픽셀 사이의 유클리드 거리를 최소화하도록 요청하면, 그것은 흐릿한 결과를 생성하는 경향이 있다[40, 58]. 이는 모든 그럴듯한 출력을 평균화하여 유클리드 거리가 최소화되어 흐릿함을 유발하기 때문이다. CNN이 우리가 진정으로 원하는 것(예: 선명하고 현실적인 이미지 출력)을 하도록 강제하는 손실 함수를 고안하는 것은 개방적인 문제이며 일반적으로 전문가 지식이 필요하다.

> It would be highly desirable if we could instead specify only a high-level goal, like “make the output indistinguishable from reality”, and then automatically learn a loss function appropriate for satisfying this goal. Fortunately, this is exactly what is done by the recently proposed Generative Adversarial Networks (GANs) [22, 12, 41, 49, 59]. GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss. Blurry images will not be tolerated since they look obviously fake. Because GANs learn a loss that adapts to the data, they can be applied to a multitude of tasks that traditionally would require very different kinds of loss functions.
>> 대신 "출력을 현실과 구분할 수 없게 만든다"와 같은 높은 수준의 목표만 명시하고, 이 목표를 만족시키는 데 적합한 손실 함수를 자동으로 학습할 수 있다면 매우 바람직할 것이다. 다행히도, 이것은 최근에 제안된 생성적 적대 네트워크(GANs)에 의해 정확히 수행된다[22, 12, 41, 49, 59]. GAN은 출력 이미지가 실제인지 가짜인지 분류하는 손실을 학습하는 동시에 이러한 손실을 최소화하기 위해 생성 모델을 훈련시킨다. 흐릿한 이미지는 명백히 가짜로 보이기 때문에 용납되지 않을 것이다. GAN은 데이터에 적응하는 손실을 학습하기 때문에 전통적으로 매우 다른 종류의 손실 함수를 필요로 하는 다양한 작업에 적용할 수 있다.

> In this paper, we explore GANs in the conditional setting. Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model [22]. This makes cGANs suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image. GANs have been vigorously studied in the last two years and many of the techniques we explore in this paper have been previously proposed. Nonetheless, earlier papers have focused on specific applications, and it has remained unclear how effective image-conditional
>> 본 논문에서는 조건부 설정에서 GAN을 살펴본다. GAN이 데이터의 생성 모델을 학습하는 것처럼 조건부 GAN(cGAN)은 조건부 생성 모델을 학습한다[22]. 이를 통해 cGAN은 이미지 간 변환 작업에 적합하며, 여기서 우리는 입력 이미지를 조건화하고 해당 출력 이미지를 생성한다. GAN은 지난 2년 동안 활발하게 연구되었으며 본 논문에서 탐구하는 많은 기술이 이전에 제안되었다. 그럼에도 불구하고, 이전의 논문은 특정 응용 프로그램에 초점을 맞추고 있으며, 이미지 조건부 기능이 얼마나 효과적인지는 여전히 불분명하다.

> GANs can be as a general-purpose solution for image-toimage translation. Our primary contribution is to demonstrate that on a wide variety of problems, conditional GANs produce reasonable results. Our second contribution is to present a simple framework sufficient to achieve good results, and to analyze the effects of several important architectural choices. Code is available at https://github.com/phillipi/pix2pix.
>> GAN은 이미지 간 변환을 위한 범용 솔루션일 수 있다. 우리의 주요 기여는 광범위한 문제에서 조건부 GAN이 합리적인 결과를 산출한다는 것을 입증하는 것이다. 우리의 두 번째 기여는 좋은 결과를 얻기에 충분한 간단한 프레임워크를 제시하고, 몇 가지 중요한 아키텍처 선택의 영향을 분석하는 것이다. 코드는 https://github.com/phillipi/pix2pix에서 이용할 수 있다.

### $2\;\mathbf{Related\;Work}$

> Structured losses for image modeling Image-to-image translation problems are often formulated as per-pixel classification or regression (e.g., [36, 55, 25, 32, 58]). These formulations treat the output space as “unstructured” in the sense that each output pixel is considered conditionally independent from all others given the input image. Conditional GANs instead learn a structured loss. Structured losses penalize the joint configuration of the output. A large body of literature has considered losses of this kind, with methods including conditional random fields [9], the SSIM metric [53], feature matching [13], nonparametric losses [34], the convolutional pseudo-prior [54], and losses based on matching covariance statistics [27]. The conditional GAN is different in that the loss is learned, and can, in theory, penalize any possible structure that differs between output and target.
>> 이미지 모델링을 위한 구조적 손실 이미지 대 이미지 변환 문제는 종종 픽셀 단위 분류 또는 회귀(예: [36, 55, 25, 32, 58])로 공식화된다. 이러한 공식은 각 출력 픽셀이 입력 이미지가 주어진 다른 모든 픽셀로부터 조건부로 독립적인 것으로 간주된다는 점에서 출력 공간을 "구조화되지 않은" 것으로 취급한다. 조건부 GAN은 대신 구조화된 손실을 학습한다. 구조적 손실은 출력의 공동 구성에 불이익을 준다. 대부분의 문헌은 조건부 무작위 필드 [9], SSIM 메트릭 [53], 특징 일치 [13], 비모수 손실 [34], 컨볼루션 유사 이전 [54] 및 일치하는 공분산 통계량에 기초한 손실을 포함한 방법으로 이러한 종류의 손실을 고려했다[27]. 조건부 GAN은 손실이 학습된다는 점에서 다르며, 이론적으로 출력과 목표 사이에 다른 가능한 구조에 불이익을 줄 수 있다.

> Conditional GANs We are not the first to apply GANs in the conditional setting. Prior and concurrent works have conditioned GANs on discrete labels [38, 21, 12], text [43], and, indeed, images. The image-conditional models have tackled image prediction from a normal map [52], future frame prediction [37], product photo generation [56], and image generation from sparse annotations [28, 45] (c.f. [44] for an autoregressive approach to the same problem). Several other papers have also used GANs for image-to-image mappings, but only applied the GAN unconditionally, relying on other terms (such as L2 regression) to force the output to be conditioned on the input. These papers have achieved impressive results on inpainting [40], future state prediction [60], image manipulation guided by user constraints [61], style transfer [35], and superresolution [33]. Each of the methods was tailored for a specific application. Our framework differs in that nothing is applicationspecific. This makes our setup considerably simpler than most others.
>> 조건부 GAN 조건부 설정에 GAN을 적용한 것은 우리가 처음이 아니다. 이전 및 동시 연구는 개별 레이블 [38, 21, 12], 텍스트 [43] 및 실제로 이미지에 GAN을 조건화했다. 이미지 조건부 모델은 일반 지도에서 이미지 예측[52], 미래 프레임 예측[37], 제품 사진 생성[56] 및 희소 주석[28,45]에서 이미지 생성을 다루었다. (같은 문제에 대한 자동 회귀 접근에 대한 c.f. [44]). 다른 여러 논문도 이미지 대 이미지 매핑에 GAN을 사용했지만, 다른 용어(예: L2 회귀 분석)에 의존하여 입력에 대해 출력을 강제로 조건화하도록 GAN만 무조건 적용했다. 이러한 논문은 인페인팅 [40], 미래 상태 예측 [60], 사용자 제약 조건에 따른 이미지 조작 [61], 스타일 전송 [35] 및 초해상도 [33]에서 인상적인 결과를 달성했다. 각 방법은 특정 용도에 맞게 조정되었습니다. 우리의 프레임워크는 어떤 것도 특정 애플리케이션에 국한되지 않는다는 점에서 다르다. 이것은 대부분의 다른 것들보다 우리의 설정을 상당히 단순하게 만든다.

> Our method also differs from the prior works in several architectural choices for the generator and discriminator. Unlike past work, for our generator we use a “U-Net”-based architecture [47], and for our discriminator we use a convolutional “PatchGAN” classifier, which only penalizes structure at the scale of image patches. A similar PatchGAN architecture was previously proposed in [35], for the purpose of capturing local style statistics. Here we show that this approach is effective on a wider range of problems, and we investigate the effect of changing the patch size.
>> 우리의 방법은 또한 발전기 및 판별기에 대한 몇 가지 아키텍처 선택에서 이전 작업과 다르다. 과거 작업과 달리, 발전기의 경우 "U-Net" 기반 아키텍처[47]를 사용하고 판별기의 경우 이미지 패치 규모에서만 구조를 불이익하는 컨볼루션 "PatchGAN" 분류기를 사용한다. 유사한 PatchGAN 아키텍처는 로컬 스타일 통계를 캡처하기 위해 [35]에서 이전에 제안되었다. 여기서는 이 접근 방식이 더 광범위한 문제에 효과적이라는 것을 보여주고 패치 크기 변경의 효과를 조사한다.

### $\mathbf{3\;Method}$

GANs are generative models that learn a mapping from random noise vector $z$ to output image $y$, $G : z \to y$ [22]. In contrast, conditional GANs learn a mapping from observed image $x$ and random noise vector $z$, to $y, G : (x, z) → y$. The generator G is trained to produce outputs that cannot be distinguished from “real” images by an adversarially  trained discriminator, $D$, which is trained to do as well as possible at detecting the generator’s “fakes”. This training procedure is diagrammed in Figure 2.
>> GAN은 랜덤 노이즈 벡터 $z$에서 출력 이미지 $y$, $G : z \to y$로의 매핑을 학습하는 생성 모델이다. 대조적으로, 조건부 GAN은 관찰된 이미지 $x$와 무작위 노이즈 벡터 $z$에서 $y, G : (x, z) → y$로의 매핑을 학습한다. 생성기 G는 적대적으로 훈련된 판별기 $D$에 의해 "실제" 이미지와 구별할 수 없는 출력을 생성하도록 훈련되며, 이는 생성기의 "가짜"를 탐지하는 데 최대한 잘 수행하도록 훈련된다. 이 교육 절차는 그림 2에 도해되어 있습니다.

#### $\mathbf{3.1\;Objective}$

> The objective of a conditional GAN can be expressed as
>> 조건부 GAN의 목표는 다음과 같이 표현할 수 있다.

$$L_{cGAN}(G,D)=E_{x,y}[\log{D(x,y)}]+$$
$$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;E_{x,z}[\log{1-D(x,G(x,z))}],$$

> where $G$ tries to minimize this objective against an adversarial $D$ that tries to maximize it, i.e. $G^{*}=arg min_{G}max_{D}L_{cGAN}(G, D)$.
>> 여기서 $G$는 최대화를 시도하는 적대적 $D$, 즉 $G^{*}=arg min_{G}max_{D}L_{cGAN}(G, D)$에 대해 이 목표를 최소화하려고 한다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-2.JPG)

> Figure 2: Training a conditional GAN to map edges→photo. The discriminator, $D$, learns to classify between fake  (synthesized by the generator) and real {edge, photo} tuples. The generator, $G$, learns to fool the discriminator. Unlike an unconditional GAN, both the generator and discriminator observe the input edge map.
>> 그림 2: 엣지→ 사진을 매핑하기 위한 조건부 GAN 훈련 판별기 $D$는 가짜(발전기에 의해 합성됨)와 실제 {edge, photo} 튜플 사이에서 분류하는 방법을 학습한다. 생성자 $G$는 판별자를 속이는 법을 배운다. 무조건적인 GAN과 달리, 생성기와 판별기 모두 입력 에지 맵을 관찰한다.

> To test the importance of conditioning the discriminator, we also compare to an unconditional variant in which the discriminator does not observe $x$:
>> 판별기 조건화의 중요성을 테스트하기 위해 판별기가 $x$를 관찰하지 않는 무조건 변종과도 비교한다.

$$L_{GAN}(G,D)=E_{y}[\log{D(y)}+$$
$$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;E_{x,y}[\log{(1-D(G(x,y))}].$$

> Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance [40]. The discriminator’s job remains unchanged, but the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense. We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:
>> 이전 접근 방식은 GAN 목표를 L2 거리[40]와 같은 보다 전통적인 손실과 혼합하는 것이 유리하다는 것을 발견했다. 판별자의 작업은 변경되지 않았지만, 생성자는 판별자를 속일 뿐만 아니라 L2 의미에서 실측 자료 출력 근처에 있어야 하는 과제를 안고 있다. 또한 L1이 흐릿함을 줄이기 위해 L2가 아닌 L1 거리를 사용하여 이 옵션을 살펴본다.

$$L_{L1}(G)=E_{x,y,z}[\parallel{y-G(x-z)\parallel{_{1}}}].$$

> Our final objective is
>> 우리의 최종 목표는

$$G^{*}=\arg{\underset{G}{\min}\;\underset{D}{\max}L_{cGAN}(G,D)+\lambda{L_{L1}}(G)}.$$

> Without $z$, the net could still learn a mapping from $x$ to $y$, but would produce deterministic outputs, and therefore fail to match any distribution other than a delta function. Past conditional GANs have acknowledged this and provided Gaussian noise $z$ as an input to the generator, in addition to $x$ (e.g., [52]). In initial experiments, we did not find this strategy effective – the generator simply learned to ignore the noise – which is consistent with Mathieu et al. [37]. Instead, for our final models, we provide noise only in the form of dropout, applied on several layers of our generator at both training and test time. Despite the dropout noise, we observe only minor stochasticity in the output of our nets. Designing conditional GANs that produce highly stochastic output, and thereby capture the full entropy of the conditional distributions they model, is an important question left open by the present work.
>> $z$가 없으면 네트워크는 여전히 $x$에서 $y$로의 매핑을 학습할 수 있지만 결정론적 출력을 생성하므로 델타 함수 이외의 분포와 일치하지 않는다. 과거의 조건부 GAN은 이를 인정하고 $x$ 외에 가우스 노이즈 $z$를 생성기에 대한 입력으로 제공했다(예: [52]). 초기 실험에서, 우리는 이 전략이 효과적이라는 것을 발견하지 못했다. 즉, 발전기는 단순히 소음을 무시하는 법을 배웠을 뿐이며, 이는 Mathieu 등과 일치한다. [37. 대신, 우리의 최종 모델의 경우, 우리는 훈련과 시험 시간 모두에서 발전기의 여러 계층에 적용되는 드롭아웃의 형태로만 노이즈를 제공한다. 드롭아웃 노이즈에도 불구하고, 우리는 네트의 출력에서 사소한 확률만을 관찰한다. 높은 확률적 출력을 생성하는 조건부 GAN을 설계하여 모델링하는 조건부 분포의 전체 엔트로피를 포착하는 것은 현재 연구에서 열려 있는 중요한 문제이다.

#### $\mathbf{3.2\; Network\;architectures}$

> We adapt our generator and discriminator architectures from those in [41]. Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu [26]. Details of the architecture are provided in the supplemental materials online, with key features discussed below.
>> 우리는 [41]의 것에서 발전기 및 판별기 아키텍처를 조정한다. 발전기와 판별기 모두 컨볼루션-BatchNorm-ReLu 형식의 모듈을 사용한다[26]. 아키텍처에 대한 자세한 내용은 아래에 설명된 주요 기능과 함께 온라인 보충 자료에 나와 있습니다.

##### $\mathbf{3.2.1\;Generator\;with\;skips}$

> A defining feature of image-to-image translation problems is that they map a high resolution input grid to a high resolution output grid. In addition, for the problems we consider, the input and output differ in surface appearance, but both are renderings of the same underlying structure. Therefore, structure in the input is roughly aligned with structure in the output. We design the generator architecture around these considerations.
>> 이미지 간 변환 문제의 정의적 특징은 고해상도 입력 그리드를 고해상도 출력 그리드에 매핑한다는 것이다. 또한, 우리가 고려하는 문제의 경우, 입력과 출력은 표면 모양이 다르지만 둘 다 동일한 기본 구조의 렌더링이다. 따라서 입력의 구조는 출력의 구조와 대략적으로 정렬됩니다. 우리는 이러한 고려 사항을 중심으로 발전기 아키텍처를 설계한다.

> Many previous solutions [40, 52, 27, 60, 56] to problems in this area have used an encoder-decoder network [24]. In such a network, the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed. Such a network requires that all information flow pass through all the layers, including the bottleneck. For many image translation problems, there is a great deal of low-level information shared between the input and output, and it would be desirable to shuttle this information directly across the net. For example, in the case of image colorizaton, the input and output share the location of prominent edges.
>> 이 영역의 문제에 대한 많은 이전 솔루션[40, 52, 27, 60, 56]은 인코더-디코더 네트워크를 사용했다[24]. 그러한 네트워크에서, 입력은 병목 계층까지 점진적으로 다운샘플을 하는 일련의 계층을 통과하며, 이때 프로세스가 역전된다. 이러한 네트워크는 모든 정보 흐름이 병목 현상을 포함한 모든 계층을 통과해야 한다. 많은 이미지 번역 문제의 경우 입력과 출력 간에 공유되는 낮은 수준의 정보가 많으며, 이 정보를 인터넷을 통해 직접 셔틀하는 것이 바람직할 것이다. 예를 들어, 이미지 색상의 경우 입력과 출력이 두드러진 가장자리의 위치를 공유합니다.

> To give the generator a means to circumvent the bottleneck for information like this, we add skip connections, following the general shape of a “U-Net” [47]. Specifically, we add skip connections between each layer $i$ and layer $n − i$, where n is the total number of layers. Each skip connection simply concatenates all channels at layer $i$ with those at layer $n − i$.
>> 발전기에 이와 같은 정보의 병목 현상을 우회할 수 있는 수단을 제공하기 위해, 우리는 "U-Net"의 일반적인 모양을 따라 건너뛰기 연결을 추가한다[47]. 구체적으로, 우리는 각 계층 $i$와 계층 $n - i$ 사이에 건너뛰기 연결을 추가한다. 여기서 n은 총 계층 수이다. 각 건너뛰기 연결은 단순히 $i$ 계층의 모든 채널을 $n - i$ 계층의 채널과 연결한다.

##### $\mathbf{3.2.2\;Markovian\;discriminator\;(PatchGAN)}$

> It is well known that the L2 loss – and L1, see Figure 3 – produces blurry results on image generation problems [31]. Although these losses fail to encourage highfrequency crispness, in many cases they nonetheless accurately capture the low frequencies. For problems where this is the case, we do not need an entirely new framework to enforce correctness at the low frequencies. L1 will already do.
>> L2 손실 및 L1 손실(그림 3 참조)은 이미지 생성 문제에 대해 흐릿한 결과를 초래한다는 것은 잘 알려져 있습니다 [31]. 이러한 손실은 고주파 선명도를 높이는 데 실패하지만, 그럼에도 불구하고 많은 경우 낮은 주파수를 정확하게 포착한다. 이러한 문제의 경우, 낮은 주파수에서 정확성을 강제하기 위해 완전히 새로운 프레임워크가 필요하지 않다. L1이면 됩니다.

> This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness (Eqn. 4). In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches. Therefore, we design a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an image is real or fake. We run this discriminator convolutationally across the image, averaging all  responses to provide the ultimate output of $D$.
>> 이는 GAN 판별기가 저주파수 정확성을 강제하기 위해 L1 항에 의존하여 고주파 구조만 모델링하도록 제한하도록 동기를 부여한다(Eqn. 4). 고주파를 모델링하려면 로컬 이미지 패치의 구조에 대한 주의를 제한하는 것으로 충분하다. 따라서 우리는 패치 규모에 따라 구조만 불이익을 주는 판별기 아키텍처(PatchGAN이라고 함)를 설계한다. 이 판별기는 이미지의 각 N × N 패치가 진짜인지 가짜인지 분류하려고 합니다. 우리는 이 판별기를 이미지 전반에 걸쳐 컨볼루션으로 실행하여 모든 응답을 평균화하여 $D$의 궁극적인 출력을 제공한다.

> In Section 4.4, we demonstrate that N can be much smaller than the full size of the image and still produce high quality results. This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied on arbitrarily large images. Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. This connection was previously explored in [35], and is also the common assumption in models of texture [15, 19] and style [14, 23, 20, 34]. Our PatchGAN can therefore be understood as a form of texture/style loss.
>> 섹션 4.4에서, 우리는 N이 이미지의 전체 크기보다 훨씬 작을 수 있고 여전히 고품질의 결과를 낼 수 있음을 보여준다. 이는 PatchGAN이 작을수록 매개 변수가 적고 실행 속도가 빠르며 임의적으로 큰 이미지에 적용할 수 있기 때문에 유리하다. 이러한 판별기는 패치 직경 이상으로 분리된 픽셀 간의 독립성을 가정하여 이미지를 마르코프 랜덤 필드로 효과적으로 모델링한다. 이러한 연결은 이전에 [35]에서 탐구되었으며, 텍스처 [15, 19] 및 스타일 [14, 23, 20, 34]의 모델에서 일반적인 가정이기도 하다. 따라서 우리의 PatchGAN은 텍스처/스타일 손실의 형태로 이해될 수 있다.

#### $\mathbf{3.3\;Optimization and inference}$

> To optimize our networks, we follow the standard approach from [22]: we alternate between one gradient descent step on D, then one step on G. We use minibatch SGD and apply the Adam solver [29].
>> 우리의 네트워크를 최적화하기 위해, 우리는 [22]의 표준 접근법을 따른다. 우리는 D에서 하나의 경사 하강 단계를 번갈아 가다가 G에서 한 단계를 번갈아 간다. 우리는 미니 배치 SGD를 사용하고 Adam solver[29]를 적용한다.

> At inference time, we run the generator net in exactly the same manner as during the training phase. This differs from the usual protocol in that we apply dropout at test time, and we apply batch normalization [26] using the statistics of the test batch, rather than aggregated statistics of the training batch. This approach to batch normalization, when the batch size is set to 1, has been termed “instance normalization” and has been demonstrated to be effective at image generation tasks [51]. In our experiments, we use batch sizes between 1 and 10 depending on the experiment.
>> 추론 시간에, 우리는 훈련 단계와 정확히 같은 방식으로 발전기 네트워크를 실행한다. 이는 테스트 시간에 드롭아웃을 적용하고, 훈련 배치의 집계 통계 대신 테스트 배치의 통계를 사용하여 배치 정규화 [26]를 적용한다는 점에서 일반적인 프로토콜과 다르다. 배치 크기가 1로 설정될 때 배치 정규화에 대한 이러한 접근 방식은 "인스턴스 정규화"라고 불렸으며 이미지 생성 작업에서 효과적인 것으로 입증되었다[51]. 우리의 실험에서 우리는 실험에 따라 1에서 10 사이의 배치 크기를 사용한다.

### $\mathbf{4\;Experiments}$

> To explore the generality of conditional GANs, we test the method on a variety of tasks and datasets, including both graphics tasks, like photo generation, and vision tasks, like semantic segmentation:
>> 조건부 GAN의 일반성을 탐색하기 위해 사진 생성과 같은 그래픽 작업과 의미 분할과 같은 비전 작업을 포함한 다양한 작업과 데이터 세트에서 방법을 테스트한다.

* > Semantic labels↔photo, trained on the Cityscapes dataset [11].
     >> 시맨틱 레이블↔사진, Cityscapes 데이터 세트에 대해 교육되었다[11].
* > Architectural labels→photo, trained on CMP Facades [42].
    >> 건축 라벨 → 사진, CMP 전면에서 교육됨 [42].
* > Map↔aerial photo, trained on data scraped from Google Maps.
    >> Google 지도에서 긁어낸 데이터에 대한 교육을 받은 Map↔ 사진.
* > BW→color photos, trained on [48].
    >> [48]에 대해 학습한 BW→ 컬러 사진.
* > Edges→photo, trained on data from [61] and [57]; binary edges generated using the HED edge detector [55] plus postprocessing.
    >> [61] 및 [57]의 데이터에 대해 훈련된 에지 → 사진. HED 에지 검출기 [55]와 후처리를 사용하여 생성된 이진 에지.
* > Sketch→photo: tests edges→photo models on human-drawn sketches from [17].
    >> Sketch→ photo: [17]의 사람이 그린 스케치에서 에지→ 사진 모델을 테스트합니다.
* > Day→night, trained on [30].
    >> 낮 → 밤, [30]에 교육됨.

> Details of training on each of these datasets are provided in the supplemental materials online. In all cases, the input and output are simply 1-3 channel images. Qualitative results are shown in Figures 7, 8, 9, 10, and 11, with additional results and failure cases in the materials online [https://phillipi.github.io/pix2pix/](https://phillipi.github.io/pix2pix/).
>> 이러한 각 데이터 세트에 대한 자세한 교육은 온라인 보충 자료에 제공된다. 모든 경우 입력 및 출력은 단순히 1-3 채널 영상입니다. 정성적 결과는 그림 7, 8, 9, 10 및 11에 나와 있으며, 추가 결과와 고장 사례는 온라인 [https://phillipi.github.io/pix2pix/](https://phillipi.github.io/pix2pix/)) 자료에 나와 있다.

#### $\mathbf{4.1\;Evaluation\;metrics}$

> Evaluating the quality of synthesized images is an open and difficult problem [49]. Traditional metrics such as perpixel mean-squared error do not assess joint statistics of the result, and therefore do not measure the very structure that structured losses aim to capture.
>> 합성 영상의 품질을 평가하는 것은 개방적이고 어려운 문제이다[49]. 픽셀당 평균 제곱 오류와 같은 전통적인 메트릭은 결과의 공동 통계를 평가하지 않으므로 구조화된 손실이 포착하는 것을 목표로 하는 바로 그 구조를 측정하지 않는다.

> In order to more holistically evaluate the visual quality of our results, we employ two tactics. First, we run “real vs fake” perceptual studies on Amazon Mechanical Turk (AMT). For graphics problems like colorization and photo generation, plausibility to a human observer is often the ultimate goal. Therefore, we test our map generation, aerial photo generation, and image colorization using this approach.
>> 결과의 시각적 품질을 보다 전체적으로 평가하기 위해 두 가지 전략을 사용한다. 첫째, 우리는 Amazon Mechanical Turk(AMT)에 대한 "실제 대 가짜" 지각 연구를 실행한다. 색칠 및 사진 생성과 같은 그래픽 문제의 경우 인간 관찰자에 대한 타당성이 종종 궁극적인 목표가 된다. 따라서 이 접근 방식을 사용하여 지도 생성, 항공 사진 생성 및 이미지 색칠을 테스트한다.

![Table 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Table-1.JPG)

> Second, we measure whether or not our synthesized cityscapes are realistic enough that off-the-shelf recognition system can recognize the objects in them. This metric is similar to the “inception score” from [49], the object detection evaluation in [52], and the “semantic interpretability” measures in [58] and [39].
>> 둘째, 우리는 우리의 합성된 도시 풍경이 기성 인식 시스템이 그 안에 있는 물체를 인식할 수 있을 정도로 충분히 현실적인지 여부를 측정한다. 이 지표는 [49]의 "인셉션 점수", [52]의 객체 감지 평가 및 [58] 및 [39]의 "의미적 해석 가능성" 측정과 유사하다.

> AMT perceptual studies For our AMT experiments, we followed the protocol from [58]: Turkers were presented with a series of trials that pitted a “real” image against a “fake” image generated by our algorithm. On each trial, each image appeared for 1 second, after which the images disappeared and Turkers were given unlimited time to respond as to which was fake. The first 10 images of each session were practice and Turkers were given feedback. No feedback was provided on the 40 trials of the main experiment. Each session tested just one algorithm at a time, and Turkers were not allowed to complete more than one session. ∼ 50 Turkers evaluated each algorithm. All images were presented at 256 × 256 resolution. Unlike [58], we did not include vigilance trials. For our colorization experiments, the real and fake images were generated from the same grayscale input. For map↔aerial photo, the real and fake images were not generated from the same input, in order to make the task more difficult and avoid floor-level results.
>> AMT 지각 연구 AMT 실험의 경우 [58]의 프로토콜을 따랐습니다. 터커는 알고리즘에 의해 생성된 "가짜" 이미지에 "실제" 이미지를 맞추는 일련의 시도를 받았다. 각각의 실험에서 각각의 이미지는 1초 동안 나타났고, 그 후 이미지는 사라졌고, 터커들은 어떤 것이 가짜인지에 대해 반응할 수 있는 무제한의 시간이 주어졌다. 각 세션의 처음 10개의 이미지는 연습이었고 Turker는 피드백을 받았다. 본 실험의 40번의 시도에 대한 피드백은 제공되지 않았다. 각 세션은 한 번에 하나의 알고리즘만 테스트했으며, 터커는 두 개 이상의 세션을 완료할 수 없었다. ~ 50명의 Turker가 각 알고리즘을 평가했습니다. 모든 이미지는 256 × 256 해상도로 표시되었다. [58]과 달리, 우리는 경계 시험을 포함하지 않았다. 색칠 실험의 경우 실제 이미지와 가짜 이미지가 동일한 그레이스케일 입력에서 생성되었다. map↔facebook 사진의 경우 작업을 더 어렵게 만들고 바닥 수준의 결과를 피하기 위해 실제 이미지와 가짜 이미지가 동일한 입력에서 생성되지 않았다.

> FCN-score While quantitative evaluation of generative models is known to be challenging, recent works [49, 52, 58, 39] have tried using pre-trained semantic classifiers to measure the discriminability of the generated stimuli as a pseudo-metric. The intuition is that if the generated images are realistic, classifiers trained on real images will be able to classify the synthesized image correctly as well. To this end, we adopt the popular FCN-8s [36] architecture for semantic segmentation, and train it on the cityscapes dataset. We then score synthesized photos by the classification accuracy against the labels these photos were synthesized from.
>> FCN 점수 생성 모델의 정량적 평가는 어려운 것으로 알려져 있지만, 최근 연구들[49, 52, 58, 39]은 사전 훈련된 의미 분류기를 사용하여 생성된 자극의 판별성을 의사 측정하려고 시도했다. 직관적으로 생성된 이미지가 사실적일 경우 실제 이미지에 대해 훈련된 분류기가 합성된 이미지를 올바르게 분류할 수 있다. 이를 위해, 우리는 의미 분할을 위해 인기 있는 FCN-8s [36] 아키텍처를 채택하고 도시 경관 데이터 세트에서 훈련한다. 그런 다음 이러한 사진이 합성된 레이블에 대해 분류 정확도에 따라 합성된 사진을 점수화한다.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-2.JPG)

> Figure 3: Different losses induce different quality of results. Each column shows results trained under a different loss. Please see https://phillipi.github.io/pix2pix/ for additional examples.
>> 그림 3: 손실이 다르면 결과의 품질이 달라진다. 각 열에는 서로 다른 손실 하에서 훈련된 결과가 표시됩니다. 추가 예는 https://phillipi.github.io/pix2pix/을 참조하십시오.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-4.JPG)

> Figure 4: Adding skip connections to an encoder-decoder to create a “U-Net” results in much higher quality results.
>> 그림 4: "U-Net"을 만들기 위해 인코더-디코더에 건너뛰기 연결을 추가하면 훨씬 더 높은 품질의 결과를 얻을 수 있습니다.

![Table 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Table-2.JPG)

> Table 2: FCN-scores for different receptive field sizes of the discriminator, evaluated on Cityscapes labels→photos. Note that input images are 256 × 256 pixels and larger receptive fields are padded with zeros.
>> 표 2: 판별기의 서로 다른 수용 필드 크기에 대한 FCN-점수로, Cityscapes 라벨→1999에서 평가된다. 입력 이미지는 256 × 256 픽셀이며 더 큰 수용 필드는 0으로 채워집니다.

#### $\mathbf{4.2\;Analysis\;of\;the\;objective\;function}$

> Which components of the objective in Eqn. 4 are important? We run ablation studies to isolate the effect of the L1 term, the GAN term, and to compare using a discriminator conditioned on the input (cGAN, Eqn. 1) against using an unconditional discriminator (GAN, Eqn. 2).
>> 등식 4의 목표 중 중요한 구성요소는 무엇입니까? 우리는 L1 항, GAN 항의 효과를 분리하고 입력(cGAN, Eqn.1)에 대해 조건화된 판별기를 사용하는 것과 무조건 판별기(GAN, Eqn.2)를 사용하는 것을 비교하기 위해 절제 연구를 실행한다.

> Figure 3 shows the qualitative effects of these variations on two labels→photo problems. L1 alone leads to reasonable but blurry results. The cGAN alone (setting λ = 0 in Eqn. 4) gives much sharper results, but introduces visual artifacts on certain applications. Adding both terms together (with λ = 100) reduces these artifacts.
>> 그림 3은 이러한 변형이 두 개의 라벨→사진 문제에 미치는 질적 영향을 보여준다. L1만으로도 합리적이지만 흐릿한 결과를 얻을 수 있습니다. cGAN 단독(등식 4에서 α = 0 설정)은 훨씬 더 날카로운 결과를 제공하지만 특정 응용 프로그램에 시각적 아티팩트를 도입한다. 두 항을 모두 더하면(θ = 100) 이러한 아티팩트가 줄어듭니다.

> We quantify these observations using the FCN-score on the cityscapes labels→photo task (Table 1): the GAN-based objectives achieve higher scores, indicating that the synthesized images include more recognizable structure. We also test the effect of removing conditioning from the discriminator (labeled as GAN). In this case, the loss does not  penalize mismatch between the input and output; it only cares that the output look realistic. This variant results in very poor performance; examining the results reveals that the generator collapsed into producing nearly the exact same output regardless of input photograph. Clearly it is important, in this case, that the loss measure the quality of the match between input and output, and indeed cGAN performs much better than GAN. Note, however, that adding an L1 term also encourages that the output respect the input, since the L1 loss penalizes the distance between ground truth outputs, which correctly match the input, and synthesized outputs, which may not. Correspondingly, L1+GAN is also effective at creating realistic renderings that respect the input label maps. Combining all terms, L1+cGAN, performs similarly well.
>> 우리는 도시경관 레이블→사진 작업(표 1)의 FCN 점수를 사용하여 이러한 관찰을 정량화한다. GAN 기반 목표는 더 높은 점수를 달성하여 합성된 이미지에 더 인식 가능한 구조가 포함되어 있음을 나타낸다. 우리는 또한 판별기(GAN으로 레이블링됨)에서 조건화를 제거하는 효과를 테스트한다. 이 경우, 손실은 입력과 출력 사이의 불일치에 불이익을 주지는 않는다. 다만 출력이 현실적으로 보이는 것에만 신경을 쓴다. 이 변형은 매우 낮은 성능을 초래한다. 결과를 조사하면 입력 사진과 관계없이 발전기가 붕괴되어 거의 동일한 출력을 생성한다는 것을 알 수 있다. 이 경우 손실이 입력과 출력 사이의 일치 품질을 측정하는 것이 분명히 중요하며, 실제로 cGAN이 GAN보다 훨씬 더 잘 수행된다. 그러나 L1 항을 추가하면 입력과 정확하게 일치하는 실측 출력과 그렇지 않을 수 있는 합성 출력 사이의 거리에 불이익을 주기 때문에 출력이 입력을 존중하는 것도 권장된다. 이에 대응하여, L1+GAN은 입력 레이블 맵을 존중하는 사실적인 렌더링을 생성하는 데도 효과적이다. 모든 항, L1+cGAN을 결합하면 비슷하게 잘 작동합니다.

> Colorfulness A striking effect of conditional GANs is that they produce sharp images, hallucinating spatial structure even where it does not exist in the input label map. One might imagine cGANs have a similar effect on “sharpening” in the spectral dimension – i.e. making images more colorful. Just as L1 will incentivize a blur when it is uncertain where exactly to locate an edge, it will also incentivize an average, grayish color when it is uncertain which of several plausible color values a pixel should take on. Specially, L1 will be minimized by choosing the median of of the conditional probability density function over possible colors. An adversarial loss, on the other hand, can in principle become aware that grayish outputs are unrealistic, and encourage matching the true color distribution [22]. In Figure 6, we investigate if our cGANs actually achieve this effect on the Cityscapes dataset. The plots show the marginal distributions over output color values in Lab color space. The ground truth distributions are shown with a dotted line. It is apparent that L1 leads to a narrower distribution than the ground truth, confirming the hypothesis that L1 encourages average, grayish colors. Using a cGAN, on the other hand, pushes the output distribution closer to the ground truth.
>> 색상성 조건부 GAN의 두드러진 효과는 입력 레이블 맵에 존재하지 않는 곳에서도 날카로운 이미지를 생성하여 공간 구조를 환각시키는 것이다. cGAN이 스펙트럼 차원에서 "선명화"에 유사한 영향을 미치는 것을 상상할 수 있다. 즉, 즉 이미지를 더욱 다채롭게 만드는 것이다. L1이 가장자리의 위치를 정확히 알 수 없을 때 흐릿한 색을 유도하는 것처럼, 픽셀이 몇 가지 그럴듯한 색상 값 중 어느 것을 취해야 할지 확실하지 않을 때 평균적인 회색을 유도한다. 특히, L1은 가능한 색상에 대한 조건부 확률 밀도 함수의 중위수를 선택함으로써 최소화될 것이다. 반면, 적대적 손실은 원칙적으로 회색빛 출력이 비현실적이라는 것을 인식하고 실제 색상 분포와 일치하도록 장려할 수 있다[22]. 그림 6에서, 우리는 cGAN이 실제로 Cityscapes 데이터 세트에 이러한 효과를 달성하는지 조사한다. 그래프에는 Lab 색 공간의 출력 색 값에 대한 주변 분포가 표시됩니다. 실측 사실 분포는 점선으로 표시됩니다. L1은 실측값보다 더 좁은 분포를 보이며, 이는 L1이 평균적인 회색빛을 장려한다는 가설을 뒷받침한다. 반면, acGAN을 사용하면 출력 분포가 실측값에 더 가깝게 된다.

#### $\mathbf{4.3.\;Analysis\;of\;the\;generator\;architecture}$

> A U-Net architecture allows low-level information to shortcut across the network. Does this lead to better results? Figure 4 compares the U-Net against an encoder decoder on cityscape generation. The encoder-decoder is created simply by severing the skip connections in the UNet. The encoder-decoder is unable to learn to generate realistic images in our experiments. The advantages of the U-Net appear not to be specific to conditional GANs: when both U-Net and encoder-decoder are trained with an L1 loss, the U-Net again achieves the superior results (Figure 4).

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-5.JPG)

> Figure 5: Patch size variations. Uncertainty in the output manifests itself differently for different loss functions. Uncertain regions become blurry and desaturated under L1. The 1x1 PixelGAN encourages greater color diversity but has no effect on spatial statistics. The 16x16 PatchGAN creates locally sharp results, but also leads to tiling artifacts beyond the scale it can observe. The 70×70 PatchGAN forces outputs that are sharp, even if incorrect, in both the spatial and spectral (colorfulness) dimensions. The full 286×286 ImageGAN produces results that are visually similar to the 70×70 PatchGAN, but somewhat lower quality according to our FCN-score metric (Table 2). Please see https://phillipi.github.io/pix2pix/ for additional examples.
>> U-Net 아키텍처는 낮은 수준의 정보를 네트워크를 통해 바로 가기를 허용합니다. 이것이 더 나은 결과로 이어집니까? 그림 4는 U-Net을 도시경관 생성에 대한 인코더 디코더와 비교한다. 인코더-디코더는 UNet에서 건너뛰기 연결을 끊는 것만으로 생성됩니다. 인코더-디코더는 실험에서 사실적인 이미지를 생성하는 방법을 배울 수 없다. U-Net의 장점은 조건부 GAN에 국한되지 않는 것으로 보인다. U-Net과 인코더-디코더가 L1 손실로 훈련되면 U-Net이 다시 우수한 결과를 달성한다(그림 4).

![Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-6.JPG)

> Figure 6: Color distribution matching property of the cGAN, tested on Cityscapes. (c.f. Figure 1 of the original GAN paper [22]). Note that the histogram intersection scores are dominated by differences in the high probability region, which are imperceptible in the plots, which show log probability and therefore emphasize differences in the low probability regions.
>> 그림 6: Cityscapes에서 테스트한 cGAN의 색상 분포 일치 특성. (c.f. 원본 GAN 논문의 그림 1 [22]) 히스토그램 교차점 점수는 확률도에서 감지할 수 없는 높은 확률 영역의 차이에 의해 지배된다는 점에 유의하십시오. 이러한 차이는 로그 확률을 나타내므로 낮은 확률 영역의 차이를 강조합니다.

![Figure 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-7.JPG)

> Figure 7: Example results on Google Maps at 512x512 resolution (model was trained on images at 256 × 256 resolution, and run convolutionally on the larger images at test time). Contrast adjusted for clarity.
>> 그림 7: 512x512 해상도의 Google 지도의 예제 결과(모델은 256x256 해상도의 이미지에 대해 학습되었으며 테스트 시 더 큰 이미지에서 컨볼루션으로 실행됨) 선명도를 위해 대비가 조정되었습니다.

#### $\mathbf{4.4.\;From\;PixelGANs\;to\;PatchGans\;to\;ImageGANs}$

> We test the effect of varying the patch size N of our discriminator receptive fields, from a 1 × 1 “PixelGAN” to a full 286 × 286 “ImageGAN”1 . Figure 5 shows qualitative 1We achieve this variation in patch size by adjusting the depth of the GAN discriminator. Details of this process, and the discriminator architecresults of this analysis and Table 2 quantifies the effects using the FCN-score. Note that elsewhere in this paper, unless specified, all experiments use 70 × 70 PatchGANs, and for this section all experiments use an L1+cGAN loss.
>> 우리는 1 × 1 "픽셀GAN"에서 전체 286 × 286 "ImageGAN"까지 판별기 수용 필드의 패치 크기 N을 변화시키는 효과를 테스트한다. 그림 5는 정성적 1GAN 판별기의 깊이를 조정하여 패치 크기의 이러한 변화를 달성한다. 이 프로세스에 대한 세부 정보와 이 분석의 판별기 설계 결과 및 표 2는 FCN 점수를 사용하여 효과를 정량화한다. 이 논문의 다른 부분에서는 명시되지 않는 한 모든 실험은 70 × 70 PatchGAN을 사용하며, 이 섹션의 경우 모든 실험은 L1+cGAN 손실을 사용한다.

> The PixelGAN has no effect on spatial sharpness, but does increase the colorfulness of the results (quantified in Figure 6). For example, the bus in Figure 5 is painted gray when the net is trained with an L1 loss, but becomes red with the PixelGAN loss. Color histogram matching is a common problem in image processing [46], and PixelGANs may be a promising lightweight solution.
>> PixelGAN은 공간 선명도에 영향을 미치지 않지만 결과의 색채를 증가시킨다(그림 6에서 정량화). 예를 들어, 그림 5의 버스는 네트가 L1 손실로 훈련될 때 회색으로 칠해지지만 픽셀 GAN 손실로 인해 빨간색이 됩니다. 색상 히스토그램 매칭은 이미지 처리에서 일반적인 문제이며 [46] PixelGAN은 유망한 경량 솔루션일 수 있다.

> Using a 16×16 PatchGAN is sufficient to promote sharp outputs, and achieves good FCN-scores, but also leads to tiling artifacts. The 70 × 70 PatchGAN alleviates these artifacts and achieves similar scores. Scaling beyond this, to the full 286 × 286 ImageGAN, does not appear to improve the visual quality of the results, and in fact gets a considerably lower FCN-score (Table 2). This may be because the ImageGAN has many more parameters and greater depth than the 70 × 70 PatchGAN, and may be harder to train.
>> 16×16 PatchGAN을 사용하면 예리한 출력을 촉진할 수 있으며, 양호한 FCN 점수를 달성하지만 타일링 아티팩트로 이어진다. 70 × 70 PatchGAN은 이러한 아티팩트를 완화하고 유사한 점수를 달성한다. 이를 넘어 전체 286 x 286 이미지로 확장GAN은 결과의 시각적 품질을 개선하지 않는 것으로 보이며, 실제로 상당히 낮은 FCN 점수를 받는다(표 2). 이는 이미지 때문일 수 있습니다.GAN은 70 × 70 PatchGAN보다 훨씬 더 많은 매개 변수와 더 큰 깊이를 가지고 있으며 훈련하기 어려울 수 있다.

![Fugure 8](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-8.JPG)

![Table 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Table-3.JPG)

![Table 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Table-4.JPG)

![Table 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Table-5.JPG)

> Fully-convolutional translation An advantage of the PatchGAN is that a fixed-size patch discriminator can be applied to arbitrarily large images. We may also apply the generator convolutionally, on larger images than those on which it was trained. We test this on the map↔aerial photo task. After training a generator on 256×256 images, we test it on 512×512 images. The results in Figure 7 demonstrate the effectiveness of this approach.
>> 완전 컨볼루션 변환 PatchGAN의 장점은 고정 크기의 패치 판별기를 임의의 큰 이미지에 적용할 수 있다는 것이다. 우리는 또한 발전기를 훈련받은 이미지보다 더 큰 이미지에 컨볼루션으로 적용할 수 있다. 우리는 이것을 map↔filename 사진 작업에서 테스트한다. 256×256 이미지에서 생성기를 교육한 후 512×512 이미지에서 테스트한다. 그림 7의 결과는 이 접근법의 효과를 입증한다.

#### $\mathbf{4.5.\;Perceptual\;validation}$

> We validate the perceptual realism of our results on the tasks of map↔aerial photograph and grayscale→color. Results of our AMT experiment for map↔photo are given in Table 3. The aerial photos generated by our method fooled participants on 18.9% of trials, significantly above the L1 baseline, which produces blurry results and nearly never fooled participants. In contrast, in the photo→map direction our method only fooled participants on 6.1% of trials, and this was not significantly different than the performance of the L1 baseline (based on bootstrap test). This may be because minor structural errors are more visible in maps, which have rigid geometry, than in aerial photographs, which are more chaotic. 
>> 우리는 지도↔사진과 그레이스케일→컬러 작업에 대한 결과의 지각적 사실성을 검증한다. map↔사진에 대한 AMT 실험 결과는 표 3에 나와 있습니다. 우리의 방법에 의해 생성된 항공 사진은 실험의 18.9%에서 참가자를 속였는데, 이는 L1 기준선을 크게 초과하여 흐릿한 결과를 생성하고 참가자를 거의 속이지 않았다. 대조적으로, photo→map 방향에서 우리의 방법은 시행의 6.1%에서만 참가자를 속였고, 이는 (부트스트랩 테스트 기반) L1 기준의 성능과 크게 다르지 않았다. 이는 경직된 기하학적 구조를 가진 지도에서 사소한 구조적 오류가 더 혼돈스러운 항공사진에서보다 더 잘 드러나기 때문일 것이다.

> We trained colorization on ImageNet [48], and tested on the test split introduced by [58, 32]. Our method, with L1+cGAN loss, fooled participants on 22.5% of trials (Table 4). We also tested the results of [58] and a variant of their method that used an L2 loss (see [58] for details). The conditional GAN scored similarly to the L2 variant of [58] (difference insignificant by bootstrap test), but fell short of [58]’s full method, which fooled participants on 27.8% of trials in our experiment. We note that their method was specifically engineered to do well on colorization.
>> ImageNet [48]에서 색칠을 교육하고 [58, 32]에 의해 도입된 테스트 분할에서 테스트했다. L1+cGAN 손실이 발생한 우리의 방법은 시행의 22.5%에서 참가자를 속였다(표 4). 또한 [58]의 결과와 L2 손실을 사용한 방법의 변형도 테스트했습니다(자세한 내용은 [58] 참조). 조건부 GAN은 [58]의 L2 변형(부트스트랩 테스트에 의해 중요하지 않은 차이)과 유사하게 점수를 매겼지만, 우리 실험의 27.8%에서 참가자들을 속인 [58]의 전체 방법에는 미치지 못했다. 우리는 그들의 방법이 색칠을 잘하도록 특별히 설계되었다는 것을 주목한다.

#### $\mathbf{4.6.\;Semantic\;segmentation}$

> Conditional GANs appear to be effective on problems where the output is highly detailed or photographic, as is common in image processing and graphics tasks. What about vision problems, like semantic segmentation, where the output is instead less complex than the input? 
>> 조건부 GAN은 이미지 처리 및 그래픽 작업에서 일반적인 것처럼 출력이 매우 상세하거나 사진인 문제에 효과적인 것으로 보인다. 대신 출력이 입력보다 덜 복잡한 의미 분할과 같은 비전 문제는 어떻습니까?

> To begin to test this, we train a cGAN (with/without L1 loss) on cityscape photo→labels. Figure 11 shows qualitative results, and quantitative classification accuracies are reported in Table 5. Interestingly, cGANs, trained without the L1 loss, are able to solve this problem at a reasonable degree of accuracy. To our knowledge, this is the first demonstration of GANs successfully generating “labels”, which are nearly discrete, rather than “images”, with their continuousvalued variation2 . Although cGANs achieve some success, they are far from the best available method for solving this problem: simply using L1 regression gets better scores than using a cGAN, as shown in Table 5. We argue that for vision problems, the goal (i.e. predicting output close to ground truth) may be less ambiguous than graphics tasks, and reconstruction losses like L1 are mostly sufficient.
>> 이를 테스트하기 위해 cityscape photo→facebook에서 (L1 손실이 있음/없음) acGAN을 훈련시킨다. 그림 11은 정성적 결과를 나타내며, 정량적 분류 정확도는 표 5에 보고된다. 흥미롭게도, L1 손실 없이 훈련된 cGAN은 합리적인 수준의 정확도로 이 문제를 해결할 수 있다. 우리가 아는 한, 이것은 연속적인 값 변동으로 "이미지"가 아닌 거의 이산적인 "라벨"을 성공적으로 생성하는 GAN의 첫 번째 시연이다2. cGAN은 어느 정도 성공을 거두지만, 이 문제를 해결하기 위해 사용할 수 있는 최상의 방법과는 거리가 멀다. 단순히 L1 회귀 분석을 사용하는 것이 c를 사용하는 것보다 더 나은 점수를 얻는다.GAN, 표 5와 같다. 우리는 비전 문제의 경우 목표(즉, 실측 진실에 가까운 출력 예측)가 그래픽 작업보다 덜 모호할 수 있으며, L1과 같은 재구성 손실은 대부분 충분하다고 주장한다.

#### $\mathbf{4.7.\;Community-driven\;Research}$

> Since the initial release of the paper and our pix2pix codebase, the Twitter community, including computer vision and graphics practitioners as well as artists, have successfully applied our framework to a variety of novel imageto-image translation tasks, far beyond the scope of the original paper. Figure 10 shows just a few examples from the #pix2pix hashtag, such as Sketch → Portrait, ”Do as I Do” pose transfer, Depth→Streetview, Background removal, Palette generation, Sketch→Pokemon, as well as the bizarrely popular #edges2cats.
>> 논문과 pix2pix 코드베이스의 최초 공개 이후, 예술가뿐만 아니라 컴퓨터 비전 및 그래픽 실무자를 포함한 트위터 커뮤니티는 원본 논문의 범위를 훨씬 뛰어넘는 다양한 새로운 이미지-이미지 번역 작업에 우리의 프레임워크를 성공적으로 적용했다. 그림 10은 Sketch → Portrait, "Do as I Do" 포즈 전송, Depth → Streetview, Background 제거, Palette 생성, Sketch→ Porte 및 기이하게 인기 있는 #sket2cats와 같은 #skit2 해시태그의 몇 가지 예를 보여줍니다.

![Figure 9](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-9.JPG)

> Figure 9: Results of our method on several tasks (data from [42] and [17]). Note that the sketch→photo results are generated by a model trained on automatic edge detections and tested on human-drawn sketches. Please see online materials for additional examples.
>> 그림 9: 여러 작업에 대한 우리 방법의 결과([42] 및 [17]의 데이터) sketch→ 사진 결과는 자동 에지 감지에 대해 훈련된 모델에 의해 생성되고 사람이 그린 스케치에 대해 테스트된다. 추가 예는 온라인 자료를 참조하십시오.

![Figure 10](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-10.JPG)

> Figure 10: Example applications developed by online community based on our pix2pix codebase: #edges2cats [3] by Christopher Hesse, Sketch → Portrait [7] by Mario Kingemann, “Do As I Do” pose transfer [2] by Brannon Dorsey, Depth→ Streetview [5] by Jasper van Loenen, Background removal [6] by Kaihu Chen, Palette generation [4] by Jack Qiao, and Sketch→ Pokemon [1] by Bertrand Gondouin.
>> 그림 10: 온라인 커뮤니티가 pix2cats 코드베이스를 기반으로 개발한 예제 응용 프로그램: #cats2cats [3] by Christopher Hesse, Sketch → Portrait [7] by Mario Kingmann의 "Do As I Do" 포즈 전송 [2] by Brannon Dorsey, Depth→ Streetview [5] by Jacebook [5] by Jas I Do [5] by Jas I Dorse By Chen [5], Jas I Do], Chen]베르트랑 곤두인의 etch→ 포켓몬 [1]입니다.

![Figure 11](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Image-to-Image-GAN/Figure-11.JPG)

> Figure 11: Applying a conditional GAN to semantic segmentation. The cGAN produces sharp images that look at glance like the ground truth, but in fact include many small, hallucinated objects.
>> 그림 11: 의미 분할에 조건부 GAN 적용 cGAN은 겉으로 보기에는 사실처럼 보이지만 실제로는 많은 작고 환각된 물체를 포함하는 날카로운 이미지를 생성한다.

### $\mathbf{5.\;Conclusion}$

> The results in this paper suggest that conditional adversarial networks are a promising approach for many imageto-image translation tasks, especially those involving highly structured graphical outputs. These networks learn a loss adapted to the task and data at hand, which makes them applicable in a wide variety of settings.
>> 본 논문의 결과는 조건부 적대적 네트워크가 특히 고도로 구조화된 그래픽 출력을 포함하는 많은 이미지-이미지 변환 작업에 유망한 접근 방식임을 시사한다. 이러한 네트워크는 당면한 작업과 데이터에 적응한 손실을 학습하여 매우 다양한 환경에서 적용할 수 있다.

#### $\mathbf{Acknowledgments}$

> We thank Richard Zhang, Deepak Pathak, and Shubham Tulsiani for helpful discussions, Saining Xie for help with the HED edge detector, and the online community for exploring many applications and suggesting improvements. This work was supported in part by NSF SMA-1514512, NGA NURI, IARPA via Air Force Research Laboratory, Intel Corp, Berkeley Deep Drive, and hardware donations by Nvidia.
>> 우리는 리처드 장, 디팍 파탁, 슈밤 툴시아니에게 도움이 되는 논의, Saining Xie에게 HED 에지 검출기에 대한 도움, 그리고 많은 응용 프로그램을 탐색하고 개선을 제안한 온라인 커뮤니티에 감사한다. 이 작업은 NSF SMA-1514512, NGA NURI, 공군 연구소를 통한 IARPA, 인텔, 버클리 딥 드라이브, 엔비디아의 하드웨어 기증에 의해 부분적으로 지원되었다.