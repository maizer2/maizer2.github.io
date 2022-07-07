---
layout: post 
title: "(GAN)Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization Translation"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review, 1.2.2.5. GAN]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

### [$$\mathbf{Arbitrary\;Style\;Transfer\;in\;Real-time\;with\;Adaptive\;Instance\;Normalization}$$](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)

##### $$\mathbf{Xun\;Huang,\;Serge\;Belongie}$$

##### $$\mathbf{Niloy\;J.\;Mitra,\;Peter\;Wonka,\;Jingwan\;Lu}$$

##### $$\mathbf{Department\;of\;Computer\;Science\;/\;Cornell\;Tech,\;Cornell\;University}$$

### $\mathbf{Abstract}$

> Gatys et al. recently introduced a neural algorithm that renders a content image in the style of another image, achieving so-called style transfer. However, their framework requires a slow iterative optimization process, which limits its practical application. Fast approximations with feed-forward neural networks have been proposed to speed up neural style transfer. Unfortunately, the speed improvement comes at a cost: the network is usually tied to a fixed set of styles and cannot adapt to arbitrary new styles. In this paper, we present a simple yet effective approach that for the first time enables arbitrary style transfer in real-time. At the heart of our method is a novel adaptive instance normalization (AdaIN) layer that aligns the mean and variance of the content features with those of the style features. Our method achieves speed comparable to the fastest existing approach, without the restriction to a pre-defined set of styles. In addition, our approach allows flexible user controls such as content-style trade-off, style interpolation, color & spatial controls, all using a single feed-forward neural network.
>> Gatys 등은 최근 콘텐츠 이미지를 다른 이미지의 스타일로 렌더링하는 신경 알고리듬을 도입하여 이른바 스타일 변환을 달성했다. 그러나 그들의 프레임워크는 느린 반복 최적화 프로세스를 필요로 하므로 실제 적용이 제한된다. 피드포워드 신경망을 사용한 빠른 근사치가 신경 스타일 변환 속도를 높이기 위해 제안되었다. 불행히도, 속도 향상에는 대가가 따른다: 네트워크는 대개 고정된 스타일 세트에 묶여 있고 임의의 새로운 스타일에 적응할 수 없다. 본 논문에서는 처음으로 임의의 스타일 변환을 실시간으로 가능하게 하는 간단하면서도 효과적인 접근 방식을 제시한다. 우리 방법의 핵심은 새로운 적응형 인스턴스 정규화(Ada)이다.IN) 내용 피쳐의 평균과 분산을 스타일 피쳐의 평균과 정렬하는 도면층입니다. 우리의 방법은 미리 정의된 스타일 세트에 제한 없이 기존 접근 방식 중 가장 빠른 속도를 달성한다. 또한 우리의 접근 방식은 단일 피드포워드 신경망을 사용하여 콘텐츠 스타일 트레이드오프, 스타일 보간, 색상 및 공간 제어와 같은 유연한 사용자 제어를 가능하게 한다.

### $\mathbf{1.\;Introduction}$

> The seminal work of Gatys et al. [16] showed that deep neural networks (DNNs) encode not only the content but also the style information of an image. Moreover, the image style and content are somewhat separable: it is possible to change the style of an image while preserving its content. The style transfer method of [16] is flexible enough to combine content and style of arbitrary images. However, it relies on an optimization process that is prohibitively slow.
>> Gatys 등의 주요 작품. [16] 심층 신경망(DNN)이 이미지의 콘텐츠뿐만 아니라 스타일 정보도 인코딩한다는 것을 보여주었다. 게다가 이미지 스타일과 콘텐츠는 어느 정도 분리할 수 있다: 콘텐츠를 보존하면서 이미지의 스타일을 변경할 수 있다. [16]의 스타일 변환 방법은 임의 이미지의 내용과 스타일을 결합할 수 있을 만큼 유연합니다. 그러나, 그것은 엄청나게 느린 최적화 프로세스에 의존한다.

> Significant effort has been devoted to accelerating neural style transfer. [24, 51, 31] attempted to train feed-forward neural networks that perform stylization with a single forward pass. A major limitation of most feed-forward methods is that each network is restricted to a single style. There are some recent works addressing this problem, but they are either still limited to a finite set of styles [11, 32, 55, 5], or much slower than the single-style transfer methods [6].
>> 신경 스타일 변환을 가속화하는 데 상당한 노력을 기울였다. [24, 51, 31]은 단일 전진 패스로 스타일링을 수행하는 피드포워드 신경망을 훈련시키려 했다. 대부분의 피드포워드 방법의 주요 한계는 각 네트워크가 단일 스타일로 제한된다는 것이다. 이 문제를 다루는 몇몇 최근 연구들이 있지만, 그것들은 여전히 유한한 스타일 세트[11, 32, 55, 5]로 제한되거나 단일 스타일 변환 방법[6]보다 훨씬 느리다.

> In this work, we present the first neural style transfer algorithm that resolves this fundamental flexibility-speed dilemma. Our approach can transfer arbitrary new styles in real-time, combining the flexibility of the optimizationbased framework [16] and the speed similar to the fastest feed-forward approaches [24, 52]. Our method is inspired by the instance normalization (IN) [52, 11] layer, which is surprisingly effective in feed-forward style transfer. To explain the success of instance normalization, we propose a new interpretation that instance normalization performs style normalization by normalizing feature statistics, which have been found to carry the style information of an image [16, 30, 33]. Motivated by our interpretation, we introduce a simple extension to IN, namely adaptive instance normalization (AdaIN). Given a content input and a style input, $AdaIN$ simply adjusts the mean and variance of the content input to match those of the style input. Through experiments, we find $AdaIN$ effectively combines the content of the former and the style latter by transferring feature statistics. A decoder network is then learned to generate the final stylized image by inverting the $AdaIN$ output back to the image space. Our method is nearly three orders of magnitude faster than [16], without sacrificing the flexibility of transferring inputs to arbitrary new styles. Furthermore, our approach provides abundant user controls at runtime, without any modification to the training process.
>> 본 연구에서는 이러한 근본적인 유연성-속도 딜레마를 해결하는 첫 번째 신경 스타일 변환 알고리듬을 제시한다. 우리의 접근 방식은 최적화 기반 프레임워크의 유연성과 가장 빠른 피드포워드 접근법과 유사한 속도를 결합하여 임의의 새로운 스타일을 실시간으로 전송할 수 있다[24, 52]. 우리의 방법은 피드포워드 스타일 변환에 놀랄 만큼 효과적인 인스턴스 정규화(IN) [52, 11] 계층에서 영감을 받았다. 인스턴스 정규화의 성공을 설명하기 위해, 우리는 이미지의 스타일 정보를 전달하는 것으로 밝혀진 특징 통계를 정규화하여 인스턴스 정규화가 스타일 정규화를 수행한다는 새로운 해석을 제안한다[16, 30, 33]. 우리의 해석에 자극을 받아, 우리는 IN에 대한 간단한 확장, 즉 적응형 인스턴스 정규화(Ada)를 도입한다.IN). 컨텐츠 입력과 스타일 입력이 주어지면, AdaIN은 단순히 내용 입력의 평균과 분산을 스타일 입력의 평균과 일치하도록 조정합니다. 실험을 통해 우리는 에이다를 찾는다.IN은 형상 통계를 전송하여 전자의 내용과 후자의 스타일을 효과적으로 결합한다. 디코더 네트워크는 Ada를 반전시켜 최종 양식화된 이미지를 생성하도록 학습된다.IN 출력을 영상 공간으로 다시 출력합니다. 우리의 방법은 임의의 새로운 스타일로 입력을 전송하는 유연성을 희생하지 않고 [16]보다 거의 3배 더 빠르다. 또한, 우리의 접근 방식은 훈련 과정을 변경하지 않고도 런타임에 풍부한 사용자 제어를 제공한다.

### $\mathbf{2.\;Related\;Work}$

> **Style transfer.** The problem of style transfer has its origin from non-photo-realistic rendering [28], and is closely related to texture synthesis and transfer [13, 12, 14]. Some early approaches include histogram matching on linear filter responses [19] and non-parametric sampling [12, 15]. These methods typically rely on low-level statistics and often fail to capture semantic structures. Gatys et al. [16] for the first time demonstrated impressive style transfer results by matching feature statistics in convolutional layers of a DNN. Recently, several improvements to [16] have been proposed. Li and Wand [30] introduced a framework based on markov random field (MRF) in the deep feature space to enforce local patterns. Gatys et al. [17] proposed ways to control the color preservation, the spatial location, and the scale of style transfer. Ruder et al. [45] improved the quality of video style transfer by imposing temporal constraints. 
>> ** 스타일 변환.** 스타일 변환의 문제는 비사진적 렌더링[28]에서 비롯되었으며 텍스처 합성 및 전송과 밀접한 관련이 있다. 일부 초기 접근법에는 선형 필터 응답에 대한 히스토그램 매칭[19]과 비모수 샘플링[12, 15]이 포함된다. 이러한 방법은 일반적으로 낮은 수준의 통계에 의존하며 종종 의미 구조를 포착하지 못한다. 게이티 외 [16] DNN의 컨볼루션 레이어에서 특징 통계를 일치시켜 인상적인 스타일 변환 결과를 처음으로 보여주었다. 최근 [16]에 대한 몇 가지 개선이 제안되었다. Li와 Wand[30]는 로컬 패턴을 적용하기 위해 심층 특징 공간에서 마르코프 랜덤 필드(MRF)를 기반으로 하는 프레임워크를 도입했다. 게이티 외 [17] 색채 보존, 공간적 위치, 스타일 변환의 크기를 조절하는 방법을 제안했다. 루더 외 [45] 시간적 제약을 가함으로써 비디오 스타일 변환의 품질을 향상시켰다.

> The framework of Gatys et al. [16] is based on a slow optimization process that iteratively updates the image to minimize a content loss and a style loss computed by a loss network. It can take minutes to converge even with modern GPUs. On-device processing in mobile applications is therefore too slow to be practical. A common workaround is to replace the optimization process with a feed-forward neural network that is trained to minimize the same objective [24, 51, 31]. These feed-forward style transfer approaches are about three orders of magnitude faster than the optimization-based alternative, opening the door to realtime applications. Wang et al. [53] enhanced the granularity of feed-forward style transfer with a multi-resolution architecture. Ulyanov et al. [52] proposed ways to improve the quality and diversity of the generated samples. However, the above feed-forward methods are limited in the sense that each network is tied to a fixed style. To address this problem, Dumoulin et al. [11] introduced a single network that is able to encode 32 styles and their interpolations. Concurrent to our work, Li et al. [32] proposed a feed-forward architecture that can synthesize up to 300 textures and transfer 16 styles. Still, the two methods above cannot adapt to arbitrary styles that are not observed during training.
>> Gatys 등의 프레임워크. [16]은 손실 네트워크에 의해 계산된 콘텐츠 손실과 스타일 손실을 최소화하기 위해 이미지를 반복적으로 업데이트하는 느린 최적화 프로세스를 기반으로 한다. 최신 GPU로도 수렴하는 데 몇 분이 걸릴 수 있다. 따라서 모바일 애플리케이션의 장치 내 처리 속도는 실용적이기에는 너무 느리다. 일반적인 해결 방법은 최적화 프로세스를 동일한 목표를 최소화하도록 훈련된 피드포워드 신경망으로 대체하는 것이다[24, 51, 31]. 이러한 피드포워드 스타일 변환 접근 방식은 최적화 기반 대안보다 약 3배 더 빠르며 실시간 애플리케이션의 문을 연다. 왕 외 [53] 다중 해상도 아키텍처를 통해 피드포워드 스타일 변환의 세분성을 개선했습니다. 율리야노프 외 [52] 생성된 샘플의 품질 및 다양성을 개선하기 위한 방법을 제안했습니다. 그러나, 위의 피드포워드 방법은 각 네트워크가 고정된 스타일에 묶여 있다는 점에서 제한적이다. 이 문제를 해결하기 위해 듀물린 외. [11] 32가지 스타일과 그들의 보간을 인코딩할 수 있는 단일 네트워크를 도입했다. 우리의 일과 동시에, Li et al. [32] 최대 300개의 텍스처를 합성하고 16개의 스타일을 전송할 수 있는 피드포워드 아키텍처를 제안했다. 그러나 위의 두 가지 방법은 훈련 중에 관찰되지 않는 임의 스타일에 적응할 수 없다.

> Very recently, Chen and Schmidt [6] introduced a feedforward method that can transfer arbitrary styles thanks to a style swap layer. Given feature activations of the content and style images, the style swap layer replaces the content features with the closest-matching style features in a patchby-patch manner. Nevertheless, their style swap layer creates a new computational bottleneck: more than 95% of the computation is spent on the style swap for 512 × 512 input images. Our approach also permits arbitrary style transfer, while being 1-2 orders of magnitude faster than [6].
>> 매우 최근에 Chen과 Schmidt[6]는 스타일 스왑 계층 덕분에 임의의 스타일을 변환할 수 있는 피드포워드 방법을 도입했다. 컨텐츠 및 스타일 이미지의 피쳐 활성화가 주어지면 스타일 스왑 계층은 패치별 방식으로 컨텐츠 피쳐를 가장 근접한 스타일 피쳐로 대체한다. 그럼에도 불구하고 스타일 스왑 계층은 새로운 계산 병목 현상을 유발한다. 512 × 512 입력 이미지에 대한 스타일 스왑에 계산의 95% 이상이 소비된다. 우리의 접근 방식은 또한 임의의 스타일 변환을 허용하지만 [6]보다 1-2배 빠르다.

> Another central problem in style transfer is which style loss function to use. The original framework of Gatys et al. [16] matches styles by matching the second-order statistics between feature activations, captured by the Gram matrix. Other effective loss functions have been proposed, such as MRF loss [30], adversarial loss [31], histogram loss [54], CORAL loss [41], MMD loss [33], and distance between channel-wise mean and variance [33]. Note that all the above loss functions aim to match some feature statistics between the style image and the synthesized image.
>> 스타일 변환의 또 다른 중심 문제는 어떤 스타일 손실 함수를 사용할 것인가이다. Gatys 등의 원래 프레임워크. [16] Gram 행렬에 의해 캡처된 형상 활성화 사이의 2차 통계량을 일치시켜 스타일을 일치시킨다. MRF 손실 [30], 적대적 손실 [31], 히스토그램 손실 [54], CORAL 손실 [41], MMD 손실 [33], 채널별 평균과 분산 사이의 거리 [33]와 같은 다른 효과적인 손실 함수가 제안되었다. 위의 모든 손실 함수는 스타일 영상과 합성 영상 사이의 일부 형상 통계를 일치시키는 것을 목표로 한다.

> **Deep generative image modeling.** There are several alternative frameworks for image generation, including variational auto-encoders [27], auto-regressive models [40], and generative adversarial networks (GANs) [18]. Remarkably, GANs have achieved the most impressive visual qualit$y$. Various improvements to the GAN framework have been proposed, such as conditional generation [43, 23], multistage processing [9, 20], and better training objectives [46, 1]. GANs have also been applied to style transfer [31] and cross-domain image generation [50, 3, 23, 38, 37, 25].
>> **심층 생성 이미지 모델링.** 가변 자동 인코더[27], 자동 회귀 모델[40] 및 생성 적대적 네트워크(GAN)를 포함한 이미지 생성을 위한 몇 가지 대안 프레임워크가 있다. 놀랍게도, GAN은 가장 인상적인 시각적 품질을 달성했다. 조건부 생성 [43, 23], 다단계 처리 [9, 20] 및 더 나은 훈련 목표 [46, 1]와 같은 GAN 프레임워크에 대한 다양한 개선이 제안되었다. GAN은 스타일 변환[31] 및 교차 도메인 이미지 생성[50, 3, 23, 38, 37, 25]에도 적용되었다.

### $\mathbf{3.\;Background}$

### $\mathbf{3.1.\;Batch\;Normalization}$

> The seminal work of Ioffe and Szegedy [22] introduced a batch normalization (BN) layer that significantly ease the training of feed-forward networks by normalizing feature statistics. BN layers are originally designed to accelerate training of discriminative networks, but have also been found effective in generative image modeling [42]. Given an input batch $x\in{}R^{N×C×H×W}$ , BN normalizes the mean and standard deviation for each individual feature channel:
>> 아이오페와 세르게디[22]의 주요 연구는 기능 통계를 정규화하여 피드포워드 네트워크의 훈련을 상당히 용이하게 하는 배치 정규화(BN) 계층을 도입했다. BN 레이어는 원래 차별적 네트워크의 훈련을 가속화하기 위해 설계되었지만, 생성 이미지 모델링에서도 효과적인 것으로 밝혀졌다[42]. 입력 배치 $x\in{}R^{N×C×H×W}$가 주어지면 BN은 각 개별 형상 채널에 대한 평균 및 표준 편차를 정규화한다.

$$BN(x)=γ(\frac{x−µ(x)}{σ(x)})+β$$

> where $γ,β\in{}R^{C}$ are affine parameters learned from data; $\mu{}(x),σ(x)\in{}R^{C}$ are the mean and standard deviation, computed across batch size and spatial dimensions independently for each feature channel:
>> 여기서 $γ,β\in{}R^{C}$는 데이터로부터 학습된 아핀 매개변수이고, $\mu{}(x),σ(x)\in{}R^{C}$ 는 각 형상 채널에 대해 독립적으로 배치 크기와 공간 차원에 걸쳐 계산되는 평균 및 표준 편차이다.

$$µ_{c}(x)=\frac{1}{NHW}\sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{nchw}$$

$$σ_{c}(x)=\sqrt{\frac{1}{NHW}\sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W}(x_{nchw}−µ_{c}(x))^{2}+ε}$$

> BN uses mini-batch statistics during training and replace them with popular statistics during inference, introducing discrepancy between training and inference. Batch renormalization [21] was recently proposed to address this issue by gradually using popular statistics during training. As another interesting application of BN, Li et al. [34] found that BN can alleviate domain shifts by recomputing popular statistics in the target domain. Recently, several alternative normalization schemes have been proposed to extend BN’s effectiveness to recurrent architectures [35, 2, 47, 8, 29, 44].
>> BN은 훈련 중에 미니 배치 통계를 사용하고 추론 중에 인기 있는 통계로 대체하여 훈련과 추론 사이의 불일치를 초래한다. 훈련 중에 인기 있는 통계를 점진적으로 사용하여 이 문제를 해결하기 위해 배치 재규격화[21]가 최근 제안되었다. BN의 또 다른 흥미로운 응용 프로그램으로서, Li 등[34]은 BN이 대상 도메인에서 인기 있는 통계를 재계산하여 도메인 이동을 완화할 수 있다는 것을 발견했다. 최근, BN의 효과를 반복 아키텍처로 확장하기 위한 몇 가지 대안적인 정규화 체계가 제안되었다[35, 2, 47, 8, 29, 44].

### $\mathbf{3.2.\;Instance\;Normalization}$

> In the original feed-forward stylization method [51], the style transfer network contains a BN layer after each convolutional layer. Surprisingly, Ulyanov et al. [52] found that significant improvement could be achieved simply by replacing BN layers with IN layers: 
>> 원래의 피드포워드 스타일화 방법[51]에서 스타일 변환 네트워크는 각 컨볼루션 레이어 뒤에 BN 레이어를 포함한다. 놀랍게도, 울리야노프 외 [52] BN 레이어를 IN 레이어로 교체하는 것만으로 상당한 개선이 가능하다는 것을 발견했다.

$$IN(x)=γ(\frac{x−µ(x)}{σ(x)})+β$$

> Different from BN layers, here $\mu{}(x)$ and $\sigma{}(x)$ are computed across spatial dimensions independently for each channel and each sample: 
>> BN 레이어와 달리, 여기서 $α(x)$와 $β(x)$는 각 채널과 각 샘플에 대해 독립적으로 공간 차원에 걸쳐 계산된다.

$$µ_{nc}(x)=\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{nchw}$$

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-1.PNG)

> Figure 1. To understand the reason for IN’s effectiveness in style transfer, we train an IN model and a BN model with (a) original images in MS-COCO [36], (b) contrast normalized images, and (c) style normalized images using a pre-trained style transfer network [24]. The improvement brought by IN remains significant even when all training images are normalized to the same contrast, but are much smaller when all images are (approximately) normalized to the same style. Our results suggest that IN performs a kind of style normalization.
>> 그림 1 스타일 변환에서 IN의 효과적 이유를 이해하기 위해, 우리는 (a) MS-COCO의 원본 이미지 [36], (b) 정규화된 이미지를 대조하고, (c) 사전 훈련된 스타일 변환 네트워크를 사용하여 스타일 정규화된 이미지를 가진 IN 모델과 BN 모델을 훈련한다[24]. IN이 가져온 개선은 모든 교육 이미지가 동일한 대비로 정규화된 경우에도 유의하지만 모든 이미지가 (대략적으로) 동일한 스타일로 정규화된 경우 훨씬 더 작다. 우리의 결과는 IN이 일종의 스타일 정규화를 수행함을 시사한다.

$$σ_{nc}(x)=\sqrt{\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}(x_{nchw}−µ_{nc}(x))^{2}+ε}$$

> Another difference is that IN layers are applied at test time unchanged, whereas BN layers usually replace minibatch statistics with population statistics.
>> IN 레이어는 시험 시간에 그대로 적용되는 반면, BN 레이어는 일반적으로 미니 배치 통계를 모집단 통계로 대체한다.

### $\mathbf{3.3.\;Conditional\;Instance\;Normalization}$

> Instead of learning a single set of affine parameters $\gamma{}$ and $\beta{}$, Dumoulin et al. [11] proposed a conditional instance normalization (CIN) layer that learns a different set of parameters $γ^{s}$ and $β^{s}$ for each style s:
>> 아핀 매개 변수 $\gamma{}$ 및 $\beta{}$의 단일 집합을 학습하는 대신 Dumoulin 등을 학습한다. [11] 각 스타일에 대해 서로 다른 매개 변수 집합 $α^{s}$ 및 $β^{s}$를 학습하는 조건부 인스턴스 정규화(CIN) 계층을 제안하였다.

$$CIN(x;s)=γ^{s}(\frac{x−µ(x)}{σ(x)})+β^{s}$$

> During training, a style image together with its index s are randomly chosen from a fixed set of styles $s\in{}{1, 2, ..., S}$ (S=32 in their experiments). The content image is then processed by a style transfer network in which the corresponding $\gamma{}$ s and $\beta{}$ s are used in the $CIN$ layers. Surprisingly, the network can generate images in completely different styles by using the same convolutional parameters but different affine parameters in IN layers. 
>> 훈련 중에 스타일 이미지와 인덱스는 고정된 스타일 집합 $s\in{}{1, 2, ..., S}$(실험에서 S=32)에서 무작위로 선택됩니다. 그런 다음 콘텐츠 이미지는 해당 $\gamma{}$s 및 $\beta{}$s가 $CIN$ 계층에서 사용되는 스타일 변환 네트워크에 의해 처리된다. 놀랍게도, 네트워크는 IN 레이어에서 동일한 컨볼루션 매개 변수를 사용하지만 다른 아핀 매개 변수를 사용하여 완전히 다른 스타일로 이미지를 생성할 수 있다.

> Compared with a network without normalization layers, a network with $CIN$ layers requires 2FS additional parameters, where F is the total number of feature maps in the network [11]. Since the number of additional parameters scales linearly with the number of styles, it is challenging to extend their method to model a large number of styles (e.g., tens of thousands). Also, their approach cannot adapt to arbitrary new styles without re-training the network.
>> 정규화 계층이 없는 네트워크와 비교하여, $CIN$ 계층이 있는 네트워크는 2FS 추가 매개 변수가 필요합니다. 여기서 F는 네트워크의 총 기능 맵 수입니다[11]. 추가 매개 변수의 수는 스타일 수에 따라 선형적으로 확장되므로, 방법을 확장하여 많은 스타일(예: 수만 개)을 모델링하는 것은 어렵다. 또한 그들의 접근 방식은 네트워크를 재교육하지 않고는 임의의 새로운 스타일에 적응할 수 없다.

### $\mathbf{4.\;Interpreting\;Instance\;Normalization}$

> Despite the great success of (conditional) instance normalization, the reason why they work particularly well for style transfer remains elusive. Ulyanov et al. [52] attribute the success of IN to its invariance to the contrast of the content image. However, IN takes place in the feature space, therefore it should have more profound impacts than a simple contrast normalization in the pixel space. Perhaps even more surprising is the fact that the affine parameters in IN can completely change the style of the output image.
>> (조건부) 인스턴스 정규화의 큰 성공에도 불구하고, 그것들이 스타일 변환에 특히 잘 작동하는 이유는 여전히 불분명하다. 율리야노프 외 [52] IN의 성공은 콘텐츠 이미지의 대비에 따른 불변성 덕분이다. 그러나 IN은 특징 공간에서 발생하므로 픽셀 공간에서 단순한 대비 정규화보다 더 깊은 영향을 미칠 수 있다. 아마도 더 놀라운 것은 IN의 아핀 파라미터가 출력 이미지의 스타일을 완전히 바꿀 수 있다는 사실이다.

> It has been known that the convolutional feature statistics of a DNN can capture the style of an image [16, 30, 33]. While Gatys et al. [16] use the second-order statistics as their optimization objective, Li et al. [33] recently showed that matching many other statistics, including channel-wise mean and variance, are also effective for style transfer. Motivated by these observations, we argue that instance normalization performs a form of style normalization by normalizing feature statistics, namely the mean and variance. Although DNN serves as a image descriptor in [16, 33], we believe that the feature statistics of a generator network can also control the style of the generated image. 
>> DNN의 컨볼루션 특징 통계는 이미지의 스타일을 포착할 수 있는 것으로 알려져 있다[16, 30, 33]. Gatys 등이 있는 동안. [16] 2차 통계량을 최적화 목표(Li 등)로 사용합니다. [33] 최근 채널별 평균과 분산을 포함한 다른 많은 통계를 일치시키는 것이 스타일 변환에도 효과적이라는 것을 보여주었다. 이러한 관찰에 의해, 우리는 인스턴스 정규화가 특징 통계, 즉 평균과 분산을 정규화함으로써 스타일 정규화의 한 형태를 수행한다고 주장한다. DNN이 [16, 33]에서 이미지 설명자 역할을 하지만, 생성기 네트워크의 특징 통계도 생성된 이미지의 스타일을 제어할 수 있다고 믿는다.

> We run the code of improved texture networks [52] to perform single-style transfer, with IN or BN layers. As expected, the model with IN converges faster than the BN model (Fig. 1 (a)). To test the explanation in [52], we then normalize all the training images to the same contrast by performing histogram equalization on the luminance channel. As shown in Fig. 1 (b), IN remains effective, suggesting the explanation in [52] to be incomplete. To verify our hypothesis, we normalize all the training images to the same style (different from the target style) using a pretrained style transfer network provided by [24]. According to Fig. 1 (c), the improvement brought by IN become much smaller when images are already style normalized. The remaining gap can explained by the fact that the style normalization with [24] is not perfect. Also, models with BN trained on style normalized images can converge as fast as models with IN trained on the original images. Our results indicate that IN does perform a kind of style normalization.
>> 개선된 텍스처 네트워크 코드[52]를 실행하여 IN 또는 BN 레이어를 사용하여 단일 스타일 변환을 수행한다. 예상대로 IN이 적용된 모델은 BN 모델보다 수렴이 빠르다(그림 1(a)). [52]의 설명을 테스트하기 위해, 우리는 휘도 채널에서 히스토그램 균등화를 수행하여 모든 훈련 이미지를 동일한 대비로 정규화한다. 도 1(b)에 도시된 바와 같이 IN은 여전히 유효하여, [52]의 설명이 불완전함을 시사한다. 우리의 가설을 검증하기 위해, 우리는 [24]에서 제공하는 사전 훈련된 스타일 변환 네트워크를 사용하여 모든 훈련 이미지를 동일한 스타일(대상 스타일과 다름)로 정규화한다. 그림 1(c)에 따르면, 이미 이미지가 스타일 정규화된 경우 IN에 의한 개선은 훨씬 더 작아진다. 나머지 간격은 [24]를 사용한 스타일 정규화가 완벽하지 않다는 사실로 설명할 수 있습니다. 또한 스타일 정규화 이미지에 대해 훈련된 BN을 가진 모델은 원본 이미지에 대해 훈련된 IN을 가진 모델만큼 빠르게 수렴할 수 있다. 우리의 결과는 IN이 일종의 스타일 정규화를 수행한다는 것을 나타낸다.

> Since BN normalizes the feature statistics of a batch of samples instead of a single sample, it can be intuitively understood as normalizing a batch of samples to be centered around a single style. Each single sample, however, may still have different styles. This is undesirable when we want to transfer all images to the same style, as is the case in the original feed-forward style transfer algorithm [51]. Although the convolutional layers might learn to compensate the intra-batch style difference, it poses additional challenges for training. On the other hand, IN can normalize the style of each individual sample to the target style. Training is facilitated because the rest of the network can focus on content manipulation while discarding the original style information. The reason behind the success of $CIN$ also becomes clear: different affine parameters can normalize the feature statistics to different values, thereby normalizing the output image to different styles.
>> BN은 단일 샘플 대신 샘플 배치의 피쳐 통계량을 정규화하므로 단일 스타일을 중심으로 샘플 배치를 정규화하는 것으로 직관적으로 이해할 수 있습니다. 그러나 각 표본은 여전히 다른 스타일을 가질 수 있습니다. 이것은 원래의 피드-포워드 스타일 변환 알고리즘의 경우와 같이 모든 이미지를 동일한 스타일로 전송하고자 할 때 바람직하지 않다[51]. 컨볼루션 레이어는 배치 내 스타일 차이를 보상하는 방법을 배울 수 있지만, 훈련에 추가적인 문제를 제기한다. 반면에 IN은 각 개별 표본의 스타일을 목표 스타일로 정규화할 수 있습니다. 네트워크의 나머지 부분은 원래 스타일 정보를 폐기하면서 콘텐츠 조작에 집중할 수 있기 때문에 교육이 용이하다. $CIN$의 성공 배경도 명확해진다. 다양한 아핀 매개 변수가 특징 통계를 다른 값으로 정규화하여 출력 이미지를 다른 스타일로 정규화할 수 있다.

### $\mathbf{5.\;Adaptive\;Instance\;Normalization}$

> If IN normalizes the input to a single style specified by the affine parameters, is it possible to adapt it to arbitrarily given styles by using adaptive affine transformations? Here, we propose a simple extension to IN, which we call adaptive instance normalization (AdaIN). $AdaIN$ receives a content input $x$ and a style input y, and simply aligns the channelwise mean and variance of $x$ to match those of $y$. Unlike BN, IN or CIN, $AdaIN$ has no learnable affine parameters. Instead, it adaptively computes the affine parameters from the style input:
>> IN이 아핀 파라미터에 의해 지정된 단일 스타일로 입력을 정규화하는 경우, 적응형 아핀 변환을 사용하여 임의로 주어진 스타일에 맞게 입력을 조정할 수 있습니까? 여기서는 적응형 인스턴스 정규화(Ada)라고 하는 IN에 대한 간단한 확장을 제안한다.IN). $AdaIN$은 콘텐츠 입력 $x$와 스타일 입력 y를 수신하고 단순히 $x$의 채널별 평균과 분산을 $y$의 평균과 분산에 일치하도록 정렬한다. BN, IN 또는 CIN과 달리 $AdaIN$에는 학습 가능한 아핀 매개 변수가 없다. 대신 스타일 입력에서 적응적으로 아핀 파라미터를 계산합니다.

$$AdaIN(x,y)=σ(y)(\frac{x−µ(x)}{σ(x)})+µ(y)$$

> in which we simply scale the normalized content input with $\sigma{}(y)$, and shift it with $\mu{}(y)$. Similar to IN, these statistics are computed across spatial locations.
>> 우리는 $\mu{}(y)$로 정규화된 콘텐츠 입력을 스케일링하고 $\mu{}(y)$로 이동하기만 하면 된다. IN과 유사하게, 이러한 통계는 공간적 위치에 걸쳐 계산된다.

> Intuitively, let us consider a feature channel that detects brushstrokes of a certain style. A style image with this kind of strokes will produce a high average activation for this feature. The output produced by $AdaIN$ will have the same high average activation for this feature, while preserving the spatial structure of the content image. The brushstroke feature can be inverted to the image space with a feed-forward decoder, similar to [10]. The variance of this feature channel can encoder more subtle style information, which is also transferred to the $AdaIN$ output and the final output image.
>> 직관적으로 특정 스타일의 브러시 스트로크를 감지하는 피쳐 채널을 고려해 보겠습니다. 이러한 유형의 스트로크가 있는 스타일 이미지는 이 기능에 대해 높은 평균 활성화를 생성합니다. $AdaIN$에 의해 생성된 출력은 콘텐츠 이미지의 공간 구조를 유지하면서 이 기능에 대해 동일한 높은 평균 활성화를 가질 것이다. 브러시 스트로크 기능은 [10]과 유사하게 피드포워드 디코더를 사용하여 영상 공간으로 반전될 수 있습니다. 이 형상 채널의 분산은 보다 미묘한 스타일 정보를 인코더할 수 있으며, 이는 $AdaIN$ 출력 및 최종 출력 영상으로도 전송된다.

> In short, $AdaIN$ performs style transfer in the feature space by transferring feature statistics, specifically the channel-wise mean and variance. Our $AdaIN$ layer plays a similar role as the style swap layer proposed in [6]. While the style swap operation is very time-consuming and memory-consuming, our $AdaIN$ layer is as simple as an IN layer, adding almost no computational cost.
>> 간단히 말해 $AdaIN$는 형상 통계, 특히 채널별 평균과 분산 등을 전송하여 형상 공간에서 스타일 변환을 수행한다. 우리의 $AdaIN$ 계층은 [6]에서 제안된 스타일 스왑 계층과 유사한 역할을 한다. 스타일 스왑 작업은 시간이 많이 걸리고 메모리도 많이 소모되지만 $AdaIN$ 계층은 IN 계층만큼 간단하여 계산 비용이 거의 들지 않는다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-2.PNG)

> Figure 2. An overview of our style transfer algorithm. We use the first few layers of a fixed VGG-19 network to encode the content and style images. An AdaIN layer is used to perform style transfer in the feature space. A decoder is learned to invert the AdaIN output to the image spaces. We use the same VGG encoder to compute a content loss Lc (Equ. 12) and a style loss Ls (Equ. 13).
>> 그림 2. 스타일 변환 알고리즘의 개요. 우리는 콘텐츠 및 스타일 이미지를 인코딩하기 위해 고정 VGG-19 네트워크의 처음 몇 계층을 사용한다. AdaIN 레이어는 형상 공간에서 스타일 변환을 수행하는 데 사용된다. 디코더는 에이다를 반전시키기 위해 학습된다.영상 공간으로 출력합니다. 우리는 동일한 VGG 인코더를 사용하여 콘텐츠 손실 Lc(Equ. 12)와 스타일 손실 Ls(Equ. 13)를 계산한다.

### $\mathbf{6.\;Experimental\;Setup}$

> Fig. 2 shows an overview of our style transfer network based on the proposed $AdaIN$ layer. Code and pretrained models (in Torch 7 [7]) are available at: [https://github.com/xunhuang1995/AdaIN-style](https://github.com/xunhuang1995/AdaIN-style)
>> 그림 2는 제안된 $Ada에 기반한 스타일 변환 네트워크의 개요를 보여준다.IN$ 레이어. 코드 및 사전 훈련된 모델(Torch 7 [7]의 경우)은 [https://github.com/xunhuang1995/AdaIN-style](https://github.com/xunhuang1995/AdaIN-style)에서 구입할 수 있습니다.

### $\mathbf{6.1.\;Architecture}$

> Our style transfer network T takes a content image c and an arbitrary style image s as inputs, and synthesizes an output image that recombines the content of the former and the style latter. We adopt a simple encoder-decoder architecture, in which the encoder f is fixed to the first few layers (up to relu4 1) of a pre-trained VGG-19 [48]. After encoding the content and style images in feature space, we feed both feature maps to an $AdaIN$ layer that aligns the mean and variance of the content feature maps to those of the style feature maps, producing the target feature maps t:
>> 우리의 스타일 변환 네트워크 T는 콘텐츠 이미지 c와 임의의 스타일 이미지를 입력으로 취하고, 전자와 스타일 후자의 콘텐츠를 재결합하는 출력 이미지를 합성한다. 우리는 인코더 f를 사전 훈련된 VGG-19의 처음 몇 개의 레이어(relu41까지)에 고정하는 간단한 인코더-디코더 아키텍처를 채택한다[48]. 형상 공간에서 내용 및 스타일 이미지를 인코딩한 후, 우리는 두 형상 맵을 $Ada에 공급한다.콘텐츠 피쳐 맵의 평균과 분산을 스타일 피쳐 맵의 평균과 분산으로 정렬하는 IN$ 계층은 대상 피쳐 맵 t:

$$t=AdaIN(f(c), f(s))$$

> A randomly initialized decoder g is trained to map t back to the image space, generating the stylized image T(c,s):
>> 무작위로 초기화된 디코더 g는 t를 이미지 공간에 다시 매핑하여 양식화된 이미지 T(c,s)를 생성하도록 훈련된다.

$$T(c,s)=g(t)$$

> The decoder mostly mirrors the encoder, with all pooling layers replaced by nearest up-sampling to reduce checkerboard effects. We use reflection padding in both f and g to avoid border artifacts. Another important architectural choice is whether the decoder should use instance, batch, or no normalization layers. As discussed in Sec. 4, IN normalizes each sample to a single style while BN normalizes a batch of samples to be centered around a single style. Both are undesirable when we want the decoder to generate images in vastly different styles. Thus, we do not use normalization layers in the decoder. In Sec. 7.1 we will show that IN/BN layers in the decoder indeed hurt performance.
>> 디코더는 대부분 인코더를 미러링하며, 모든 풀링 레이어는 체커보드 효과를 줄이기 위해 가장 가까운 업샘플링으로 대체된다. 테두리 아티팩트를 피하기 위해 f와 g 모두에서 반사 패딩을 사용한다. 또 다른 중요한 아키텍처 선택은 디코더가 인스턴스, 배치 또는 표준화 계층을 사용해야 하는지 여부이다. 4장에서 논의된 바와 같이 IN은 각 샘플을 단일 스타일로 정규화하고 BN은 단일 스타일을 중심으로 샘플 배치를 정규화한다. 둘 다 디코더가 매우 다른 스타일로 이미지를 생성하기를 원할 때 바람직하지 않다. 따라서, 우리는 디코더에서 정규화 계층을 사용하지 않는다. 7.1절에서 우리는 디코더의 IN/BN 레이어가 실제로 성능을 해친다는 것을 보여줄 것이다.

### $\mathbf{6.2.\;Training}$

> We train our network using MS-COCO [36] as content images and a dataset of paintings mostly collected from WikiArt [39] as style images, following the setting of [6]. Each dataset contains roughly 80, 000 training examples. We use the adam optimizer [26] and a batch size of 8 content-style image pairs. During training, we first resize the smallest dimension of both images to 512 while preserving the aspect ratio, then randomly crop regions of size 256 × 256. Since our network is fully convolutional, it can be applied to images of any size during testing. Similar to [51, 11, 52], we use the pre-trained VGG19 [48] to compute the loss function to train the decoder:
>> 우리는 [6]의 설정에 따라 MS-COCO[36]를 콘텐츠 이미지로 사용하고 WikiArt[39]에서 대부분 수집된 그림 데이터 세트를 스타일 이미지로 사용하여 네트워크를 훈련한다. 각 데이터 세트에는 약 80,000개의 교육 예가 포함되어 있다. Adam optimizer [26]와 8개의 콘텐츠 스타일 이미지 쌍의 배치 크기를 사용한다. 훈련 중에 우리는 먼저 가로 세로 비율을 유지하면서 두 이미지의 최소 크기를 512로 조정한 다음 256 × 256 크기의 영역을 무작위로 자른다. 우리의 네트워크는 완전히 컨볼루션이기 때문에 테스트하는 동안 모든 크기의 이미지에 적용할 수 있다. [51, 11, 52]와 유사하게, 우리는 사전 훈련된 VGG19[48]를 사용하여 손실 함수를 계산하여 디코더를 훈련시킨다.

$$L=L_{c}+λL_{s}$$

> which is a weighted combination of the content loss $L_{c}$ and the style loss $L^{s}$ with the style loss weight $\lambda{}$. The content loss is the Euclidean distance between the target features and the features of the output image. We use the $AdaIN$ output t as the content target, instead of the commonly used feature responses of the content image. We find this leads to slightly faster convergence and also aligns with our goal of inverting the $AdaIN$ output t.
>> 이는 스타일 손실 가중치 $\lambda{}$와 함께 내용 손실 $L_{c}$과 스타일 손실 $L^{s}$의 가중 조합이다. 내용 손실은 대상 특징과 출력 이미지의 특징 사이의 유클리드 거리이다. 콘텐츠 이미지의 일반적으로 사용되는 기능 응답 대신 $AdaIN$ 출력 t를 콘텐츠 대상으로 사용합니다. 우리는 이것이 약간 더 빠른 수렴으로 이어지고 $AdaIN$ 출력 t를 반전시키는 우리의 목표와도 일치한다는 것을 발견했다.

$$L_{c}=\vert{}\vert{}f(g(t))−t\vert{}\vert{}_{2}$$

> Since our $AdaIN$ layer only transfers the mean and standard deviation of the style features, our style loss only matches these statistics. Although we find the commonly used Gram matrix loss can produce similar results, we match the IN statistics because it is conceptually cleaner. This style loss has also been explored by Li et al. [33].
>> $AdaIN$ 도면층은 스타일 피쳐의 평균 및 표준 편차만 전송하므로 스타일 손실은 이러한 통계량과만 일치합니다. 일반적으로 사용되는 그램 행렬 손실이 유사한 결과를 낼 수 있다는 것을 발견하지만, 개념적으로 더 깨끗하기 때문에 IN 통계와 일치한다. 이 스타일 손실은 Li 등에 의해서도 탐구되었다. [33].

$$L_{s}=\sum_{i=1}^{L}\vert{}\vert{}µ(φ_{i}(g(t)))−µ(φ_{i}(s))\vert{}\vert{}_{2}+\sum_{i=1}^{L}\vert{}\vert{}σ(φ_{i}(g(t)))−σ(φ_{i}(s))\vert{}\vert{}_{2}$$

> where each $φ_{i}$ denotes a layer in VGG-19 used to compute the style loss. In our experiments we use relu1 1, relu2 1, relu3 1, relu4 1 layers with equal weights.
>> 여기서 각 $φ_{i}$는 스타일 손실을 계산하는 데 사용되는 VGG-19의 레이어를 나타냅니다. 우리의 실험에서 우리는 동일한 가중치를 가진 relu11, relu21, relu31, relu41 레이어를 사용한다.

### $\mathbf{7.\;Results}$

### $\mathbf{7.1.\;Comparison\;with\;other\;methods}$

> In this subsection, we compare our approach with three types of style transfer methods: 1) the flexible but slow optimization-based method [16], 2) the fast feed-forward method restricted to a single style [52], and 3) the flexible patch-based method of medium speed [6]. If not mentioned otherwise, the results of compared methods are obtained by running their code with the default configurations. 1 For [6], we use a pre-trained inverse network provided by the authors. All the test images are of size 512 × 512. 
>> 이 하위 절에서는 우리의 접근 방식을 세 가지 유형의 스타일 변환 방법과 비교한다. 1) 유연하지만 느린 최적화 기반 방법[16], 2) 단일 스타일로 제한된 빠른 피드 포워드 방법[52], 3) 유연한 패치 기반 중간 속도 방법[6]이다. 달리 언급되지 않은 경우, 비교 방법의 결과는 기본 구성으로 코드를 실행하여 얻는다. 1 [6]의 경우, 우리는 저자가 제공한 사전 훈련된 역 네트워크를 사용한다. 모든 테스트 이미지의 크기는 512 × 512입니다.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-3.PNG)

> Figure 3. Quantitative comparison of different methods in terms of style and content loss. Numbers are averaged over 10 style images and 50 content images randomly chosen from our test set.
>> 그림 3. 스타일 및 콘텐츠 손실 측면에서 다양한 방법을 정량적으로 비교합니다. 숫자는 테스트 세트에서 무작위로 선택한 10개의 스타일 이미지와 50개의 콘텐츠 이미지에 걸쳐 평균화된다.

> **Qualitative Examples.** In Fig. 4 we show example style transfer results generated by compared methods. Note that all the test style images are never observed during the training of our model, while the results of [52] are obtained by fitting one network to each test style. Even so, the quality of our stylized images is quite competitive with [52] and [16] for many images (e.g., row 1, 2, 3). In some other cases (e.g., row 5) our method is slightly behind the quality of [52] and [16]. This is not unexpected, as we believe there is a three-way trade-off between speed, flexibility, and qualit $y$. Compared with [6], our method appears to transfer the style more faithfully for most compared images. The last example clearly illustrates a major limitation of [6], which attempts to match each content patch with the closest-matching style patch. However, if most content patches are matched to a few style patches that are not representative of the target style, the style transfer would fail. We thus argue that matching global feature statistics is a more general solution, although in some cases (e.g., row 3) the method of [6] can also produce appealing results. 
>> **퀄리티 예.** 그림 4에서 우리는 비교 방법에 의해 생성된 스타일 변환 결과의 예를 보여준다. [52]의 결과는 각 테스트 스타일에 하나의 네트워크를 적합시켜 얻는 반면, 모델의 훈련 중에는 모든 테스트 스타일 이미지가 관찰되지 않는다. 그럼에도 불구하고 스타일링된 이미지의 품질은 많은 이미지(예: 1, 2, 3행)에 대해 [52] 및 [16]과 상당히 경쟁적이다. 일부 다른 경우(예: 5행)의 경우, 우리의 방법은 [52]와 [16]의 품질보다 약간 뒤떨어진다. 속도, 유연성 및 품질 $y$ 사이에 3가지 균형이 있다고 믿기 때문에 이는 예상 밖의 일이 아니다. [6]과 비교하여, 우리의 방법은 대부분의 비교 이미지에 대해 스타일을 더 충실하게 전송하는 것으로 보인다. 마지막 예에서는 각 컨텐츠 패치를 가장 가까운 스타일 패치와 일치시키려고 하는 [6]의 주요 제한을 명확하게 보여 줍니다. 그러나 대부분의 내용 패치가 대상 스타일을 나타내지 않는 몇 가지 스타일 패치와 일치하면 스타일 변환이 실패합니다. 따라서 일부 경우(예: 3행) [6]의 방법도 매력적인 결과를 낼 수 있지만, 전역 특징 통계를 일치시키는 것이 더 일반적인 해결책이라고 주장한다.

> **Quantitative evaluations.** Does our algorithm trade off some quality for higher speed and flexibility, and if so by how much? To answer this question quantitatively, we compare our approach with the optimization-based method [16] and the fast single-style transfer method [52] in terms of the content and style loss. Because our method uses a style loss based on IN statistics, we also modify the loss function in [16] and [52] accordingly for a fair comparison (their results in Fig. 4 are still obtained with the default Gram matrix loss). The content loss shown here is the same as in [52, 16]. The numbers reported are averaged over 10 style images and 50 content images randomly chosen from the test set of the WikiArt dataset [39] and MS-COCO [36].
>> **퀄리티 평가.** 우리의 알고리즘은 더 빠른 속도와 유연성을 위해 어느 정도의 품질을 교환하는가? 그렇다면 얼마만큼? 이 질문에 정량적으로 답하기 위해, 우리는 콘텐츠 및 스타일 손실 측면에서 우리의 접근 방식을 최적화 기반 방법[16] 및 빠른 단일 스타일 변환 방법[52]과 비교한다. 우리의 방법은 IN 통계를 기반으로 하는 스타일 손실을 사용하기 때문에 공정한 비교를 위해 [16]과 [52]의 손실 함수도 수정한다(그림 4의 결과는 여전히 기본 그램 행렬 손실로 얻어진다). 여기에 표시된 콘텐츠 손실은 [52, 16]과 동일합니다. 보고된 숫자는 WikiArt 데이터 세트[39] 및 MS-COCO[36]의 테스트 세트에서 무작위로 선택된 10개의 스타일 이미지와 50개의 콘텐츠 이미지에 대해 평균화된다.

> As shown in Fig. 3, the average content and style loss of our synthesized images are slightly higher but comparable to the single-style transfer method of Ulyanov et al. [52]. In particular, both our method and [52] obtain a style loss similar to that of [16] between 50 and 100 iterations of optimization. This demonstrates the strong generalization ability of our approach, considering that our network has never seen the test styles during training while each network of [52] is specifically trained on a test style. Also, note that our style loss is much smaller than that of the original content image. 
>> 그림 3에서 보는 바와 같이, 합성된 이미지의 평균 함량 및 스타일 손실은 약간 높지만, Ulyanov 등의 단일 스타일 변환 방식과 비교할 수 있다. [52. 특히, 우리의 방법과 [52] 모두 [16]과 유사한 스타일 손실을 50번과 100번 사이에서 얻는다. 이것은 [52]의 각 네트워크가 테스트 스타일에 대해 특별히 훈련되는 동안 우리의 네트워크가 훈련 중에 테스트 스타일을 본 적이 없다는 점을 고려하여 우리의 접근 방식의 강력한 일반화 능력을 보여준다. 또한 스타일 손실은 원본 콘텐츠 이미지보다 훨씬 작습니다.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-4.PNG)

> Figure 4. Example style transfer results. All the tested content and style images are never observed by our network during training.
>> 그림 4. 스타일 변환 결과 예제입니다. 테스트된 모든 콘텐츠 및 스타일 이미지는 교육 중에 네트워크에서 관찰되지 않는다.

> **Speed analysis.** Most of our computation is spent on content encoding, style encoding, and decoding, each roughly taking one third of the time. In some application scenarios such as video processing, the style image needs to be encoded only once and $AdaIN$ can use the stored style statistics to process all subsequent images. In some other cases (e.g., transferring the same content to different styles), the computation spent on content encoding can be shared.
>> **속도 분석.** 우리 계산의 대부분은 콘텐츠 인코딩, 스타일 인코딩 및 디코딩에 사용되며, 각각은 대략 3분의 1의 시간이 걸린다. 비디오 처리와 같은 일부 응용 프로그램 시나리오에서 스타일 이미지는 한 번만 인코딩되어야 하며 $AdaIN$는 저장된 스타일 통계를 사용하여 모든 후속 이미지를 처리할 수 있습니다. 일부 다른 경우(예: 동일한 콘텐츠를 다른 스타일로 전송하는 경우)에는 콘텐츠 인코딩에 소비되는 계산이 공유될 수 있습니다.

> In Tab. 1 we compare the speed of our method with previous ones [16, 52, 11, 6]. Excluding the time for style encoding, our algorithm runs at 56 and 15 FPS for 256 × 256 and 512 × 512 images respectively, making it possible to process arbitrary user-uploaded styles in real-time. Among algorithms applicable to arbitrary styles, our method is nearly 3 orders of magnitude faster than [16] and 1-2 orders of magnitude faster than [6]. The speed improvement over [6] is particularly significant for images of higher resolution, since the style swap layer in [6] does not scale well to high resolution style images. Moreover, our approach achieves comparable speed to feed-forward methods limited to a few styles [52, 11]. The slightly longer processing time of our method is mainly due to our larger VGG-based network, instead of methodological limitations. With a more efficient architecture, our speed can be further improved.
>> 표 1에서 우리는 우리의 방법의 속도를 이전의 방법과 비교한다[16, 52, 11, 6]. 스타일 인코딩 시간을 제외하고, 우리의 알고리듬은 각각 256 × 256 및 512 × 512 이미지에 대해 56 및 15 FPS로 실행되므로 임의의 사용자 업로드 스타일을 실시간으로 처리할 수 있다. 임의 스타일에 적용할 수 있는 알고리듬 중, 우리의 방법은 [16]보다 거의 3배 빠르고 [6]보다 1-2배 빠르다. [6]의 스타일 스왑 계층은 고해상도 스타일 이미지로 잘 확장되지 않기 때문에, [6]에 대한 속도 향상은 특히 고해상도 이미지에서 중요하다. 또한, 우리의 접근 방식은 몇 가지 스타일로 제한된 피드포워드 방법과 비슷한 속도를 달성한다[52, 11]. 우리 방법의 처리 시간이 약간 길어진 것은 주로 방법론적 한계 대신 더 큰 VGG 기반 네트워크 때문이다. 보다 효율적인 아키텍처를 통해 속도를 더욱 향상시킬 수 있습니다.

![Table 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Table-1.PNG)

> Table 1. Speed comparison (in seconds) for 256 × 256 and 512 × 512 images. Our approach achieves comparable speed to methods limited to a small number styles [52, 11], while being much faster than other existing algorithms applicable to arbitrary styles [16, 6]. We show the processing time both excluding and including (in parenthesis) the style encoding procedure. Results are obtained with a Pascal Titan X GPU and averaged over 100 images.
>> 표 1. 256 × 256 및 512 × 512 이미지에 대한 속도 비교(초) 우리의 접근 방식은 임의의 스타일에 적용되는 다른 기존 알고리듬보다 훨씬 빠르면서도 작은 숫자 스타일[52, 11]로 제한된 방법과 비슷한 속도를 달성한다[16, 6]. 우리는 스타일 인코딩 절차를 제외한 처리 시간과 포함(괄호 안에) 처리 시간을 보여준다. 결과는 Pascal Titan X GPU로 얻으며 평균 100개 이상의 이미지를 얻었다.

### $\mathbf{7.2.\;Additional\;experiments.}$

> In this subsection, we conduct experiments to justify our important architectural choices. We denote our approach described in Sec. 6 as Enc-AdaIN-Dec. We experiment with a model named Enc-Concat-Dec that replaces $AdaIN$ with concatenation, which is a natural baseline strategy to combine information from the content and style images. In addition, we run models with BN/IN layers in the decoder, denoted as Enc-AdaIN-BNDec and Enc-AdaIN-INDec respectivel $y$. Other training settings are kept the same. 
>> 이 하위 절에서는 중요한 아키텍처 선택을 정당화하기 위한 실험을 수행한다. 우리는 6절에 기술된 우리의 접근 방식을 Enc-AdaIN-Dec로 나타낸다. 우리는 $AdaIN$ 를 연결로 대체하는 Enc-Concat-Dec라는 모델을 실험하는데, 이는 콘텐츠와 스타일 이미지의 정보를 결합하는 자연스러운 기준선 전략이다. 또한, 우리는 각각 Enc-AdaIN-BNDec와 Enc-AdaIN-INDec로 표시된 디코더에서 BN/IN 레이어가 있는 모델을 실행한다. 다른 교육 설정은 동일하게 유지됩니다.

> In Fig. 5 and 6, we show examples and training curves of the compared methods. In the image generated by the EncConcat-Dec baseline (Fig. 5 (d)), the object contours of the style image can be clearly observed, suggesting that the network fails to disentangle the style information from the content of the style image. This is also consistent with Fig. 6, where Enc-Concat-Dec can reach low style loss but fail to decrease the content loss. Models with BN/IN layers also obtain qualitatively worse results and consistently higher losses. The results with IN layers are especially poor. This once again verifies our claim that IN layers tend to normalize the output to a single style and thus should be avoided when we want to generate images in different styles.
>> 그림 5와 6에서는 비교한 방법의 예와 훈련 곡선을 보여준다. EncConcat-Dec 기준선에 의해 생성된 이미지(그림 5(d))에서 스타일 이미지의 객체 윤곽선을 명확하게 관찰할 수 있으며, 이는 네트워크가 스타일 이미지의 콘텐츠에서 스타일 정보를 분리하지 못함을 시사한다. 이는 Enc-Concat-Dec이 낮은 스타일 손실에는 도달할 수 있지만 내용물 손실은 감소시키지 못하는 그림 6과도 일치한다. BN/IN 층을 가진 모델은 질적으로 더 나쁜 결과와 지속적으로 더 높은 손실을 얻는다. IN 레이어를 사용한 결과는 특히 좋지 않습니다. 이는 IN 레이어가 출력을 단일 스타일로 정규화하는 경향이 있으므로 다른 스타일로 이미지를 생성하려는 경우 피해야 한다는 우리의 주장을 다시 한 번 검증한다.

### $\mathbf{7.3.\;Runtime\;controls}$

> To further highlight the flexibility of our method, we show that our style transfer network allows users to control the degree of stylization, interpolate between different styles, transfer styles while preserving colors, and use different styles in different spatial regions. Note that all these controls are only applied at runtime using the same network, without any modification to the training procedure. 
>> 우리 방법의 유연성을 더욱 강조하기 위해, 우리는 스타일 변환 네트워크가 사용자가 스타일화의 정도를 제어하고, 다른 스타일 간에 보간하고, 색상을 보존하면서 스타일을 전송하고, 다른 공간 영역에서 다른 스타일을 사용할 수 있음을 보여준다. 이러한 모든 제어는 교육 절차를 수정하지 않고 동일한 네트워크를 사용하는 런타임에만 적용됩니다.

> **Content-style trade-off.** The degree of style transfer can be controlled during training by adjusting the style weight $\lambda{}$ in Eqa. 11. In addition, our method allows content-style trade-off at test time by interpolating between feature maps that are fed to the decoder. Note that this is equivalent to interpolating between the affine parameters of AdaIN.
>> > ** 콘텐츠 스타일 절충.** 스타일 변환의 정도는 Eqa.11에서 스타일 가중치 $\lambda{}$를 조정하여 훈련 중에 제어할 수 있다. 또한, 우리의 방법은 디코더에 공급되는 기능 맵 간에 보간하여 테스트 시 콘텐츠 스타일의 트레이드오프를 허용한다. 이것은 AdaIN의 아핀 파라미터들 사이에서 보간하는 것과 같다.

$$T(c,s,α)=g((1−α)f(c)+αAdaIN(f(c),f(s)))$$

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-5.PNG)

> Figure 5. Comparison with baselines. AdaIN is much more effective than concatenation in fusing the content and style information. Also, it is important not to use BN or IN layers in the decoder.
>> 그림 5. 기준선과의 비교. AdaIN은 콘텐츠와 스타일 정보를 융합하는 데 연결보다 훨씬 효과적이다. 또한, 디코더에서 BN 또는 IN 레이어를 사용하지 않는 것이 중요하다.

![Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-6.PNG)

> Figure 6. Training curves of style and content loss.
>> 그림 6. 스타일 및 컨텐츠 손실의 교육 곡선.

> The network tries to faithfully reconstruct the content image when $α=0$, and to synthesize the most stylized image when α=1. As shown in Fig. 7, a smooth transition between content-similarity and style-similarity can be observed by changing α from 0 to 1. 
>> 네트워크는 $param=0$일 때 콘텐츠 이미지를 충실하게 재구성하고, $α=0$일 때 가장 양식화된 이미지를 합성하려고 한다. 이미지 7에서 보는 바와 같이, α를 0에서 1로 변화시킴으로써 content-similarity 과 style-similarity의 원활한 전이를 관찰할 수 있다.

> **Style interpolation.** To interpolate between a set of $K$ style images $s_{1}, s_{2}, ..., s_{K}$ with corresponding weights $w_{1}, w_{2}, ..., w_{K}$ such that $\sum_{k=1}^{K}w_{k}=1$, we similarly interpolate between featu과e maps (results shown in Fig. 8): 
>> **스타일 보간.** $\sum_{k=1}^{K}w_{k}=1$이 되도록 해당 가중치 $w_{1}, w_{2}, ..., w_{K}$를 가진 $K$ 스타일 이미지 $s_{1}, s_{2}, ..., s_{K}$ 세트 사이에 보간하기 위해, 우리는 featee 맵(그림에 표시된 그림) 간에 유사하게 보간한다. 8):

$$T(c,s_{1,2},...K, w_{1,2},...K)=g(\sum{k=1}{K}w_{k}AdaIN(f(c), f(s_{k})))$$

> **Spatial and color control.** Gatys et al. [17] recently introduced user controls over color information and spatial locations of style transfer, which can be easily incorporated into our framework. To preserve the color of the content image, we first match the color distribution of the style image to that of the content image (similar to [17]), then perform a normal style transfer using the color-aligned style image as the style input. Examples results are shown in Fig. 9.
>> **공간 및 색상 제어.** 게이티 외 [17] 최근 스타일 변환의 색상 정보 및 공간 위치에 대한 사용자 제어가 도입되었으며, 이는 우리의 프레임워크에 쉽게 통합될 수 있다. 콘텐츠 이미지의 색상을 보존하기 위해 먼저 스타일 이미지의 색상 분포를 콘텐츠 이미지의 색상 분포([17]와 유사)와 일치시킨 다음 색상 정렬된 스타일 이미지를 스타일 입력으로 사용하여 정상적인 스타일 변환을 수행한다. 예시적인 결과는 그림 9와 같다.

![FIgure 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-7.PNG)

> Figure 7. Content-style trade-off. At runtime, we can control the balance between content and style by changing the weight α in Equ. 14.
>> 그림 7. 콘텐츠 유형 균형 조정. 런타임에, 우리는 14번 방정식의 가중치 in를 변경함으로써 내용과 스타일 사이의 균형을 조절할 수 있다.

![Figure 8](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-8.PNG)

> Figure 8. Style interpolation. By feeding the decoder with a convex combination of feature maps transferred to different styles via AdaIN (Equ. 15), we can interpolate between arbitrary new styles
>> 그림 8. 스타일 보간. AdaIN를 통해 다른 스타일로 전송된 피처 맵의 볼록한 조합을 디코더에 공급함으로써(Equ. 15) 임의의 새로운 스타일 간에 보간할 수 있다.

> In Fig. 10 we demonstrate that our method can transfer different regions of the content image to different styles. This is achieved by performing $AdaIN$ separately to different regions in the content feature maps using statistics from different style inputs, similar to [4, 17] but in a completely feed-forward manner. While our decoder is only trained on inputs with homogeneous styles, it generalizes naturally to inputs in which different regions have different styles.
>> 그림 10에서 우리는 우리의 방법이 콘텐츠 이미지의 다른 영역을 다른 스타일로 전송할 수 있다는 것을 보여준다. $Ada를 수행하면 됩니다.IN$는 콘텐츠의 다른 영역에 별도로 [4, 17]과 유사하지만 완전히 피드포워드 방식으로 다른 스타일 입력의 통계를 사용하여 매핑한다. 우리의 디코더는 동종 스타일을 가진 입력에만 훈련되지만, 지역마다 스타일이 다른 입력에 자연스럽게 일반화된다.

### $\mathbf{8.\;Discussion\;and\;Conclusion}$

> In this paper, we present a simple adaptive instance normalization (AdaIN) layer that for the first time enables arbitrary style transfer in real-time. Beyond the fascinating applications, we believe this work also sheds light on our understanding of deep image representations in general.
>> 본 논문에서는 처음으로 임의의 스타일 변환을 실시간으로 가능하게 하는 간단한 적응형 인스턴스 정규화 (AdaIN) 계층을 제시한다. 매력적인 응용 프로그램 외에도, 우리는 이 작업이 전반적으로 심층 이미지 표현에 대한 우리의 이해를 조명한다고 믿는다.

> It is interesting to consider the conceptual differences between our approach and previous neural style transfer methods based on feature statistics. Gatys et al. [16] employ an optimization process to manipulate pixel values to match feature statistics. The optimization process is replaced by feed-forward neural networks in [24, 51, 52]. Still, the network is trained to modify pixel values to indirectly match feature statistics. We adopt a very different approach that directly aligns statistics in the feature space in one shot, then inverts the features back to the pixel space.
>> 특징 통계를 기반으로 우리의 접근 방식과 이전의 신경 스타일 변환 방법 사이의 개념적 차이를 고려하는 것은 흥미롭다. 게이티 외 [16] 최적화 프로세스를 사용하여 형상 통계와 일치하도록 픽셀 값을 조작한다. 최적화 프로세스는 [24, 51, 52]에서 피드포워드 신경망으로 대체된다. 그러나 네트워크는 기능 통계와 간접적으로 일치하도록 픽셀 값을 수정하도록 훈련된다. 우리는 한 번의 촬영으로 특징 공간의 통계를 직접 정렬한 다음 특징을 픽셀 공간으로 반전시키는 매우 다른 접근 방식을 채택한다.

![Figure 9](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-9.PNG)

> Figure 9. Color control. Left: content and style images. Right: color-preserved style transfer result.
>> 그림 9. 색 조절. 왼쪽: 컨텐츠 및 스타일 이미지. 오른쪽: 색상 보존 스타일 변환 결과.

![Figure 10](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-07-(GAN)Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation/Figure-10.PNG)

> Figure 10. Spatial control. Left: content image. Middle: two style images with corresponding masks. Right: style transfer result.
>> 그림 10. 공간 제어. 왼쪽: 내용 이미지. Middle: 해당하는 마스크가 있는 두 개의 스타일 이미지. 오른쪽: 스타일 변환 결과입니다.

> Given the simplicity of our approach, we believe there is still substantial room for improvement. In future works we plan to explore more advanced network architectures such as the residual architecture [24] or an architecture with additional skip connections from the encoder [23]. We also plan to investigate more complicated training schemes like the incremental training [32]. Moreover, our $AdaIN$ layer only aligns the most basic feature statistics (mean and variance). It is possible that replacing $AdaIN$ with correlation alignment [49] or histogram matching [54] could further improve quality by transferring higher-order statistics. Another interesting direction is to apply $AdaIN$ to texture synthesis.
>> 우리의 접근 방식이 단순하다는 점을 고려할 때, 우리는 여전히 상당한 개선의 여지가 있다고 믿는다. 향후 작업에서는 잔류 아키텍처[24] 또는 인코더에서 추가 건너뛰기 연결이 있는 아키텍처[23]와 같은 보다 고급 네트워크 아키텍처를 탐색할 계획이다. 또한 증분 훈련과 같은 더 복잡한 훈련 체계를 조사할 계획이다[32]. 또한, 우리의 $AdaIN$ 계층은 가장 기본적인 특징 통계(평균과 분산)만 정렬한다. $AdaIN$를 상관 정렬 [49] 또는 히스토그램 일치 [54]로 대체하면 고차 통계량을 전송하여 품질을 더욱 향상시킬 수 있습니다. 또 다른 흥미로운 방향은 텍스처 합성에 $AdaIN$를 적용하는 것이다.

### $\mathbf{Acknowledgments}$

> We would like to thank Andreas Veit for helpful discussions. This work was supported in part by a Google Focused Research Award, AWS Cloud Credits for Research and a Facebook equipment donation.
>> 우리는 도움이 되는 논의에 대해 Andreas Veit에게 감사드리고 싶습니다. 이 연구는 구글 포커스 리서치 어워드, AWS 클라우드 크레디트 포 리서치, 페이스북 장비 기부에 의해 부분적으로 지원되었다.

---

### $\mathbf{References}$

<a href="#footnote_1_2" name="footnote_1_1">[1]</a> M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein gan. arXiv preprint arXiv:1701.07875, 2017. 2

<a href="#footnote_2_2" name="footnote_2_1">[2]</a> J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 2

<a href="#footnote_3_2" name="footnote_3_1">[3]</a> K. Bousmalis, N. Silberman, D. Dohan, D. Erhan, and D. Krishnan. Unsupervised pixel-level domain adaptation with generative adversarial networks. arXiv preprint arXiv:1612.05424, 2016. 2

<a href="#footnote_4_2" name="footnote_4_1">[4]</a> A. J. Champandard. Semantic style transfer and turning two-bit doodles into fine artworks. arXiv preprint arXiv:1603.01768, 2016. 8

<a href="#footnote_5_2" name="footnote_5_1">[5]</a> D. Chen, L. Yuan, J. Liao, N. Yu, and G. Hua. Stylebank: An explicit representation for neural image style transfer. In CVPR, 2017. 1

<a href="#footnote_6_2" name="footnote_6_1">[6]</a> T. Q. Chen and M. Schmidt. Fast patch-based style transfer of arbitrary style. arXiv preprint arXiv:1612.04337, 2016. 1, 2, 4, 5, 6, 7

<a href="#footnote_7_2" name="footnote_7_1">[7]</a> R. Collobert, K. Kavukcuoglu, and C. Farabet. Torch7: A matlab-like environment for machine learning. In NIPS Workshop, 2011. 4

<a href="#footnote_8_2" name="footnote_8_1">[8]</a> T. Cooijmans, N. Ballas, C. Laurent, C¸ . Gulc¸ehre, and ¨ A. Courville. Recurrent batch normalization. In ICLR, 2017. 2

<a href="#footnote_9_2" name="footnote_9_1">[9]</a> E. L. Denton, S. Chintala, R. Fergus, et al. Deep generative image models using a laplacian pyramid of adversarial networks. In NIPS, 2015. 2

<a href="#footnote_10_2" name="footnote_10_1">[10]</a> A. Dosovitskiy and T. Brox. Inverting visual representations with convolutional networks. In CVPR, 2016. 4

<a href="#footnote_11_2" name="footnote_11_1">[11]</a> V. Dumoulin, J. Shlens, and M. Kudlur. A learned representation for artistic style. In ICLR, 2017. 1, 2, 3, 5, 6, 7 

<a href="#footnote_12_2" name="footnote_12_1">[12]</a> A. A. Efros and W. T. Freeman. Image quilting for texture synthesis and transfer. In SIGGRAPH, 2001. 1

<a href="#footnote_13_2" name="footnote_13_1">[13]</a> A. A. Efros and T. K. Leung. Texture synthesis by nonparametric sampling. In ICCV, 1999. 1

<a href="#footnote_14_2" name="footnote_14_1">[14]</a> M. Elad and P. Milanfar. Style-transfer via texture-synthesis. arXiv preprint arXiv:1609.03057, 2016. 1

<a href="#footnote_15_2" name="footnote_15_1">[15]</a> O. Frigo, N. Sabater, J. Delon, and P. Hellier. Split and match: example-based adaptive patch sampling for unsupervised style transfer. In CVPR, 2016. 1

<a href="#footnote_16_2" name="footnote_16_1">[16]</a> L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. In CVPR, 2016. 1, 2, 3, 5, 6, 7, 8

<a href="#footnote_17_2" name="footnote_17_1">[17]</a> L. A. Gatys, A. S. Ecker, M. Bethge, A. Hertzmann, and E. Shechtman. Controlling perceptual factors in neural style transfer. In CVPR, 2017. 1, 7, 8

<a href="#footnote_18_2" name="footnote_18_1">[18]</a> I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NIPS, 2014. 2 

<a href="#footnote_19_2" name="footnote_19_1">[19]</a> D. J. Heeger and J. R. Bergen. Pyramid-based texture analysis/synthesis. In SIGGRAPH, 1995. 1

<a href="#footnote_20_2" name="footnote_20_1">[20]</a> X. Huang, Y. Li, O. Poursaeed, J. Hopcroft, and S. Belongie. Stacked generative adversarial networks. In CVPR, 2017. 2

<a href="#footnote_21_2" name="footnote_21_1">[21]</a> S. Ioffe. Batch renormalization: Towards reducing minibatch dependence in batch-normalized models. arXiv preprint arXiv:1702.03275, 2017. 2

<a href="#footnote_22_2" name="footnote_22_1">[22]</a> S. Ioffe and C. Szeged$y$. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In JMLR, 2015. 2

<a href="#footnote_23_2" name="footnote_23_1">[23]</a> P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017. 2, 8

<a href="#footnote_24_2" name="footnote_24_1">[24]</a> J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016. 1, 2, 3, 8

<a href="#footnote_25_2" name="footnote_25_1">[25]</a> T. Kim, M. Cha, H. Kim, J. Lee, and J. Kim. Learning to discover cross-domain relations with generative adversarial networks. arXiv preprint arXiv:1703.05192, 2017. 2

<a href="#footnote_26_2" name="footnote_26_1">[26]</a> D. Kingma and J. Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 5

<a href="#footnote_27_2" name="footnote_27_1">[27]</a> D. P. Kingma and M. Welling. Auto-encoding variational bayes. In ICLR, 2014. 2

<a href="#footnote_28_2" name="footnote_28_1">[28]</a> J. E. Kyprianidis, J. Collomosse, T. Wang, and T. Isenberg. State of the” art: A taxonomy of artistic stylization techniques for images and video. TVCG, 2013. 1

<a href="#footnote_29_2" name="footnote_29_1">[29]</a> C. Laurent, G. Pereyra, P. Brakel, Y. Zhang, and Y. Bengio. Batch normalized recurrent neural networks. In ICASSP, 2016. 2

<a href="#footnote_30_2" name="footnote_30_1">[30]</a> C. Li and M. Wand. Combining markov random fields and convolutional neural networks for image synthesis. In CVPR, 2016. 1, 2, 3

<a href="#footnote_31_2" name="footnote_31_1">[31]</a> C. Li and M. Wand. Precomputed real-time texture synthesis with markovian generative adversarial networks. In ECCV, 2016. 1, 2

<a href="#footnote_32_2" name="footnote_32_1">[32]</a> Y. Li, C. Fang, J. Yang, Z. Wang, X. Lu, and M.-H. Yang. Diversified texture synthesis with feed-forward networks. In CVPR, 2017. 1, 2, 8

<a href="#footnote_33_2" name="footnote_33_1">[33]</a> Y. Li, N. Wang, J. Liu, and X. Hou. Demystifying neural style transfer. arXiv preprint arXiv:1701.01036, 2017. 1, 2, 3, 5

<a href="#footnote_34_2" name="footnote_34_1">[34]</a> Y. Li, N. Wang, J. Shi, J. Liu, and X. Hou. Revisiting batch normalization for practical domain adaptation. arXiv preprint arXiv:1603.04779, 2016. 2

<a href="#footnote_35_2" name="footnote_35_1">[35]</a> Q. Liao, K. Kawaguchi, and T. Poggio. Streaming normalization: Towards simpler and more biologically-plausible normalizations for online and recurrent learning. arXiv preprint arXiv:1610.06160, 2016. 2

<a href="#footnote_36_2" name="footnote_36_1">[36]</a> T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Com- ´ mon objects in context. In ECCV, 2014. 3, 5 

<a href="#footnote_37_2" name="footnote_37_1">[37]</a> M.-Y. Liu, T. Breuel, and J. Kautz. Unsupervised image-to-image translation networks. arXiv preprint arXiv:1703.00848, 2017. 2

<a href="#footnote_38_2" name="footnote_38_1">[38]</a> M.-Y. Liu and O. Tuzel. Coupled generative adversarial networks. In NIPS, 2016. 2

<a href="#footnote_39_2" name="footnote_39_1">[39]</a> K. Nichol. Painter by numbers, wikiart. https://www.kaggle.com/c/painter-by-numbers, 2016. 5 

<a href="#footnote_40_2" name="footnote_40_1">[40]</a> A. v. d. Oord, N. Kalchbrenner, and K. Kavukcuoglu. Pixel recurrent neural networks. In ICML, 2016. 2 

<a href="#footnote_41_2" name="footnote_41_1">[41]</a> X. Peng and K. Saenko. Synthetic to real adaptation with deep generative correlation alignment networks. arXivpreprint arXiv:1701.05524, 2017. 2

<a href="#footnote_42_2" name="footnote_42_1">[42]</a> A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR, 2016. 2

<a href="#footnote_43_2" name="footnote_43_1">[43]</a> S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee. Generative adversarial text to image synthesis. In ICML, 2016. 2

<a href="#footnote_44_2" name="footnote_44_1">[44]</a> M. Ren, R. Liao, R. Urtasun, F. H. Sinz, and R. S. Zemel. Normalizing the normalizers: Comparing and extending network normalization schemes. In ICLR, 2017. 2

<a href="#footnote_45_2" name="footnote_45_1">[45]</a> M. Ruder, A. Dosovitskiy, and T. Brox. Artistic style transfer for videos. In GCPR, 2016. 1

<a href="#footnote_46_2" name="footnote_46_1">[46]</a> T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training gans. In NIPS, 2016. 2

<a href="#footnote_47_2" name="footnote_47_1">[47]</a> T. Salimans and D. P. Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In NIPS, 2016. 2

<a href="#footnote_48_2" name="footnote_48_1">[48]</a> K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015. 4, 5

<a href="#footnote_49_2" name="footnote_49_1">[49]</a> B. Sun, J. Feng, and K. Saenko. Return of frustratingly easy domain adaptation. In AAAI, 2016. 8

<a href="#footnote_50_2" name="footnote_50_1">[50]</a> Y. Taigman, A. Polyak, and L. Wolf. Unsupervised crossdomain image generation. In ICLR, 2017. 2

<a href="#footnote_51_2" name="footnote_51_1">[51]</a> D. Ulyanov, V. Lebedev, A. Vedaldi, and V. Lempitsk$y$. Texture networks: Feed-forward synthesis of textures and stylized images. In ICML, 2016. 1, 2, 4, 5, 8

<a href="#footnote_52_2" name="footnote_52_1">[52]</a> D. Ulyanov, A. Vedaldi, and V. Lempitsk$y$. Improved texture networks: Maximizing quality and diversity in feed-forward stylization and texture synthesis. In CVPR, 2017. 1, 2, 3, 5, 6, 7, 8

<a href="#footnote_53_2" name="footnote_53_1">[53]</a> X. Wang, G. Oxholm, D. Zhang, and Y.-F. Wang. Multimodal transfer: A hierarchical deep convolutional neural network for fast artistic style transfer. arXiv preprint arXiv:1612.01895, 2016. 2

<a href="#footnote_54_2" name="footnote_54_1">[54]</a> P. Wilmot, E. Risser, and C. Barnes. Stable and controllable neural texture synthesis and style transfer using histogram losses. arXiv preprint arXiv:1701.08893, 2017. 2, 8

<a href="#footnote_55_2" name="footnote_55_1">[55]</a> H. Zhang and K. Dana. Multi-style generative network for real-time transfer. arXiv preprint arXiv:1703.06953, 2017. 1




 