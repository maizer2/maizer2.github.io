---
layout: post 
title: "(Diffusion)DiffFashion논문 분석"
categories: [1. Computer Engineering]
tags: [1.2.2.1. Computer Vision, 1.7. Paper Review]
---

### [CV Paper List](https://maizer2.github.io/1.%20computer%20engineering/2023/02/01/paper-of-diffusion.html)

## Abstract

DiffFashion논문은 Diffusion을 사용한 style transfer 논문이다.

style transfer는 content와 style를 입력으로 받아 content의 구조는 유지하면서 style의 스타일을 전송하는(입히는) 연구이다.

다음 DiffFashion Fig. 1.을 통해 style transfer은 무엇인지에 대해 알 수 있다.

![Figure-1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-1.PNG)

<center>Fig. 1.</center>

그림에서 Appearance와 Bag은 논문에서 clothing image[content image]와 reference appearance image[style image]로 설명하고 있다.

전통적인 style transfer논문에서는 content와 style로 표현하지만 본 논문의 저자는 appearance와 clothing으로 표현한다.

저자는 Diffusion모델의 style transfer 과정에서 clothing[content] image와 appearance[style] image들의 모양이 크게 차이 날 경우 reverse diffusion process과정에서 clothing[content]이미지의 구조를 현실적으로 유지하는 것이 어렵다고 말하였다.

이 문제를 해결하기 위해서 저자는 본 논문의 모델을 소개하는데 모델의 기능은 다음과 같다.

* Novel diffusion model-based unsupervised structure-aware method
    * 조건화된 labels을 사용하여 자동으로 semantic masks를 만들고, 이를 통해 clothing[content]이미지를 분리한다.
    * 앞서 얻은 masks를 diffsuion의 denosing[reverse diffusion process]과정에서 clothing[content]의 구조를 보존하기 위해 가이드[guidance]로 사용된다.
    * ~~appearance[style]과 clothing/structure[content] guidance는 pre-trained vision Transformer(ViT)를 사용한다.~~

## I. Introduction

논문 저자는 fashion style transfer에서 GAN의 문제점은 appearance[style]이미지를 transfer할 때 clothing[content]이미지의 도메인이 아닌 이미지를 transfer할 때 구조와 외관을 제어하기 어렵다고 말하였다.

하지만 최근 크게 주목받고 있는 diffusion model을 사용한 style transfer논문 [15]와 [16]은 GAN에서 부족했던 구조와 외관을 모두 고려하였다.

논문 [[15]](https://maizer2.github.io/1.%20computer%20engineering/2023/03/15/(diffusion)DiffuseIT.html)는 본 논문의 base가 되는 논문으로써, Diffusion model과 special structural appearance loss를 사용하여 비슷한 도메인(객체)들간의 appearance transfer의 우수한 성능을 보여주었다.

하지만 본 논문은 위 논문의 다음과 같은 두가지 문제가 있다고 설명하였다.

1. Pretrained ViT를 사용하여 ,비슷한 도메인(객체)을 기반으로 image transfer를 적용해, 본 논문이 하고자하는 의류 이미지에 적용하기에는 부적적하고, 위 논문을 의류 이미지에 적용할 경우 비현실적인 패턴과 외관을 전단하고, 비의류 이미지를 사용하여(본 논문이 해양생물을 사용하는 것과 같이) tranfer할 경우 거의 불가능하다고 본 논문은 제안하였다.
2. image tranfer mothod들은 훈련에 있어 결과의 대상이 되는 샘플이 대량으로 필요한데, unsupervised learning 특성상 라벨이 없는 훈련이기에 inference시 입력 이미지의 structure 정보를 소실(쉽게말해 입력 이미지 도메인이 없어 결과물이 입력과 다른 모양이 출력될 수 있다)되는 경우가 많다.

## II. Related work

### A. Fashion Design

AI에서 Fashion Design method는 주어진 Fashion 객체에 새로운 style로 변환하는것을 목표로합니다.

위 methods는 GAN을 사용하여 연구되었다. 

* Design Inspiration from Generative Networks[[3]](https://arxiv.org/pdf/1804.00921.pdf)
    * 의류 content[shape]를 유지하고 style latent vector를 통해 의상에 style을 transfer한다.

* FashionGAN: Display your fashion design using ConditionalGenerative Adversarial Nets[[8]](https://onlinelibrary.wiley.com/doi/epdf/10.1111/cgf.13552)
    * sketch 이미지에 Texture 이미지의 style을 transfer한다.

* Toward Intelligent Fashion Design: A Texture and Shape Disentangled Generative Adversarial Network [[2]](https://dl.acm.org/doi/pdf/10.1145/3567596)
    * StyleGAN v2의 patch-based style transfer한다.

하지만 이들은 texture reference images를 사용할 수 없고, 훈련 데이터셋에 존재하는 style에서 샘플을 생성하는 것으로 제한된다고 하였다.

### B. GAN-based Image Transfer

Image-to-Image translation은 한 도메인의 이미지를 다른 도메인의 이미지로 변환하는 인공지능 기술입니다. 이 과정에서 paired와 unpaired 방법이 있습니다. 

* Paired Image-to-Image translation:
    * Paired 방식은 각 입력 이미지에 대응하는 정답 이미지(ground truth)가 함께 제공되는 학습 데이터셋을 사용합니다. 이러한 쌍(pair)은 입력 이미지와 목표 출력 이미지 간의 직접적인 관계를 학습하는 데 도움이 됩니다. 이 방식은 정확한 변환을 학습할 수 있지만, 쌍을 이룬 데이터셋을 구하기 어려울 수 있습니다.
    * 예를 들어, 흑백 사진을 컬러 사진으로 변환하는 경우, paired 방식에서는 같은 사진의 흑백 버전과 컬러 버전이 쌍으로 이루어진 데이터셋을 사용하여 모델을 학습시킵니다.
    * 대표적인 알고리즘으로 Pix2Pix가 있습니다.

* Unpaired Image-to-Image translation:
    * Unpaired 방식은 입력 이미지와 출력 이미지가 쌍으로 제공되지 않는 데이터셋을 사용합니다. 대신 두 도메인에서 각각 샘플링된 이미지들로 구성된 데이터셋이 제공됩니다. 이 방식은 쌍을 이룬 데이터를 구하기 어려운 경우에 유용하지만, 일반적으로 정확성이 paired 방식보다 떨어질 수 있습니다.
    * 예를 들어, 여름 풍경 사진을 겨울 풍경으로 변환하는 경우, unpaired 방식에서는 여름 풍경 사진들과 겨울 풍경 사진들이 각각 모인 데이터셋을 사용하여 모델을 학습시킵니다.
    * 대표적인 알고리즘으로 CycleGAN이 있습니다.

요약하면, paired 방식은 정확한 변환을 학습하기 위해 입력과 출력 이미지가 직접 연결된 쌍을 사용하며, unpaired 방식은 두 도메인의 이미지 샘플들만 사용하여 학습합니다.

본 논문에서는 두 방식 모두 대량의 학습 데이터가 필요하고, GAN의 인코딩 구조상 외관의 정보를 분리하기 어려우며, 마지막으로 두 도메인 간의 격차가 너무 큰 경우, 결과물이 정상적으로 변환되지 않을 수 있으며 원래 도메인의 정보가 소실될 수 있다고 하였습니다.

### C. Diffusion Model-based Image transfer

Denoising Diffusion Probabilistic Models (DDPMs)는 generative model이며, image-to-image translation 분야에서 GAN을 뛰어넘는 결과를 보여준다.

* Palette: Image-to-image diffusion models[[18]](https://arxiv.org/abs/2111.05826)
    * 처음으로 diffusion을 image-to-image translation에 적용하였고, 컬러화, 인페인팅 및 기타 작업에서 좋은 결과를 얻었습니다.
    * 그러나 이 방법은 label이 필요하므로, unsupervised learning에는 적용이 불가능합니다.

* Diffusion-based image translation using disentangled style and content representation[[15]](https://arxiv.org/abs/2209.15264)
    * Transformer인 DINO-ViT[[16]](https://arxiv.org/abs/2201.00424)의 가이드를 사용하여 현실성을 크게 향상시켰습니다.

하지만 unsupervised learning에 적용하기에는 여전히 부족하다고 하였다.

### D. Neural Style Transfer (NST)

ANN과 CNN 그외의 방법을 사용하는 NST 방식에서는 크게 두가지 유형의 접근 방식이 존재한다.

1. Statistical methods
    * 유명한 style transfer 논문인 [[27]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), [[28]](https://arxiv.org/pdf/1703.06868.pdf)는 훈련 데이터셋의 통계적 공간에서 sampling을 통해 style을 얻게된다.

2. Non-parametric methods
    * Non-parametric이란 비모수적 방법이라 하며, 모수 방법은 데이터 분포의 모수(Parameter)를 가정하고 이를 기반으로 모델링을 수행하는 반면, 비모수 방법은 데이터의 분포에 대한 가정 없이 데이터를 직접 이용하여 분석합니다.
    * 입력 데이터를 직접 이용해서 분석함으로서 입력 이미지의 구조와 스타일 이미지의 텍스처를 고려하여 새로운 이미지를 생성하므로, 모수 방법보다 스타일 전송이 자연스럽게 보이도록 도와줍니다.
    * Non-parametric의 대표적인 method는 이미지 퀼트(Image Quilting) 방식이다. 
        * 이 방법은 스타일 이미지로부터 작은 패치를 추출하고, 원본 이미지의 각 부분에 해당 패치를 재구성하여 스타일이 전송된 결과를 생성합니다. 이 프로세스는 높은 해상도의 결과물을 생성할 수 있지만, 명확한 패치 경계가 있는 경우 결과물이 불완전하게 보일 수 있습니다.
        * PatchGAN 방식이 대표적이다.

대부분 두 방법을 통해 style trasfer한 후 CNN을 통해 마무리 하는데, NST방식은 예술적 스타일 전송에 잘 작동하지만, content/style 분리과정에는 적합하지 않다.

본 논문은 clothing[content] image와 appearance[style] image간의 구체적으로 관련된 객체가 없다고 가정한다.

## III. PRELIMINARY OF DENOISING DIFFUSION PROBABILISTIC MODEL

* Diffusion probabilistic models란?
    * Diffusion 모델은 열역학에서 확산의 수학적 과정을 생성모델에 적용한 인공지능 모델이다.
    * 기체가 확산하는 매우 작은 시간에서, 원자는 정규분포를 따르는 거리만큼 퍼지게 됩니다.
    * 반대로 확산을 반대로 되돌릴 때도, 매우 작은 시간에서 정규분포를 따르는 거리만큼 되돌리게되면 확산 이전의 상태로 되돌릴 수 있다라는 과학적 근거에서 바탕되었다.
    * 앞에서 설명한 확산 과정을 diffusion forward process라고 하고, 역확산 과정을 diffusion reverse process라고 한다.

### Forward process

* Input data $x$에 점차적(마르코브 체인)으로, $t = 1, 2, \cdots, T$까지, 정규분포(가우시안 분포)를 따르는 noise를 추가한 $x_{t}$를 sampling합니다.
* Step $t-1$의 $x_{t-1}$에 노이즈를 추가하여 next step $t$의 $x_{t}$를 얻는 과정을 식으로 작성하면 다음과 같다.

$$ q(x_{t}\vert{}x_{t-1})=N(x_{t};\sqrt{1-\beta{}_{t}}x_{t-1},\beta{}_{t}I)$$

* 위 표준 정규분포 표기법은 다차원 표준 정규분포(multivariate normal distribution)이며, 분산을 1이 아닌 공분산 행렬로 나타낸 것이다.
* 위 정규분포 표기법을 읽는 방법 다음과 같다.
    * 정규분포 $x_{t}$는 평균이 $\sqrt{1-\beta_{t}}x_{t-1}$이고 분산이 $\beta_{t}I$이며, 여기서 $\beta_{t}$는 작은 상수이다.
    * 평균은 이전 step $t-1$의 $x_{t-1}$에 매우 작은 상수값인 $\sqrt{1-\beta_{t}}$를 곱해줬다.
    * 분산은 공분산 행렬에 작은 상수 $\beta_{t}$를 곱해줬다.

결과적으로 위 과정을 $T$ step만큼 해주면 최종적으로 noise data가 될 것이고, step 0(원본 데이터)에서 step T(noise data)까지 변환하는 식은 다음과 같다.

$$ q(x_{1:T}\vert{}x_{0})=\prod_{t=1}^{T}q(x_{t}\vert{}x_{t-1}) $$

하지만 모든 step을 진행하지 않아도 최종 결과 $x_{T}$를 얻을 수 있다.

$$ x_{T} = \sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{(1-\bar{\alpha}_{t})}\cdot\epsilon$$

* 변수들은 다음과 같다.
    * $\epsilon\sim{}N(0,I)$
    * $\alpha_{t} = 1-\beta_{t}$
    * $\bar{\alpha}_{t}=\prod_{s=1}^{t}\alpha_{s}$
* 위 과정을 해석하면 다음과 같다.
    * 최종 $x_{t}$는 first step $x_{0}$을 통해서 얻을 수 있다.
    * $\sqrt{\bar{\alpha_{t}}}x_{0}$는 $x_{0}\cdot\sqrt{\prod_{s=1}^{t}(1-\beta_{s})}$이다.
        * $x_{0}$은 input data를 의미한다.
        * $\sqrt{\prod_{s=1}^{t}(1-\beta_{s})}$은 최종적으로 곱해질 노이즈의 양이다.
        * 이 식은 $\prod_{s=1}^{t}$만큼 $1-\beta_{s}$ 곱해준 값을 $x_{0}$에 곱해줌으로써, 원본 데이터에 점진적으로 노이즈가 더해지는 과정을, 한번에 식으로 표현하였다.
    * $\sqrt{(1-\bar{\alpha_{t}})}\cdot{}\epsilon$은 $\sqrt{(1-\prod_{s=1}^{t}(1-\beta_{s}))}\cdot{}\epsilon{}$이다.
        * $\epsilon$은 정규분포에서 sampling한 값이다.
        * $\prod_{s=1}^{t}(1-\beta_{s})$은 매우 작은 값이 될 것 이다.
        * $1-\prod_{s=1}^{t}(1-\beta_{s})$은 1에서 매우 작은값을 빼주는 것은 1보다 작은 1의 근사값이 될 것이다.
        * $\sqrt{(1-\prod_{s=1}^{t}(1-\beta_{s}))}$또한 1보다 작은 1의 근사값이 될 것이다.
        * $\sqrt{(1-\prod_{s=1}^{t}(1-\beta_{s}))}\cdot{}\epsilon{}$은 정규분포에서 sampling한 값의 1보다 작은 1의 근사값을 곱해준 값으로 sampling한 값보다 조금 작아지게 될것이다.
    * 따라서 $ \sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{(1-\bar{\alpha}_{t})}\cdot\epsilon $ 은 input data에 최종적으로 더해질 노이즈의 양을 곱해주고 1보다 조금 더 작은 소수값을 더해준다.

### Reverse process

* Forward process에서 점진적으로 추가된 노이즈를 반대로 점진적으로 제거하여 데이터를 원래상태로 되돌린다.
* 원래대로 되돌리는 $x_{t}$를 사용하여 $x_{t-1}$과정은 신경망 매개변수 $\theta$를 통해 forward process의 gaussian transition(정규 분포만큼 추가해준 노이즈)를 reverse(원래대로 되돌린다)한다. 식은 다음과 같다.

$$ p_{\theta}(x_{t-1}\vert{}x_{t})=N(x_{t-1}; \mu_{\theta}(x_{t}, t), \sigma{}^{2}I) $$

* $\mu_{\theta{}}(x_{t},t)$는 $x_{t}$와 $t$를 입력받는 네트워크이다.
* 