---
layout: post 
title: "(Diffusion)DiffFashion: Reference-based Fashion Design with
Structure-aware Transfer by Diffusion Models"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review]
---

### [Diffusion Literature List](https://maizer2.github.io/1.%20computer%20engineering/2023/02/01/Literature-of-diffusion.html)

# DiffFashion: Reference-based Fashion Design with Structure-aware Transfer by Diffusion Models

## Abstract

> Image-based fashion design with AI techniques has attracted increasing attention in recent years.
>> 최근에는 인공지능 기술을 활용한 이미지 기반 패션 디자인이 주목받고 있습니다.

> We focus on a new fashion design task, where we aim to transfer a reference appearance image onto a clothing image while preserving the structure of the clothing image.
>> 우리는 의복 이미지의 구조를 보존하면서 기존의 참조 외모 이미지를 의복 이미지로 전환하는 새로운 패션 디자인 작업에 초점을 맞추고 있습니다.

> It is a challenging task since there are no reference images available for the newly designed output fashion images.
>> 이는 새롭게 디자인된 출력 패션 이미지에 대한 참조 이미지가 없기 때문에 도전적인 작업입니다.

> Although diffusion-based image translation or neural style transfer (NST) has enabled flexible style transfer, it is often difficult to maintain the original structure of the image realistically during the reverse diffusion, especially when the referenced appearance image greatly differs from the common clothing appearance.
>> 확산 기반 이미지 변환 또는 신경망 스타일 전이(NST)는 유연한 스타일 전이를 가능하게 하지만, 참조 외모 이미지가 일반적인 의복 모양과 크게 다를 때는 역 확산 과정에서 이미지의 원래 구조를 현실적으로 유지하는 것이 어려울 수 있습니다.

> To tackle this issue, we present a novel diffusion model-based unsupervised structure-aware transfer method to semantically generate new clothes from a given clothing image and a reference appearance image.
>> 이 문제를 해결하기 위해, 우리는 의복 이미지와 참조 외모 이미지를 이용하여 의미론적으로 새로운 의복을 생성하는 확산 모델 기반의 비지도 구조 인식 전이 방법을 제안합니다.

> In specific, we decouple the foreground clothing with automatically generated semantic masks by conditioned labels.
>> 구체적으로, 우리는 조건화된 라벨에 의해 자동으로 생성된 의미론적 마스크를 사용하여 전경 의복을 분리합니다.

> And the mask is further used as guidance in the denoising process to preserve the structure information.
>> 그리고 마스크는 노이즈 제거 과정에서 구조 정보를 보존하기 위한 가이드로 더 활용됩니다.

> Moreover, we use the pre-trained vision Transformer (ViT) for both appearance and structure guidance.
>> 또한, 외모와 구조 가이드 모두에 대해 사전 훈련된 비전 트랜스포머(ViT)를 사용합니다.

> Our experimental results show that the proposed method outperforms state-of-the-art baseline models, generating more realistic images in the fashion design task.
>> 실험 결과, 우리가 제안한 방법은 최신 기준 모델보다 성능이 우수하며, 패션 디자인 작업에서 더 현실적인 이미지를 생성합니다.

> Code and demo can be found at https://github.com/Rem105-210/DiffFashion.
>> 코드와 데모는 https://github.com/Rem105-210/DiffFashion에서 찾을 수 있습니다.

![Figure-1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-1.PNG)

> Fig. 1. Two examples of a reference-based fashion design task. For a given image pair, i.e., a bag and a referenced appearance image, our method can generate a new image with appearance similarity to the appearance image and structure similarity to the bag image.
>> 사진 1. 참조 기반 패션 디자인 과제의 두 가지 예입니다. 주어진 이미지 쌍, 즉 가방과 참조된 외관 이미지의 경우, 우리의 방법은 외관 이미지와 유사하고 가방 이미지와 구조적 유사성을 가진 새로운 이미지를 생성할 수 있다.

![Figure-2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-2.png)

> Fig. 2. The pipeline of our approach. (a): We add noise to clothing image $x_{S}^{A}$, and then use different label conditions to estimate the noise in the denoising process. The semantic mask of the $x_{0}^{S}$ can be obtained from the noise difference. (b): We denoise the reference appearance image $x_{0}^{A}$. In the denoising process, we use the mask in (a) to replace the background with pixel values obtained from the encoding process at the same timestamp. (c) and (d): We use DINO-VIT features to compute structure loss  between $x_{t}^{A}$ and $x_{0}^{S}$, appearance loss between $x_{t}^{A}$ and $x_{0}^{A}$, to guide the denoising process. Purple dots and yellow dots represent the denoising process with the same timesteps respectively.
>> 그림 2. 우리 접근 방식의 파이프라인. (a): 의류 이미지 $x_{0}^{S}$에 노이즈를 추가한 다음, 노이즈 제거 프로세스에서 노이즈를 추정하기 위해 다른 레이블 조건을 사용한다. $x_{0}^{S}$의 시맨틱 마스크는 노이즈 차이에서 얻을 수 있다. (b): 기준 외관 이미지 $x_{0}^{A}$를 노이즈 제거한다. 노이즈 제거 프로세스에서는 (a)의 마스크를 사용하여 동일한 타임스탬프에서 인코딩 프로세스에서 얻은 픽셀 값으로 배경을 대체합니다. (c) 및 (d): 우리는 노이즈 제거 프로세스를 안내하기 위해 DINO-VIT 기능을 사용하여 $x_{t}^{A}$와 $x_{0}^{S}$ 사이의 구조 손실, $x_{t}^{A}$와 $x_{0}^{A}$ 사이의 외관 손실을 계산한다. 보라색 점과 노란색 점은 각각 동일한 시간 단계를 가진 노이즈 제거 프로세스를 나타냅니다.

## I. Introduction

> Image-based fashion design with artificial intelligence (AI) techniques [1]–[6] has attracted increasing attention in recent years. 
>> 의상 이미지에 다른 도메인 객체의 외관을 참조하여 새로운 의류 디자인을 생성하려는 패션 디자인의 새로운 과제가 있다. 

> There is a growing expectation that AI can provide inspiration for human designers to create new fashion designs. 
>> 인공지능 기술을 활용한 이미지 기반 패션 디자인[1]~[6]은 최근 주목 받고 있다. 

> One of the emerging tasks in fashion design is to add specific texture elements from non-fashion domain images into clothing images to create new fashions. 
>> 인공지능은 인간 디자이너가 새로운 패션 디자인을 만드는 데 영감을 제공할 수 있는 능력을 갖고 있기 때문이다. 

> For example, given a clothing image, a designer may want to generate a new clothes design with the appearance of another domain object as a reference, as shown in Fig. 1.
>> 예를 들어 의류 이미지를 제공하면 디자이너는 다른 도메인 객체의 외관을 참조하여 새로운 옷 디자인을 생성하고자 할 수 있다(Fig. 1).

> Generative adversarial network (GAN)-based methods [2], [7], [8] can be adopted in the common fashion design tasks to generate new clothes. 
>> 일반적인 패션 디자인 작업에서 새로운 옷을 생성하는 데 GAN(Generative adversarial network)-기반 기법[2], [7], [8]을 채택할 수 있다. 

> However, GAN-based methods can hardly have good control over the appearance and shape of clothes when transferring from non-fashion domain images. 
>> 그러나 GAN 기반 기술은 패션 도메인에서 아닌 이미지로부터 외관과 형태를 제어하기 어렵다. 

> Recently, diffusion models [9]–[11] have been widely explored due to the realism and diversity of their results, and have been applied in various generative areas, such as text image generation [12], [13] and image translation [14]. 
>> 최근 확산 모델[9]~[11]은 결과의 현실감과 다양성 때문에 널리 탐구되고 있으며, 텍스트 이미지 생성[12], [13] 및 이미지 번역 [14]과 같은 다양한 생성 분야에서 적용되고 있다. 

> Some approaches [15], [16] consider both structure and appearance in image transfer. 
>> 일부 접근 방식[15], [16]은 이미지 전송에서 구조와 외관을 모두 고려한다. 

> Kwon et al. [15] use a diffusion model and a special structural appearance loss for appearance transfer, which performs well in transforming the appearance between similar objects, such as from zebras to horses and from cats to dogs.
>> Kwon et al. [15]은 확산 모델과 특수 구조 외관loss를 외관 전송에 사용하여, 얼룩말에서 말로, 고양이에서 개로 같은 유사한 객체들 사이의 외관 변환에 우수한 성능을 보인다.

> However, there are two main challenges when applying the commonly used image transfer methods to the reference-based fashion design task shown in Fig. 1. 
>> 그러나, 그림 1에서 보여진 참조 기반 패션 디자인 작업에 일반적으로 사용되는 이미지 전송 방법을 적용할 때는 두 가지 주요한 문제가 있다. 

> First, common image transfer methods only consider the translation between semantically similar images or objects. 
>> 첫째, 일반적인 이미지 전송 방법은 의미론적으로 유사한 이미지나 객체 간의 전송만을 고려한다. 

> For example, the transformation in [15] is based on the similarity of the semantically related objects in vision transformer (ViT) [17] features. 
>> 예를 들어 [15]의 변환은 비전 트랜스포머(ViT) [17] 기능 내 의미론적으로 관련된 객체의 유사성을 기반으로 한다. 

> In the reference-based fashion design task, the semantic features of reference appearance images are always far different from clothing images. 
>> 그러나 참조 외관 이미지의 의미론적 특징은 항상 의류 이미지와 매우 다르다. 

> As a result, commonly used image transfer methods usually generate unrealistic fashions in this task and difficult to transfer the appearance. 
>> 따라서 이러한 상황에서 일반적으로 사용되는 이미지 전송 방법은 이 작업에서 비현실적인 패션을 생성하고 외관을 전달하기 어렵게 만든다. 

> Besides, These methods only transfer the style or appearance, which hardly converts the appearance to a suitable texture material by using a non-clothing image. 
>> 게다가 이러한 방법은 스타일이나 외관만 전달하며, 비의류 이미지를 사용하여 외관을 적합한 질감 재료로 전환하는 것이 거의 불가능하다. 

> Second, image transfer methods [18] usually require a large number of samples from both source and target domains. 
>> 둘째, 이미지 전송 방법[18]은 일반적으로 소스 및 대상 도메인에서 많은 샘플을 필요로 한다. 

> However, there are no samples available for newly designed output domains, resulting in a lack of guidance during the transfer process. 
>> 그러나 새롭게 설계된 출력 도메인에 대한 샘플이 없으므로 전송 과정에서 도움이 되는 안내가 부족하다. 

> Thus, the generated new fashion images are likely to lose the structural information of the input clothing images.
>> 따라서 생성된 새로운 패션 이미지는 입력 의류 이미지의 구조 정보를 잃어버리는 경우가 많다.

## II. Related work

### A. Fashion Design

> Fashion design models aim to design new clothing from a given clothing collection.
>> 패션 디자인 모델은 주어진 의류 컬렉션에서 새로운 의류를 디자인하는 것을 목표로 합니다.

> Sbai et al. [3] use GAN to learn the encoding of clothes, and then use the latent vector to perform the stylistic transformation.
>> Sbai et al. [3]는 GAN을 사용하여 의류의 인코딩을 학습하고, 그 다음 잠재 벡터를 사용하여 스타일 변환을 수행합니다.

> Cui et al. [8] use the sketch image of the clothes to control the generated structure.
>> Cui et al. [8]은 의류의 스케치 이미지를 사용하여 생성된 구조를 제어합니다.

> Good results have been achieved in terms of structural control.
>> 구조 제어 측면에서 좋은 결과를 얻었습니다.

> Yan et al. [2] use a patch-based structure to implement texture transfer on generated objects.
>> Yan et al. [2]는 생성된 객체에 대한 텍스처 전이를 구현하기 위해 패치 기반 구조를 사용합니다.

> However, they cannot use other images as texture references and their tasks are limited to generating new samples from existing clothes collections.
>> 그러나 그들은 다른 이미지를 텍스처 참조로 사용할 수 없으며, 그들의 작업은 기존 의류 컬렉션에서 새로운 샘플을 생성하는 것으로 제한됩니다.

> As a result, due to the unreliable training problem of GAN, more advanced methods are needed to achieve improved realism in generated effects.
>> 결과적으로 GAN의 불안정한 훈련 문제로 인해, 생성된 효과의 개선된 현실감을 얻기 위해 더 나은 방법이 필요합니다.

### B. GAN-based Image Transfer

> The image-to-image translation aims to learn the mapping between the source and the target domains, often using a GAN network.
>> 이미지 간 변환은 종종 GAN 네트워크를 사용하여 소스 도메인과 대상 도메인 사이의 매핑을 학습하는 것을 목표로 합니다.

> Paired data methods like [19], [20] use the target image corresponding to each input for the condition in the discriminator.
>> [19], [20]과 같은 대응 데이터 방법은 판별기에서 각 입력에 대한 대상 이미지를 사용하여 조건을 만듭니다.

> Unpaired data methods like [21]-[23] decouple the common content space and the specific style space in an unsupervised way.
>> [21]-[23]과 같은 대응하지 않는 데이터 방법은 공통 콘텐츠 공간과 특정 스타일 공간을 비지도 방식으로 분리합니다.

> But both these methods require amounts of data from both domains.
>> 그러나 이러한 두 가지 방법 모두 두 도메인에서 대량의 데이터가 필요합니다.

> Besides, the encoding structure of GANs makes it difficult to decouple appearance and structural information.
>> 게다가 GAN의 인코딩 구조는 외관과 구조 정보를 분리하기 어렵게 만듭니다.

> When the gap between the two domains is too large, the result may not be transformed [23]-[25] or have lost information from the original domain [26].
>> 두 도메인 간의 격차가 너무 큰 경우, 결과물은 변환되지 않을 수 있습니다 [23]-[25] 또는 원래 도메인에서 정보를 잃어버릴 수 있습니다 [26].

### C. Diffusion Model-based Image Transfer

> Recently, denoising diffusion probabilistic models (DDPMs) have emerged as a promising alternative to GANs in image-to-image translation tasks. 
>> 최근에, 잡음 제거 확산 확률 모델(DDPMs)이 이미지 전송 작업에서 GAN에 대한 유망한 대안으로 등장했습니다.

> Palette [18] firstly applies the diffusion model in image translation and achieves good results in colorization, inpainting, and other tasks. 
>> Palette[18]은 이미지 전송에 확산 모델을 처음으로 적용하고, 컬러화, 인페인팅 및 기타 작업에서 좋은 결과를 얻었습니다.

> However, this approach requires the target image as a condition for diffusion, making it infeasible for unsupervised tasks.
>> 그러나 이 방법은 확산의 조건으로 대상 이미지가 필요하므로, 무감독 작업에는 적용이 불가능합니다.

> For appearance transfer, DiffuseIT [15] uses the same DINO-ViT guidance as [16], which greatly improves the realism of the transformation. 
>> 외모 전송에는 DiffuseIT[15]이 [16]의 DINO-ViT 가이드를 사용하여 동일한 결과를 보여주었으며, 이로 인해 변환의 현실성이 크게 향상되었습니다.

> However, it still cannot solve the problem of lacking matching objects in the clothing design task.
>> 그러나 여전히 의복 디자인 작업에서 일치하는 객체 부재 문제를 해결할 수 없습니다.

### D. Neural Style Transfer (NST)

> Neural style transfer (NST) has shown great success in transferring artistic styles. There are mainly two types of approaches to modeling the style or visual texture in NST. 
>> 신경 스타일 전송 (NST)은 예술적 스타일을 전송하는 데 큰 성공을 거두었습니다. NST에서 스타일 또는 시각적 질감을 모델링하는 데는 주로 두 가지 유형의 접근 방식이 있습니다.

> One is based on statistical methods [27], [28], in which the style is characterized as a set of spatial summary statistics.
>> 하나는 통계 방법[27], [28]을 기반으로하며, 여기서 스타일은 공간 요약 통계의 집합으로 특징화됩니다.

> The other is based on non-parametric methods, such as using Markov Random Field [29], [30], in which they swap the content neural patches with the most similar ones to transfer the style.
>> 다른 하나는 마르코프 랜덤 필드[29], [30]와 같은 비모수적 방법을 기반으로 하며, 여기서 콘텐츠 신경 패치를 가장 유사한 패치로 교체하여 스타일을 전송합니다.

> After texture modeling, a pre-trained convolutional neural network (CNN) network is used to complete the style transfer. 
>> 질감 모델링 후, 사전 훈련 된 컨볼루션 신경망 (CNN) 네트워크를 사용하여 스타일 전송을 완료합니다.

> Although NST-based methods work well for global artistic style transfer, their content/style decoupling process is not suitable for fashion design. 
>> NST 기반 방법은 전반적인 예술적 스타일 전송에 대해 잘 작동하지만, 콘텐츠/스타일 분리 과정은 패션 디자인에 적합하지 않습니다. 

> In addition, NST-based methods assume the transfer is between similar objects or domains. 
>> 또한, NST 기반 방법은 전송이 유사한 객체 또는 도메인 간에 이루어진다고 가정합니다.

> Tumanyan et al. [16] propose a new NST loss from DINO-ViT, which succeeds in transferring appearance between two semantically related objects, such as “cat and dog” or “orange and ball”. 
>> Tumanyan et al. [16]은 DINO-ViT에서 새로운 NST 손실을 제안하여 "고양이와 개" 또는 "오렌지와 공"과 같이 의미론적으로 관련된 두 객체 간의 외모 전송에 성공했습니다.

> However, in our task, there are no specific related objects between the clothing image and the appearance image.
>> 그러나, 우리의 작업에서는 의복 이미지와 외모 이미지 간에 구체적으로 관련된 객체가 없습니다.

## III. PRELIMINARY OF DENOISING DIFFUSION PROBABILISTIC MODEL

> Diffusion probabilistic models [9]–[11] are a type of latent variable model that consists of a forward diffusion process and a reverse diffusion process.
>> 확산 확률 모델(Diffusion probabilistic models)은 전방확산 과정과 후방확산 과정으로 이루어진 잠재변수 모델의 일종입니다.

> In the forward process, we gradually add noise to the data, and then sample the latent $x_{t}$ for $t = 1, ..., T$ as a sequence. 
>> 전방 과정에서는 데이터에 점차적으로 노이즈를 추가하고, $t = 1, ..., T$에 대한 잠재 변수 $x_{t}$를 시퀀스로 샘플링합니다. 

> Noise added to data in each step is sampled from a Gaussian distribution, and the transmission can be represented as $q(x_{t}\vert{}x_{t-1}) = N(\sqrt{1 − β_{t}}x_{t-1}, β_{t}I)$, where the Gaussian variance $(β_{t})_{t=0}^{T}$ can either be learned or scheduled.
>> 각 단계에서 데이터에 추가된 노이즈는 가우시안 분포에서 샘플링되며, 전송은 $q(x_{t}\vert{}x_{t-1}) = N(\sqrt{1 − β_{t}}x_{t-1}, β_{t}I)$로 표현됩니다. 여기서 가우시안 분산 $(β_{t})_{t=0}^{T}$은 학습되거나 스케줄링될 수 있습니다.

> Importantly, the final latent encoding by the forward process can be directly obtained by, 
>> 중요한 것은 전방 과정에 따른 최종 잠재 인코딩은 다음과 같이 직접 얻을 수 있다는 것입니다.

$$x_{t} = \sqrt{\bar{α}_{t}}x_{0} + \sqrt{(1 − \bar{α}_{t})ε}, ε ∼ N(0, I), (1) $$

> where $α_{t} = 1 − β_{t}$ and $\bar{α_{t}}$ $= \prod_{s=1}^{t} α_{s}$.
>> 여기서 $α_{t} = 1 − β_{t}$이고, $\bar{α_{t}}$ $= \prod_{s=1}^{t} α_{s}$입니다.

> Then in the reverse process, the diffusion model learns to reconstruct the data by denoising gradually. A neural network is applied to learn the parameter θ to reverse the Gaussian transitions by predicting $x_{t-1}$ from $x_{t}$ as follow:
>> 그런 다음 후방 과정에서는 확산 모델이 점진적으로 노이즈를 제거하여 데이터를 재구성하도록 합니다. 신경망은 매개변수 θ을 학습하여 Gaussian 전이를 반전시키는 데 적용됩니다. $x_{t}$에서 $x_{t-1}$을 예측합니다.

$$ p_{θ}(x_{t-1}\vert{}x_{t}) = N(x_{t-1}; µ_{θ}(x_{t}, t), σ^{2}I). (2) $$

> To achieve a better image quality, the neural network takes the sample $x_{t}$ and timestamp $t$ as input, and predicts the noise added to $x_{t-1}$ in the forward process instead of directly predicting the mean of $x_{t-1}$.
>> 이를 위해 신경망은 샘플 $x_{t}$와 타임스탬프 $t$를 입력으로 사용하고, $x_{t-1}$의 평균을 직접 예측하는 대신 전방 과정에서 $x_{t-1}$에 추가된 노이즈를 예측합니다.

> The denoising process can be defined as: 
>> 노이즈 제거 과정은 다음과 같이 정의될 수 있습니다.

$$ µ_{θ}(x_{t}, t) = \frac{1}{\sqrt{\bar{α}_{t}}}(x_{t} − \frac{1 − \bar{α}_{t}}{\sqrt{1 − \bar{α}_{t}}}ε_{θ}(x_{t}, t)), (3) $$

> where $ε_{θ}(x_{t}, t)$ is the diffusion model trained by optimizing the objective, i.e., 
>> 여기서 $ε_{θ}(x_{t}, t)$는 목적함수를 최적화하여 교육된 확산 모델입니다.

$$ min_{θ}L(θ) = E_{t,x_{0},ε}[(ε − ε_{θ}(\sqrt{\bar{α}_{t}}x_{0}) + \sqrt{(1 − \bar{α}_{t})ε}, t))^{2}]. (4) $$

> In the image translation task, there are two mainstream methods to complete the translation. 
>> 이미지 번역 작업에서는 두 가지 주요 방법이 있습니다. 

> One is using the conditional diffusion model, which takes extra conditions, such as text and labels as input in the denoising process. 
>> 하나는 조건부 확산 모델을 사용하는 것으로, 이 경우에는 텍스트나 레이블과 같은 추가 조건을 denoising 과정에서 입력으로 사용합니다. 

> Then the diffusion model $ε_{θ}$ in Eq. (3) and Eq. (4) can be replaced with $ε_{θ}(x_{t}, t, y)$, where $y$ is the condition.
>> 그러면 식 (3)과 (4)에서 더 이상 $ε_{θ}$가 아니라 $ε_{θ}(x_{t}, t, y)$와 같은 형태로 표현됩니다. 여기서 $y$는 조건입니다.

> The other type of method [31] uses pre-trained classifiers to guide the diffusion model in the denoising process and freezes the weights of the diffusion model. 
>> 다른 방법 [31]은 사전 훈련된 분류기를 사용하여 denoising 과정에서 확산 모델을 가이드하고 확산 모델의 가중치를 고정시키는 방법입니다. 

> With the diffusion model and a pre-trained classifier $p_{φ}(y\vert{}x_{t})$, the denoising process $µ_{θ}(x_{t}, t)$ in Eq. (3) can be supplemented with the gradient of the classifier, i.e., 
>> 산 모델(diffusion model)과 미리 학습된 분류기 $p_{φ}(y\vert{}x_{t})$를 사용하여, 수식 (3)에 나타난 잡음 제거(denoising) 프로세스 $µ_{θ}(x_{t}, t)$를 분류기의 기울기(gradient)와 함께 보완할 수 있습니다. 즉 다음의 식과 같다.

$$ \hat{µ}_{θ}(x_{t}, t) = µ_{θ}(x_{t}, t) + σ_{t}∇logp_{φ}(y\vert{}x_{t}). $$

## IV. PROPOSED METHOD

### A. Overview of Fashion Design with DiffFashion:

> Given a clothing image $x_{0}^{S}$ and a reference appearance image $x_{0}^{A}$, our proposed DiffFashion aims to design a new clothing fashion that preserves the structure in $x_{0}^{S}$ and transfers the appearance from $x_{0}^{A}$ while keeping it natural, as shown in Fig. 2.
>> 주어진 의류 이미지 $x_{0}^{S}$와 참조 외모 이미지 $x_{0}^{A}$를 기반으로, DiffFashion은 $x_{0}^{S}$의 구조를 보존하고 $x_{0}^{A}$의 외모를 전달하여 자연스러운 새로운 의류를 디자인하는 것을 목표로 합니다. 이는 그림 2에서 볼 수 있습니다.

> We list two main challenges in this task. First, there are no given reference images for the output result since there is no standard answer for fashion design. Without the supervision of the ground truth, it is difficult to train the model. Second, preserving the structure information from the given input clothing image while transferring the appearance is also being under-explored.
>> 이 작업에서는 두 가지 주요 도전 과제가 있습니다. 첫째, 패션 디자인에 대한 표준 답변이 없으므로 출력 결과에 대한 참조 이미지가 제공되지 않습니다. 따라서 실제 상황의 지도 없이 모델을 훈련하는 것이 어렵습니다. 둘째, 주어진 입력 의류 이미지의 구조 정보를 보존하면서 외모를 전달하는 것도 아직 연구가 미흡합니다.

> To address those two challenges, we present the DiffFashon, which is a novel structure-aware transfer model with the diffusion model. 
>> 이 두 가지 도전 과제를 해결하기 위해, 우리는 확산 모델을 사용한 새로운 구조 인식 전이 모델인 DiffFashion을 제안합니다. 

> We use the diffusion model [32] pre-trained on Imagenet [33] for all the denoising processes in DiffFashion.
>> DiffFashion에서 모든 노이즈 제거 과정에는 Imagenet [33]에서 사전 훈련된 확산 모델 [32]을 사용합니다.

> First, we decouple the foreground clothing with a generated semantic mask by conditioned labels, as shown in Fig. 2 (a). 
>> 먼저, 조건부 레이블을 사용하여 생성된 시맨틱 마스크로 전경 의류를 분리합니다(그림 2(a) 참조). 

> Then, we encode the appearance image $x_{0}^{A}$ with DDPM, and denoise it with mask guidance to preserve the structure information, as shown in Fig. 2 (b). 
>> 그런 다음, DDPM으로 외모 이미지 $x_{0}^{A}$를 인코딩하면서 실수를 하지 않도록 도와주는 마스크 지도를 사용하여 노이즈를 제거하여 구조 정보를 보존합니다(그림 2(b) 참조). 

> Moreover, we use the DINO-ViT [17] for both appearance and structure guidance during the denoising process, as shown in Fig. 2 (c) and (d).
>> 또한, DINO-ViT [17]를 사용하여 노이즈 제거 과정에서 외모와 구조 모두를 안내합니다(그림 2(c)와 2(d) 참조).

> The details are illustrated in the following sections.
>> 자세한 내용은 다음 섹션에서 설명됩니다.

### B. Mask Generation by Label Condition

> To decouple the foreground clothing and background, we generate a semantic mask for the input clothing image $x_{0}^{S}$ with label conditions. The generated semantic mask is also used for preserving the structure information in later steps.
>> 전경 의류와 배경을 분리하기 위해 레이블 조건에 따른 입력 의류 이미지 $x_{0}^{S}$에 대한 시맨틱 마스크를 생성합니다. 생성된 시맨틱 마스크는 이후 단계에서 구조 정보를 보존하는 데 사용됩니다.

> Existing methods commonly use additional inputs to obtain the foreground region. 
>> 기존 방법들은 일반적으로 전경 영역을 얻기 위해 추가적인 입력을 사용합니다.

> However, this leads to increased annotation expenses. Inspired by [34], we propose a mask generation approach that can obtain the foreground clothing area without external information or segmentation models. 
>> 그러나 이는 주석 비용이 증가하는 원인이 됩니다. [34]에서 영감을 받아, 외부 정보나 분할 모델 없이 전경 의류 영역을 얻을 수 있는 마스크 생성 방법을 제안합니다.

> Our approach leverages the label-conditional diffusion model to obtain the desired result.
>> 우리의 방법은 레이블 조건부 확산 모델을 활용하여 원하는 결과를 얻습니다.

> In the denoising process of the label-conditional diffusion model, there can be different noise estimates for the same latent given negative label conditions like phone and bag. 
>> 레이블 조건부 확산 모델의 노이즈 제거 과정에서, 전화나 가방과 같은 부정적인 레이블 조건이 주어질 때, 같은 잠재 변수에 대해 서로 다른 노이즈 추정치가 있을 수 있습니다.

> For these different noise estimates, the regions of the foreground object that are denoised tend to vary little in background regions but greatly in object regions. 
>> 이러한 다른 노이즈 추정치에 대해, 노이즈 제거된 전경 객체의 영역은 배경 영역보다는 덜 변화하지만, 객체 영역에서는 크게 변화합니다.

> By taking the difference in the noise area, we can obtain the mask of the object to be edited, as shown in Fig. 2(a).
>> 노이즈 영역의 차이를 취함으로써, Fig. 2(a)와 같이 편집 대상 객체의 마스크를 얻을 수 있습니다.

> Instead of generating a mask with the latent of the forward process like [34], we observe that in the denoising process, $x_{t}^{S}$ has less perceptual appearance information than $x_{qt}^{S}$ (the image in the forward process with timestamp $t$). 
>> [34]와 같은 전방 과정의 잠재 변수로 마스크를 생성하는 대신, 노이즈 제거 과정에서 $x_{t}^{S}$는 전방 과정에서 타임스탬프 $t$에 해당하는 이미지인 $x_{qt}^{S}$보다는 덜 인지적인 외형 정보를 갖습니다.

> Therefore, we generate a mask from the image in the denoising process $x_{t}^{S}$ instead of the image $x_{qt}^{S}$ in the forward process. 
>> 따라서 전방 과정의 이미지 $x_{qt}^{S}$ 대신 노이즈 제거 과정의 이미지 $x_{t}^{S}$에서 마스크를 생성합니다.

> Although the structure of $x_{t}^{S}$ may have some slight variations, it still provides a better representation of the overall structure information of the foreground object.
>> $x_{t}^{S}$의 구조가 약간 다르더라도, 전경 객체의 전반적인 구조 정보를 더 잘 나타내므로 더 좋은 표현을 제공합니다.

> Specifically, we input the clothing image $x_{0}^{S}$ into the diffusion model. 
>> 구체적으로, 의류 이미지 $x_{0}^{S}$를 확산 모델에 입력합니다.

> After DDPM encoding in the forward process, we obtain the image latent $x_{T/2}^{S}$ in half of the reverse process. 
>> 전방 과정에서 DDPM 인코딩 후, 우리는 역방향 과정의 절반에서 이미지 잠재 변수 $x_{T/2}^{S}$를 얻습니다.

> Denote the foreground label as $y_{p}$, representing the foreground clothing object. 
>> 전경 의류 객체를 나타내는 전경 레이블 $y_{p}$로 표시합니다.

> Then the noise map for the foreground clothing can be obtained by
>> 그런 다음, 전경 의류의 노이즈 맵은 전경 레이블 $y_{p}$에 대해 얻을 수 있습니다.

$$ M_p = ε_{θ}( x̂_{T/2}^{S} , T/2, y_p), (5) $$

> where $x̂_{T/2}^{S}$ is the estimated source image predicted from $x_{T/2}^{S}$ by Tweedie’s method [35], i.e.,
>> 여기서 $x̂_{T/2}^{S}$는 Tweedie의 방법 [35]을 통해 예측된 $x_{T/2}^{S}$에서 추정된 소스 이미지를 나타냅니다. 즉,

$$ x̂_{t} = \frac{x_{T/2}}{\sqrt{\bar{α_{T/2}}}} − \frac{\sqrt{1−\bar{α_{T/2}}}}{\sqrt{\bar{α_{T/2}}}}ε_{θ}(x_{T/2} , T/2, y_{p}). (6) $$

> Denote non-foreground labels as $y_{n}$, representing negative objects.
>> 비전경 레이블은 음의 객체를 나타내는 $y_{n}$으로 표시합니다.

> We use $N$ different non-foreground label conditions to get an averaged noise map, i.e., 
>> 우리는 $N$개의 다른 비전경 레이블 조건을 사용하여 평균 잡음 맵을 얻습니다. 즉,

$$ M_{n} = \frac{1}{N}\sum^{N}_{i=1}ε_{θ}(x̂_{T/2}^{S}, T/2, y_{i}), (7) $$

> where $i ∈ {1, ..., N}$.
>> 여기서 $i ∈ {1, ..., N}$ 입니다.

> The difference between the two noise maps $M_{p}$ and $M_{n}$ can be obtained. 
>> 두 잡음 맵 $M_{p}$와 $M_{n}$의 차이를 구할 수 있습니다.

> Then we set a threshold for binarization, which returns an editable semantic mask $M$ for the foreground clothing region.
>> 그런 다음 이진화를 위한 임계값을 설정하여, 전경 의류 영역에 대한 수정 가능한 시맨틱 마스크 $M$을 반환합니다.

### C. Mask-guided Structure Transfer Diffusion

> It is difficult to transfer the appearance of the original image to a new fashion clothing image when the gap between the two domains is too large [16].
>> 두 도메인 간의 격차가 너무 큰 경우, 원래 이미지의 모양을 새로운 패션 의류 이미지로 전달하는 것은 어렵습니다 [16].

> Because such methods control the appearance by a single loss of guidance, the redundant appearance information of the structure clothing reference image cannot be completely eliminated.
>> 이러한 방법은 단일 가이드 손실로 모양을 제어하므로 구조 의류 참조 이미지의 중복 모양 정보를 완전히 제거할 수 없습니다.

> Besides, when using a natural non-clothing image for appearance reference, the generated texture maybe not be suitable for clothing, because these models only transfer the style or appearance. 
>> 또한 모양 참조로 자연 비 의류 이미지를 사용할 때, 생성된 텍스처는 옷에 적합하지 않을 수 있습니다. 이 모델들은 스타일 또는 모양만 전달하기 때문입니다.

> The appearance cannot be converted to a suitable texture material like cotton for clothing.
>> 모양은 옷처럼 적합한 텍스처 재료로 변환될 수 없습니다.

> In DiffFashion, to address this problem, rather than transferring from the input clothing image $x_{0}^{S}$, we transfer from the reference appearance image $x_{0}^{A}$ to the output fashion clothing image with the guidance of the structural information of the input clothing image.
>> DiffFashion에서 이 문제를 해결하기 위해 입력 의류 이미지 $x_{0}^{S}$에서 전달하는 대신 구조 정보의 가이드로 참조 모양 이미지 $x_{0}^{A}$에서 출력 패션 의류 이미지로 전달합니다.

> Inspired by [36], it has been shown that for the same DDPM encoding latent with different label conditions used for denoising, the resulting natural images have similar textures and semantic structures.
>> [36]에서 영감을 받아, 동일한 DDPM 인코딩 잠재 변수에 대해 노이즈 제거에 사용되는 다른 레이블 조건으로 생성된 자연 이미지는 유사한 질감과 시맨틱 구조를 가짐이 입증되었습니다.

> We use the latent $x_{t}^{A}$ of the reference appearance image to transfer more appearance information to the output fashion. Besides, the texture of the appearance image can be transferred more realistic and suitable for clothing in the denoising process.
>> 참조 모양 이미지의 잠재 변수 $x_{t}^{A}$를 사용하여 출력 패션에 더 많은 모양 정보를 전달합니다. 또한, 모양 이미지의 더 적합하고 현실적인 텍스처를 노이즈 제거 과정에서 옷에 적합하도록 전달할 수 있습니다.

> Meanwhile, the semantic mask $M$ obtained from the previous step is used to preserve the structure of the clothing image.
>> 한편, 이전 단계에서 얻은 시맨틱 마스크 $M$은 의류 이미지의 구조를 보존하는 데 사용됩니다.

> As shown in Fig. 2(b), the appearance image $x_{0}^{A}$ is first used to encode by the forward process of DDPM. 
>> 그림 2(b)에서, 모양 이미지 $x_{0}^{A}$은 먼저 DDPM의 전방 과정으로 인코딩됩니다.

> Then the mask-guided denoising process is employed.
>> 그런 다음 마스크-유도 노이즈 제거 과정이 적용됩니다.

> Specifically, at each step in the denoising process, we estimate the new prediction $x_{t}^{A}$ from the diffusion model as follows, 
>> 구체적으로, 노이즈 제거 과정에서 각 단계마다 우리는 확산 모델에서 새로운 예측 $x_{t}^{A}$를 다음과 같이 추정합니다.

$$ x_{t}^{A} = \frac{1}{α_{t+1}}(x_{t+1}^{A} − \frac{1 − α_{t+1}}{\sqrt{1 − \bar{α}_{t+1}}} ε_{θ}(x_{t+1}^{A}, t + 1, y_{p})). (8) $$

> Then we combine the transferred foreground appearance $x_{t}^{A}$ and the clothing image of corresponding timestamp $x_{qt}^{S}$ with the generated mask $M$ as guidance, i.e., 
>> 그런 다음 생성된 마스크 $M$을 안내로 사용하여 전달된 전경 모습 $x_{t}^{A}$와 해당 타임스탬프의 의류 이미지 $x_{qt}^{S}$를 결합합니다. 즉,

$$ \tilde{x}_{t}^{A} = M\cdot{}x_{t}^{A} + (1 − M)\cdot{}[ω_{mix}\cdot{}x_{qt}^{S} + (1 − ω_{mix})· x_{t}^{A}], (9) $$

> where $ω_{mix}$ is the mix ratio of the appearance image and the clothing image. 
>> 여기서 $ω_{mix}$는 모습 이미지와 의류 이미지의 혼합 비율입니다. 

> This change ensures that the appearance information in the mask is transferred, while other structural information keeps consistent with the clothing image.
>> 이러한 변경으로 마스크의 모습 정보가 전달되면서 다른 구조 정보는 의류 이미지와 일관성을 유지합니다.

### D. ViT Feature Guidance

> As mentioned in [15], [16], the structure features and appearance features can be separated with DINO-ViT [17]. 
>> [15], [16]에서 언급한 대로, DINO-ViT [17]를 사용하여 구조적 특징과 모습 특징을 분리할 수 있습니다. 

> We use both appearance guidance and structure guidance in the denoising process to keep the output image realistic.
>> 우리는 출력 이미지를 실제적으로 유지하기 위해 노이즈 제거 과정에서 모습 안내와 구조 안내를 모두 사용합니다.

> Following [15], [16], we employ the [CLS] tokens in the last layer of ViT to guide the semantic appearance information as follows, 
>> [15], [16]에 따르면, ViT의 마지막 레이어 [CLS] 토큰을 다음과 같이 사용하여 시맨틱 모습 정보를 안내합니다.

$$ L_{app}(x_{0}^{A}, x̂_{t}^{A} ) = \vert{}\vert{}e^{L}_{[CLS]}(x_{0}^{A}) − e^{L}_{[CLS]}(x̂_{t}^{A})\vert{}\vert{}_{2} + λ_{MSE}\vert{}\vert{}x_{0}^{A} − x̂_{t}^{A}\vert{}\vert{}_{2}, (10) $$

> where $e_{[CLS]}^L$ is the last layer $[CLS]$ token, and $λ_{MSE}$ is the coefficient of global statistic loss between images.
>> 여기서 $e_{[CLS]}^L$는 마지막 레이어 $[CLS]$ 토큰이며, $λ_{MSE}$는 이미지 간 전역 통계 손실의 계수입니다.

> To better leverage the appearance between the object and the appearance image, we use the object semantic mask $M$ to remove the background pixel of $x̂_{t}^{A}$ in Eq. 10, and only compute the appearance loss of the object within the mask.
>> 객체와 모습 이미지 사이의 모습을 더욱 잘 활용하기 위해, 우리는 Eq. 10에서 객체 시맨틱 마스크 $M$을 사용하여 $x̂_{t}^{A}$의 배경 픽셀을 제거하고 마스크 내의 객체의 모습 손실만을 계산합니다.

> In addition, we adopt a patch-wise method in the structural loss to better leverage the local features. 
>> 추가적으로, 구조적 손실에서는 지역적인 특징을 더욱 잘 활용하기 위해 패치별 방법을 채택합니다. 

> We adopt the i-th key vector in the l-th attention layer of the ViT model, denoted as $k_{i}^{l}(x_{t})$, to guide the structural information of the i-th patch of the original clothing image as follows, 
>> ViT 모델의 l번째 어텐션 레이어에서 i번째 키 벡터 $k_{i}^{l}(x_{t})$를 사용하여 원래 의류 이미지의 i번째 패치의 구조 정보를 안내합니다. 이를 위해 다음과 같은 공식을 사용합니다:

$$ L_{struct}(x_{0}^{A}, x̂_{t}^{A}) = −\sum_{i} log(\frac{sim(k_{i}^{l,S}, k_{i}^{l,A})}{sim(k_{i}^{l,S}, k_{j}^{l,A}) + \sum_{j\neq{}i} sim(k_{i}^{l,S}, k_{j}^{l,A})}), (11) $$

> where sim(·, ·) is the exponential value of normalized cosine similarity,  i.e.,
>> 여기서 sim(·, ·)은 정규화된 코사인 유사도의 지수 값입니다. 즉.,

$$ sim(k_{i}^{I,S},k_{j}^{I,A})=exp(cos(k_{i}^{I}(x_{0}^{S}),k^{I}_{j}(x̂_{t}^{A})/\tau{}), (12) $$

> and τ is the temperature parameter.
>> 그리고 τ는 온도 매개 변수입니다.

> By using the loss in Eq. (11), we minimize the loss between keys at the same position of two images while maximizing the loss between keys of different positions. 
>> 우리는 Eq. (11)에서의 손실을 사용하여 두 이미지의 동일한 위치의 키 간 손실을 최소화하면서 다른 위치의 키 간 손실을 극대화합니다.

> Then our total loss for guidance is as follows:
>> 그런 다음, 우리의 지도를 위한 총 손실은 다음과 같습니다:

$$ L_{total} = λ_{struct}L_{struct} + λ_{app}L_{app}, (13) $$

> where $λ_{struct}$ and $λ_{app}$ are the coefficients of the structure loss and appearance loss.
>> 여기서 $λ_{struct}$와 $λ_{app}$는 구조 손실과 외모 손실의 계수입니다.

![Table-1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Table-1.PNG)

![Table-2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Table-2.PNG)

![Table-3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Table-3.PNG)

## V. EXPERIMENTS

> In this section, we describe our fashion design dataset and experiment settings.
>> 이 섹션에서는, 우리의 패션 디자인 데이터셋과 실험 설정에 대해 설명합니다.

> We also demonstrate the qualitative and quantitative results to show the effectiveness of our proposed method.
>> 또한, 우리가 제안한 방법의 효과를 보여주기 위해 질적 및 양적 결과를 제시합니다.

![Figure-3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-3.PNG)

> Samples from our proposed dataset of OceanBag. The left part shows some examples of marine life images, and the right part shows some samples of bag images.
>> 제안된 OceanBag 데이터 세트의 샘플. 왼쪽 부분은 해양 생물 이미지의 몇 가지 예를 보여주고, 오른쪽 부분은 가방 이미지의 몇 가지 샘플을 보여준다.

## A. Dataset

> To our best knowledge, there is no specific reference-based fashion design dataset currently.
>> 우리가 알기로는, 현재 특정한 참조 기반 패션 디자인 데이터셋이 없습니다.

> Thus, we collect a new dataset, namely OceanBag, with real handbag images and ocean animal images as reference appearances for generating new fashion designs.
>> 그래서, 우리는 새로운 데이터셋인 OceanBag을 수집했습니다. 이 데이터셋은 새로운 패션 디자인을 생성하는 데 참조 모양으로 실제 핸드백 이미지와 해양 동물 이미지를 사용합니다.

> OceanBag has 6,000 photos of handbags in various scenes and 2,400 pictures of various marine lives in the real world, among various marine scenes such as fish swimming on the ocean floor.
>> OceanBag은 다양한 장면에서 촬영한 핸드백 6,000장과 물고기가 바닷바닥에서 수영하는 등의 다양한 해양 장면에서 찍은 해양 생물 2,400장을 포함합니다. 

> The 2,400 marine scene images contain more than 80 kinds of marine organisms, 50% of which are fish, as well as starfish, crabs, algae, and other sea creatures, as shown in Fig. 3.
>> 이 2,400장의 해양 장면 이미지에는 80종 이상의 해양 생물, 그 중 50%는 물고기이며, 해변에서 볼 수 있는 민들레, 게, 해초 등 다른 바다 생물도 포함됩니다. 이는 그림 3에서 보여집니다.

> In our experiments, we screened 30 images for experiments based on diversity such as background complexity, species, and quantity of organisms.
>> 우리의 실험에서는 배경 복잡성, 종류, 생물 수 등의 다양성을 기반으로 실험을 위해 30개의 이미지를 선별합니다.

> We refer to images with solid backgrounds as simple backgrounds, while those with real scenes are referred to as complex backgrounds.
>> 배경이 단색인 이미지를 간단한 배경, 실제 장면이 있는 이미지를 복잡한 배경이라고 합니다.

> The complex ratio in Table I shows the proportion of complex background images in the dataset.
>> Table I에서 볼 수 있는 복잡도 비율은 데이터셋에서 복잡한 배경 이미지의 비율을 보여줍니다.

> The complex background of the marine biological dataset is usually real ocean pictures such as the seabed and the deep sea.
>> 해양 생물 데이터셋의 복잡한 배경은 일반적으로 해저나 심해 등의 실제 바다 사진입니다.

> For the bag images in the dataset, the complex backgrounds often include scenes of mall containers or tables.
>> 데이터셋의 핸드백 이미지의 복잡한 배경은 대개 상점의 컨테이너나 테이블 등의 장면을 포함합니다.

## B. Experimental Setup

> We conduct all experiments using a label-conditional diffusion model [32] pre-trained on the ImageNet dataset [33] with 256 × 256 resolution.
>> 우리는 256 × 256 해상도로 ImageNet 데이터셋 [33]에서 사전 학습 된 라벨 조건부 확산 모델 [32]을 사용하여 모든 실험을 수행합니다.

> In all experiments, we use a diffusion step of $T = 60$ and re-sampling repetitions of $N = 10$.
>> 모든 실험에서, $T = 60$의 확산 단계와 $N = 10$의 재샘플링 반복을 사용합니다.

> In a single RTX 3090 unit, it takes 20 seconds to generate each mask and 120 seconds to generate each image.
>> 하나의 RTX 3090 장치에서, 각 마스크를 생성하는 데 20초, 각 이미지를 생성하는 데 120초가 걸립니다.

> For fairness of comparison, other parameters in the diffusion model are kept the same as [15].
>> 비교의 공정성을 위해, 확산 모델의 다른 매개변수는 [15]와 동일하게 유지됩니다.

> In the mask generation part, we set the binarization threshold to -0.2.
>> 마스크 생성 부분에서, 우리는 이진화 임계값을 -0.2로 설정합니다.

> Due to the stochastic nature of the diffusion model, we generate masks using three different sets of labels, including “cellphone, forklift, pillow”, “waffle iron, washer, guinea pig” and “brambling, echidna, custard apple”.
>> 확산 모델의 확률적 특성 때문에, 우리는 "핸드폰, 포크리프트, 베개", "와플 아이언, 세탁기, 기니피그", "브램블링, 에친다, 커스터드 애플"과 같은 세 가지 다른 라벨 세트를 사용하여 마스크를 생성합니다.

> Then we choose the best one among them for guidance.
>> 그런 다음 그 중 가장 좋은 결과를 선택합니다.

> To ensure a fair comparison, We run the baseline DiffuseIT [15] three times as ours.
>> 공정한 비교를 위해, 우리는 DiffuseIT [15]를 우리 것과 같이 세 번 실행합니다.

> In the guidance part, to mitigate the uncontrollable effect of the mask and avoid information loss when the structural gap between the two objects is too large, we use mask guidance in the first 50% steps of the denoising stage, and the mix ratio $ω_{mix}$ is set to 0.98.
>> 가이드 부분에서, 두 객체 간의 구조적 차이가 너무 커지면 정보 손실을 피하고 마스크의 불안정한 효과를 완화하기 위해, 우리는 노이즈 제거 단계의 처음 50%에서 마스크 가이드를 사용하고, 혼합 비율 $ω_{mix}$는 0.98로 설정합니다.

> In the ViT guidance part, we set the coefficient of appearance loss $λ_{app}$ to 0.1 and 1 for structure loss $λ_{struct}$.
>> ViT 가이드 부분에서, 외모 손실 계수 $λ_{app}$를 0.1 및 구조 손실 계수 $λ_{struct}$를 1로 설정합니다.

> And we keep other parameters the same as DiffuseIT [15].
>> 그리고 우리는 다른 매개 변수를 DiffuseIT [15]와 동일하게 유지합니다.

## C. Evaluation Methods and Metrics

> There is currently no existing automatic metric suitable for evaluating fashion design across two natural images.
>> 현재 두 개의 자연 이미지 간의 패션 디자인을 평가하기에 적합한 자동 메트릭이 존재하지 않습니다.

> To keep the fashion image realistic, the migration degree of the appearance and the similarity of the structure sometimes are mutually contradictory when measured.
>> 패션 이미지를 현실적으로 유지하기 위해서는 외모의 이동 정도와 구조의 유사성은 때로 상호 모순적인 요소일 수 있습니다.

> To compare among different methods, we follow existing appearance transfer/fashion design works [15], [16], [37]–[40], which rely on human perceptual evaluation to validate the results.
>> 다른 방법들간의 비교를 위해, 우리는 인간의 지각 평가를 기반으로 한 기존의 외모 전이/패션 디자인 작업 [15], [16], [37] - [40]와 같이 결과를 검증합니다.

## D. Experimental Results

> We perform both quantitative and qualitative evaluations on the OceanBag dataset.
>> 우리는 OceanBag 데이터셋에서 양적 및 질적 평가를 수행합니다.

> We compare our model with SplicingViT [16], DiffuseIT [15], WCT2 [41] and STROTSS [42].
>> 우리는 SplicingViT [16], DiffuseIT [15], WCT2 [41], STROTSS [42]와 우리의 모델을 비교합니다.

> Fig. 4 shows qualitative results for all methods.
>> 그림 4는 모든 방법의 질적 결과를 보여줍니다.

> In all examples, it can be seen that in terms of fashion design, our method has achieved better performances in terms of realism and structure, while completing appearance transfer.
>> 모든 예제에서, 패션 디자인 측면에서 우리의 방법은 외모 전이를 완료하면서 현실성과 구조면에서 더 나은 성능을 보여줍니다.

> As for the DINO-ViT-based image-to-image translation methods, DiffuseIT successfully keeps the structure for most images, but it shows less appearance similarity.
>> DINO-ViT 기반 이미지 대 이미지 번역 방법에서 DiffuseIT은 대부분의 이미지에서 구조를 유지하지만 외모 유사성이 적습니다.

> SplicingViT transfers the appearance well, but its results are far away from realistic fashion images.
>> SplicingViT는 외모를 잘 전달하지만 결과는 현실적인 패션 이미지와 멀리 떨어져 있습니다.

> NST methods like STROTSS and WCT2 effectively retain the structure of the source image, but WCT2 outputs exhibit limited changes apart from color adjustments.
>> STROTSS와 WCT2와 같은 NST 방법은 소스 이미지의 구조를 효과적으로 유지하지만, WCT2 출력물은 색 조정을 제외하고는 제한된 변경을 보입니다.

> Although STROTSS successfully transfers the appearance, its results often suffer from color bleeding artifacts and thus show less authenticity.
>> STROTSS는 외모를 성공적으로 전달하지만, 그 결과는 종종 색상 출혈 아티팩트로 인해 신뢰성이 적습니다.

> We also conduct a user study to evaluate the samples and obtain subjective evaluations from participants.
>> 우리는 또한 사용자 조사를 실시하여 샘플을 평가하고 참가자의 주관적 평가를 얻습니다.

> Specifically, we ask 30 users to score all the output fashion images from all methods for each input pair.
>> 구체적으로, 우리는 30명의 사용자에게 각 입력 쌍에 대한 모든 방법의 출력 패션 이미지를 점수 매기도록 요청합니다.

> Detailed questions we have asked are as follows: 1) Is the picture realistic? 2) Is the image’s structure similar to the input image? 3) Is the output appearance similar to the input appearance image?
>> 우리가 물었던 자세한 질문은 다음과 같습니다: 1) 그림이 현실적인가? 2) 이미지의 구조가 입력 이미지와 유사한가? 3) 출력 외모가 입력 외모 이미지와 유사한가?

> The scores range from 0 to 100.
>> 점수는 0에서 100까지 범위가 있습니다.

> The overall score is the average of the three scores.
>> 전반적인 점수는 세 점수의 평균입니다.

> We show the averaged subjective evaluation results in Table II.
>> 우리는 테이블 II에서 평균 주관적 평가 결과를 보여줍니다.

> Our model obtains the best score in the overall performance and appearance correlation, and the second place in structure similarity and realism.
>> 우리 모델은 전반적인 성능과 외모 상관관계에서 가장 높은 점수를 얻으며, 구조 유사성과 현실성에서 두 번째로 높은 점수를 얻습니다.

> WCT2 shows the best in realism and structure similarity scores, but it shows the worst score in appearance correlation because the outputs are almost unchanged from the inputs except for the overall color.
>> WCT2는 현실성과 구조 유사성 점수에서 가장 높은 점수를 보입니다. 그러나 전반적인 색상을 제외하고 출력물은 입력물과 거의 변하지 않기 때문에 외모 상관관계 점수에서 가장 나쁜 점수를 보입니다.

> Both the qualitative and subjective evaluations show the effectiveness of our proposed method.
>> 질적 및 주관적 평가 모두 우리의 제안 방법의 효과성을 보여줍니다.

> Following [16], we also adopt other pre-trained models to evaluate the result.
>> [16]을 따르면 다른 사전 학습 모델을 사용하여 결과를 평가합니다.

> We use the classifier pre-trained with the ImageNet dataset given by improved DDPM [32] and calculate the average classification loss.
>> 우리는 개선된 DDPM [32]에서 제공하는 ImageNet 데이터셋으로 사전 학습 된 분류기를 사용하고 평균 분류 손실을 계산합니다.

> We also apply Mask-RCNN pre-trained on the COCO dataset to detect the mask of the object of each method.
>> 또한 COCO 데이터셋에서 사전 학습 된 Mask-RCNN을 적용하여 각 방법의 객체 마스크를 감지합니다.

> The results are shown in Table III.
>> 결과는 테이블 III에 나와 있습니다.

> Our model achieves the lowest classification loss.
>> 우리의 모델은 가장 낮은 분류 손실을 달성합니다.

> At the same time, since Mask-RCNN is trained on out-of-distribution (OOD) data, the overall recall rate is quite low.
>> 동시에, Mask-RCNN은 분포 밖 (OOD) 데이터로 훈련되었기 때문에 전반적인 리콜 비율이 매우 낮습니다.

> Our model demonstrates the second-best performance after WCT2, but WCT2 only transforms the color for the whole image.
>> 우리의 모델은 WCT2 이후 두 번째로 좋은 성능을 보이며, WCT2는 전체 이미지에 대한 색 변환만 수행합니다.

> Besides, we calculate the color difference histogram (CDH) [43] between the result and appearance image for each method.
>> 또한, 우리는 각 방법에 대해 결과와 외모 이미지 사이의 색상 차이 히스토그램 (CDH) [43]를 계산합니다.

> Our method achieves better appearance similarity than image translation methods.
>> 우리의 방법은 이미지 전환 방법보다 더 나은 외모 유사성을 보입니다.

> Although NST methods like STROTSS have a better CDH, they tend to transfer the whole image with simple color transformation, as shown in Fig. 4.
>> STROTSS와 같은 NST 방법은 더 나은 CDH를 가지고 있지만, 그들은 간단한 색 변환과 함께 전체 이미지를 전송하는 경향이 있습니다.

![Figure-4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-4.PNG)

> Fig. 4. Comparison with other state-of-the-art (SOTA) methods. Our results show better performance in both appearance and structure similarity.
>> 그림 4. 다른 최첨단(SOTA) 방법과의 비교. 우리의 결과는 외관과 구조 유사성 모두에서 더 나은 성능을 보여준다.

![Figure-5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-5.PNG)

> Fig. 5. Illustration of Mask Generation by Label Condition.
>> 그림 5. 라벨 조건별 마스크 생성 그림.

![Figure-6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-6.PNG)

> Fig. 6. An example of fashion output with a generated messy mask. (a) and (b) are our results with and without mask guidance, respectively. (c) is the result of DiffuseIT.
>> 그림 6. 지저분한 마스크가 생성된 패션 출력의 예. (a)와 (b)는 각각 마스크 지침이 있는 결과와 없는 결과입니다. (c)는 DiffuseIT의 결과입니다.

![Figure-7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-7.PNG)

> Fig. 7. Examples that show the mask effectiveness. (a) and (b) show the results of our method with or without mask guidance, respectively
>> 그림 7. 마스크 효과를 나타내는 예. (a)와 (b)는 각각 마스크 지침이 있거나 없는 우리의 방법의 결과를 보여준다.

![Figure-8](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-8.PNG)

> Fig. 8. Comparison with label-conditional DiffuseIT. Our results and results from DiffuseIT with label-conditional diffusion models are shown in (a) and (b), respectively.
>> 그림 8. 라벨 조건부 확산과의 비교IT. Diffusion의 결과와 결과라벨 조건부 확산 모델이 있는 IT는 각각 (a)와 (b)에 나와 있다.

![Figure-9](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-03-13-(diffusion)difffashion/Figure-9.PNG)

> Fig. 9. Examples of the DiffuseIT model with the text guidance. “Handbag” to “Handbag with marine life pattern” and “Ocean style Handbag” are prompts for (a) and (b), respectively.
>> 그림 9. 확산의 예텍스트 안내가 포함된 IT 모델. '핸드백'에서 '해양생물 패턴 핸드백', '오션스타일 핸드백'은 각각 (a), (b)의 프롬프트다.

## E. Ablation Study

> In order to verify the effectiveness of the method, we study the individual components of our technical designs through several ablation studies as illustrated from Fig. 5 to Fig. 9.
>> 우리는 그림 5에서 그림 9까지 설명된 것처럼 몇 가지 축소 연구를 통해 기술 디자인의 개별 구성 요소를 연구하여 방법의 효과성을 검증합니다.

> Mask Generation: We randomly select several bag images with backgrounds from ImageNet and our dataset.
>> 마스크 생성: 우리는 ImageNet과 우리의 데이터셋에서 배경과 함께 몇 가지 가방 이미지를 무작위로 선택합니다.

> We keep the same experimental setup as Section V-B and show the masks in Fig. 5.
>> 우리는 섹션 V-B와 동일한 실험 설정을 유지하고, 그림 5에 마스크를 보여줍니다.

> For most images, it can generate a foreground object mask that is suitable for our models.
>> 대부분의 이미지에 대해, 우리의 모델에 적합한 전경 객체 마스크를 생성할 수 있습니다.

> Due to the randomness of diffusion, in the last column, we show the scene where the mask is not good enough.
>> 확산의 무작위성으로 인해 마지막 열에서는 마스크가 충분히 좋지 않은 경우의 장면을 보여줍니다.

> But even so, our model still outperforms other models, as shown in Fig. 6.
>> 그러나 그래도 우리의 모델은 다른 모델보다 우수한 결과를 보여줍니다 (그림 6 참조).

> Mask Guidance: We conduct an experiment on our model without the mask guidance part, as shown in Fig. 7.
>> 마스크 가이드: 우리는 마스크 가이드 부분이 없는 모델에 대한 실험을 수행합니다 (그림 7 참조).

> Fig. 7(a) shows the result without mask guidance and Fig. 7(b) presents the outputs of our model with mask guidance.
>> 그림 7(a)는 마스크 가이드 없이 결과를 보여주고, 그림 7(b)는 마스크 가이드를 사용하여 우리의 모델의 출력을 나타냅니다.

> Without mask guidance, in many images, the structure of the bag is destroyed during diffusion.
마스크 가이드 없이 많은 이미지에서, 확산 도중 가방의 구조가 파괴됩니다.

> In the last row of the figure, we show that for some images, using a mask may reduce the correlation of appearance, but this is still enough to complete the transfer task.
>> 그림 7(a)는 마스크 가이드 없이 결과를 보여주고, 그림 7(b)는 마스크 가이드를 사용하여 우리의 모델의 출력을 나타냅니다.

> In order to solve a small number of such problems, we set the probability of 0.2 when applying without using mask guidance.
>> 마스크 가이드 없이 많은 이미지에서, 확산 도중 가방의 구조가 파괴됩니다.

> Label-Condition: Because our model uses the diffusion model with label-condition, for a fair comparison, we replace the diffusion model of diffuseIT with the same model as ours and use the label “bag” for the condition in the denoising stage.
>> 마지막 행에서는 일부 이미지의 경우 마스크 사용이 외모의 상관관계를 감소시킬 수 있지만 여전히 전송 작업을 완료하는 데 충분합니다.

> Fig. 8(a) shows the results of DiffuseIT with label condition, and Fig. 8(b) presents our method.
>> 이러한 문제를 해결하기 위해, 마스크 가이드를 사용하지 않고 적용할 때 0.2의 확률을 설정합니다.

> Our method still shows better results in structure preservation, appearance similarity, and authenticity.
>> 라벨 조건: 우리의 모델은 라벨 조건이 있는 확산 모델을 사용하므로, 공정한 비교를 위해 diffuseIT의 확산 모델을 우리와 동일한 모델로 대체하고 노이즈 제거 단계에서 조건으로 "가방"을 사용합니다.

> In addition, We show some results of a multi-modal guided diffusion model trained on the same amount of data.
>> 그림 8(a)는 라벨 조건이 있는 DiffuseIT의 결과를 보여주며, 그림 8(b)는 우리의 방법을 나타냅니다.

> Fig. 9 shows the result of DiffuseIT with the text guidance.
>> 우리의 방법은 여전히 구조 보존, 외모 유사성, 진실성에서 더 나은 결과를 보입니다.

> “Handbag” to “Handbag with marine life pattern” and “Ocean style Handbag” are prompts for (a) and (b), respectively.
>> 그림 9는 "핸드백"에서 "해양 생물 무늬 핸드백"으로, "오션 스타일 핸드백"으로 각각 프롬프트를 설정한 DiffuseIT의 결과를 보여줍니다.

> We can see that a text-guided model cannot complete the task well.
>> 그림에서 볼 수 있듯이, 텍스트 가이드 모델은 작업을 완료하는 데 실패합니다.

## VI. CONCLUSION AND FUTURE WORK

> We tackle a new problem set in the context of fashion design: designing new clothing fashion from a given clothing image and a natural appearance image, and keeping the structure of the clothing with a similar appearance to the natural image.
>> 우리는 패션 디자인의 문맥에서 새로운 문제를 다루고 있습니다: 주어진 의류 이미지와 자연적인 외모 이미지로부터 새로운 의류 패션을 디자인하면서 의류의 구조를 유지하고 자연 이미지와 유사한 외모를 유지하는 것입니다.

> We propose a novel diffusion-based image-to-image translation framework by swapping the input latent with structure transfer.
>> 우리는 입력 latent와 구조 전송을 교환하여 새로운 확산 기반 이미지 변환 프레임워크를 제안합니다.

> And the model is guided by an automatically generated foreground mask and both structure and appearance information from the pre-trained DINO-ViT model.
>> 그리고 모델은 자동으로 생성된 전경 마스크와 사전 훈련 된 DINO-ViT 모델에서 구조 및 외모 정보 모두에 의해 가이드됩니다.

> The experimental results have shown that our proposed method outperforms most baselines, demonstrating that our method can better balance authenticity and structure preservation while also achieving appearance migration.
>> 실험 결과, 우리가 제안한 방법은 대부분의 기준을 능가하여, 우리의 방법이 진실성과 구조 보존을 더 잘 균형잡히게 유지하면서 외모 이전을 달성할 수 있음을 보여주고 있습니다.

> Due to the randomness of diffusion, the mask cannot guarantee good results every time.
>> 확산의 무작위성 때문에, 마스크는 항상 좋은 결과를 보장할 수 없습니다.

> In the future, we will try to constrain the diffusion model using the information condition of other modalities to generate better masks.
>> 앞으로, 우리는 다른 모달리티의 정보 조건으로 확산 모델을 제한하여 더 나은 마스크를 생성하도록 시도할 것입니다.

## ACKNOWLEDGMENTS

> This work is supported by National Natural Science Foundation of China (62106219) and Natural Science Foundation of Zhejiang Province (QY19E050003).
>> 이 연구는 중국 국가 자연과학 재단 (62106219)과 절강성 자연과학 재단 (QY19E050003)의 지원을 받았습니다.