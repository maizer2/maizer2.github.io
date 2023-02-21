---
layout: post 
title: "(Diffusion)High-Resolution Image Synthesis with Latent Diffusion Models"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review]
---

### [Diffusion Literature List](https://maizer2.github.io/1.%20computer%20engineering/2023/02/01/Literature-of-diffusion.html)

# High-Resolution Image Synthesis with Latent Diffusion Models

## Abstract

> By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. 
>> 이미지 형성 프로세스(image formation process)를 노이즈 제거 자동 인코더(denoising autoencoders)의 순차적 응용 프로그램(sequential application)으로 분해하여(decomposing) 확산 모델(DM)은 이미지 데이터 및 그 이상(beyond)에서 최첨단(state-of-the-art) 합성 결과를 달성한다.

> Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. 
>> 또한, 그들의 공식화는 재교육(retraining) 없이 이미지 생성 프로세스를 제어하는 안내 메커니즘(guiding mechanism)을 허용한다.

> However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. 
>> 그러나 이러한 모델은 일반적으로 픽셀 공간(pixel space)에서 직접 작동하기 때문에, 강력한 DM의 최적화는 종종 수백 일의 GPU를 소모하며 순차적 평가(sequential evaluations)로 인해 추론(inference) 비용이 많이 든다.

> To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. 
>> 품질과 유연성(quality and flexibility)을 유지하면서 제한된 계산 자원(limited computational resources)에 대한 DM 훈련을 가능하게 하기 위해, 우리는 그것들을 강력한 사전 훈련된 자동 인코더(pretrained autoencoders)의 잠재 공간(the latent space)에 적용한다.

> In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. 
>> 이전 연구와 달리, 그러한 표현에 대한 확산 모델(diffusion models)을 훈련하면 처음으로 복잡성 감소(complexity reduction)와 세부 보존(detail preservation) 사이에서 거의 최적의 지점(a near-optimal point)에 도달할 수 있어, 시각적 충실도(visual fidelity)가 크게 향상된다.

> By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. 
>> 교차 주의(cross-attention) 계층을 모델 아키텍처에 도입함으로써 확산 모델(diffusion models)을 텍스트 또는 경계 상자(bounding boxes)와 같은 일반적인 조건 입력(conditioning inputs)을 위한 강력하고 유연한 생성기로 전환하고 고해상도 합성(high-resolution synthesis)이 컨볼루션 방식(convolutional manner)으로 가능해진다.

> Our latent diffusion models (LDMs) achieve new state-of-the-art scores for image inpainting and class-conditional image synthesis and highly competitive performance on various tasks, including text-to-image synthesis, unconditional image generation and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.
>> 우리의 잠재 확산 모델(latent diffusion models)(LDM)은 픽셀 기반DMs(pixel-based DMs)에 비해 계산 요구 사항을 크게 줄이면서 이미지 인페인팅(image inpainting)및 클래스 조건부 이미지 합성(class-conditional image)을 위한 새로운 점수와 텍스트-이미지 합성(text-to-image synthesis) , 무조건 이미지 생성(unconditional image generation) 및 초해상도(super-resolution)를 포함한 다양한 작업에서 매우 경쟁력 있는 성능을 달성한다.

1. Introduction

> Image synthesis is one of the computer vision fields with the most spectacular recent development, but also among those with the greatest computational demands. 
>> 이미지 합성(Image synthesis)은 최근에 가장 눈부신 발전을 이룬 컴퓨터 비전(computer vision) 분야 중 하나이지만, 가장 큰 계산 수요(computational demands)를 가진 분야 중 하나이기도 하다. 

> Especially high-resolution synthesis of complex, natural scenes is presently dominated by scaling up likelihood-based models, potentially containing billions of parameters in autoregressive (AR) transformers [66,67]. 
>> 특히 복잡하고 자연스러운 장면의 고해상도 합성(high-resolution synthesis of complex, natural scenes)은 현재 가능성 기반 모델의 스케일업(likelihood-based models scaling up)에 의해 지배되고 있으며, 잠재적으로 자기 회귀(autoregressive)(AR) 변압기(transformers)에 수십억 개의 매개 변수(billions of parameters)가 포함되어 있다[66,67].

> In contrast, the promising results of GANs [3, 27, 40] have been revealed to be mostly confined to data with comparably limited variability as their adversarial learning procedure does not easily scale to modeling complex, multi-modal distributions. 
>> 대조적으로, GAN의 유망한 결과[3, 27, 40]는 적대적 학습(adversarial learning) 절차가 복잡한 멀티-모달 분포(multi-modal distributions)를 모델링하는 것으로 쉽게 확장되지 않기 때문에 비교적(comparably) 가변성이 제한된(limited variability) 데이터에 대부분 국한된(confined) 것으로 밝혀졌다(revealed).

> Recently, diffusion models [82], which are built from a hierarchy of denoising autoencoders, have shown  to achieve impressive results in image synthesis [30,85] and beyond [7,45,48,57], and define the state-of-the-art in class-conditional image synthesis [15,31] and super-resolution [72]. 
>> 최근 노이즈 제거 자동 인코더의 계층(hierarchy of denoising autoencoders) 구조로 구축된 확산 모델(diffusion models)[82]은 이미지 합성(image synthesis)[30,85] 및 그 이상[7,45,48,57]에서 인상적인 결과를 달성하고 클래스 조건부 이미지 합성(conditional image synthesis)[15,31] 및 초해상도(super-resolution)[72]에서 최첨단(state-of-the-art)을 정의하는 것으로 나타났다.

> Moreover, even unconditional DMs can readily be applied to tasks such as inpainting and colorization [85] or stroke-based synthesis [53], in contrast to other types of generative models [19,46,69].
>> 또한, 다른 유형의 생성 모델(other types of generative models)[19,46,69]과는 달리 무조건적인 DM(unconditional DMs)조차도 인페인팅(inpainting) 및 컬러화(colorization)[85] 또는 스트로크 기반 합성(stroke-based synthesis)[53]과 같은 작업에 쉽게 적용할 수 있다.

> Being likelihood-based models, they do not exhibit mode-collapse and training instabilities as GANs and, by heavily exploiting parameter sharing, they can model highly complex distributions of natural images without involving billions of parameters as in AR models [67].
>> 가능성 기반 모델(likelihood-based models)이기 때문에, 그들은 GAN으로서 모드 붕괴(mode-collapse)와 훈련 불안정성(training instabilities)을 나타내지 않으며, 매개 변수 공유를 크게 활용함(heavily exploiting parameter sharing)으로써, AR 모델에서와 같이 수십억 개의 매개 변수를 포함하지 않고 자연 이미지의 매우 복잡한 분포(highly complex distributions of natural images)를 모델링할 수 있다[67].

> Democratizing High-Resolution Image Synthesis DMs belong to the class of likelihood-based models, whose mode-covering behavior makes them prone to spend excessive amounts of capacity (and thus compute resources) on modeling imperceptible details of the data [16, 73]. 
>> 고해상도 이미지 합성 민주화 DMs(Democratizing High-Resolution Image Synthesis DMs)은 가능성 기반 모델(likelihood-based models)에 속하며, 모드 커버 동작(mode-covering behavior)으로 인해 데이터의 감지할 수 없는(imperceptible) 세부 사항을 모델링하는 데 과도한 용량(따라서 계산 리소스)을 소비하기 쉽다[16, 73]. 

> Although the reweighted variational objective [30] aims to address this by undersampling the initial denoising steps, DMs are still computationally demanding, since training and evaluating such a model requires repeated function evaluations (and gradient computations) in the high-dimensional space of RGB images.
>> 재조정된 변형 목표(eweighted variational objective)[30]는 초기 노이즈 제거 단계(initial denoising steps)를 과소 샘플링(undersampling)하여 이를 해결하는 것을 목표로 하지만(aims to address), 이러한 모델을 훈련하고 평가하려면 RGB 이미지의 고차원 공간(high-dimensional space of RGB images)에서 반복적인 함수 평가(및 그레이디언트 계산(gradient computations))가 필요하기 때문에 DMs은 여전히 계산적으로 요구된다. 

> As an example, training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - 1000 V100 days in [15]) and repeated evaluations on a noisy version of the input space render also inference expensive, so that producing 50k samples takes approximately 5 days [15] on a single A100 GPU. 
>> 예를 들어, 가장 강력한 DMs을 훈련하는 데는 수백 일(예: 하루에 150 - 1000개의 V100[15])이 걸리는 경우가 많으며, 노이즈가 많은 버전의 입력 공간 렌더링(input space render)에 대한 반복적인 평가(repeated evaluations)도 비용이 많이 들기 때문에 단일 A100 GPU에서 50k 샘플을 생산하는 데 약 5일[15]이 걸린다. 

> This has two consequences for the research community and users in general: Firstly, training such a model requires massive computational resources only available to a small fraction of the field, and leaves a huge carbon footprint [65, 86]. 
>> 이는 일반적으로 연구 커뮤니티와 사용자에게 두 가지 결과를 초래한다: 첫째, 그러한 모델을 훈련하는 데는 현장의 극히 일부에서만 이용할 수 있는 방대한 계산 자원이 필요하며, 막대한 탄소 발자국(carbon  footprint)을 남긴다[65, 86]. 

> Secondly, evaluating an already trained model is also expensive in time and memory, since the same model architecture must run sequentially for a large number of steps (e.g. 25 - 1000 steps in [15]).
>> 둘째로, 이미 훈련된 모델(evaluating an already trained model)을 평가하는 것은 동일한 모델 아키텍처가 많은 단계(예: [15]에서 25 - 1000 단계) 동안 순차적으로 실행되어야 하기 때문에 시간과 메모리 면에서도 비용이 많이 든다.

> To increase the accessibility of this powerful model class and at the same time reduce its significant resource consumption, a method is needed that reduces the computational complexity for both training and sampling. 
>> 이 강력한 모델 클래스의 접근성(accessibility)을 높이고 동시에(at the same time) 상당한 리소스 소비(significant resource consumption)를 줄이기 위해서는 훈련과 샘플링 모두에 대한 계산 복잡성(complexity)을 줄이는 방법이 필요하다. 

> Reducing the computational demands of DMs without impairing their performance is, therefore, key to enhance their accessibility.
>> 따라서 성능을 손상(impairing)시키지 않고 DM의 계산 요구를 줄이는 것이 접근성을 향상시키는(enhance their accessibility) 핵심(key)이다.

> Departure to Latent Space: Our approach starts with the analysis of already trained diffusion models in pixel space: Fig. 2 shows the rate-distortion trade-off of a trained model. 
>> 잠재 공간으로의 출발: 우리의 접근 방식은 픽셀 공간(pixel space)에서 이미 훈련된 확산 모델(diffusion models)의 분석으로 시작한다. 그림 2는 훈련된 모델의 속도-왜곡 트레이드오프(rate-distortion trade-off)를 보여준다. 

> As with any likelihood-based model, learning can be roughly divided into two stages: First is a perceptual compression stage which removes high-frequency details but still learns little semantic variation. In the second stage, the actual generative model learns the semantic and conceptual composition of the data (semantic compression). 
>> 다른 가능성 기반 모델과 마찬가지로 학습은 크게(roughly)  두 단계로 나눌 수 있다(divided into two stages): 첫 번째는 고주파 세부 사항(high-frequency details)을 제거하지만 여전히 의미론적 변화(semantic variation)를 거의(little) 학습하지 않는 지각 압축 단계(perceptual compression stage)이다. 두 번째 단계에서는 실제 생성 모델(actual generative model)이 데이터의 의미론적, 개념적 구성(semantic and conceptual composition)(의미론적 압축(semantic compression))을 학습한다. 

> We thus aim to first find a perceptually equivalent, but computationally more suitable space, in which we will train diffusion models for high-resolution image synthesis.
>> 따라서 우리는 먼저 고해상도 이미지 합성을 위한 확산 모델(diffusion models)을 훈련하는 지각적으로 동등(perceptually equivalent)하지만, 계산적으로 더 적합한 공간(computationally more suitable space)을 찾는 것을 목표로 한다.

> Following common practice [11, 23, 66, 67, 96], we separate training into two distinct phases: First, we train an autoencoder which provides a lower-dimensional (and thereby efficient) representational space which is perceptually equivalent to the data space. 
>> 일반적인 관행(common practice)[11, 23, 66, 67, 96]에 따라 우리는 훈련을 두 개의 뚜렷한 단계(distinct phases)로 분리한다: 먼저, 우리는 데이터 공간과 지각적으로 동일한(perceptually equivalent) 저차원(lower-dimensional)(따라서(and thereby) 효율적인) 표현 공간(representational space)을 제공하는 자동 인코더(autoencoder)를 훈련시킨다. 

> Importantly, and in contrast to previous work [23,66], we do not need to rely on excessive spatial compression, as we train DMs in the learned latent space, which exhibits better scaling properties with respect to the spatial dimensionality. 
>> 중요한 것은 이전 연구[23,66]와 달리(in contrast to), 우리는 공간 차원성과 관련하여 더 나은 스케일링 특성(scaling properties)을 나타내는(exhibits) 학습된 잠재 공간(learned latent space)에서 DM을 훈련하기 때문에 과도한 공간 압축에 의존할 필요가 없다는 것이다. 

> The reduced complexity also provides efficient image generation from the latent space with a single network pass. We dub the resulting model class Latent Diffusion Models (LDMs).
>> 복잡성 감소(reduced complexity)는 또한 단일 네트워크 패스로 잠재 공간에서 효율적인 이미지 생성(efficient image generation)을 제공한다. 우리는 결과 모델 클래스(the resulting model class)를 잠재 확산 모델(Latent Diffusion Models)(LDMs)라고 칭한다.

> A notable advantage of this approach is that we need to train the universal autoencoding stage only once and can therefore reuse it for multiple DM trainings or to explore possibly completely different tasks [81].
>> 이 접근 방식의 주목할 만한(notable) 장점은 범용 자동 인코딩 단계(the universal autoencoding stage)를 한 번만(only once) 훈련하면 되므로 여러 DM 훈련에 재사용하거나 가능한(possibly) 완전히(completely) 다른 작업을 탐색할 수 있다는 것이다[81]. 

> This enables efficient exploration of a large number of diffusion models for various image-to-image and text-to-image tasks. 
>> 이를 통해 다양한 이미지 대 이미지(image-to-image) 작업 및 텍스트 대 이미지(text-to-image) 작업을 위한 많은 확산 모델(diffusion models)을 효율적으로 탐색(efficient exploration)할 수 있다. 

> For the latter, we design an architecture that connects transformers to the DM’s UNet backbone [71] and enables arbitrary types of token-based conditioning mechanisms, see Sec. 3.3.
>> 후자의 경우(For the latter), 우리는 트랜스포머(transformers)를 DM의 UNet 백본(backbone)[71]에 연결하고 임의의 유형의 토큰 기반 조건화 메커니즘(token-based conditioning mechanisms)을 가능하게 하는 아키텍처를 설계한다(3.3절 참조).

> In sum, our work makes the following 
>> 요약하면, 우리의 작업은 다음을 만든다 

> 기여(ontributions):

> (i) In contrast to purely transformer-based approaches [23, 66], our method scales more graceful to higher dimensional data and can thus (a) work on a compression level which provides more faithful and detailed reconstructions than previous work (see Fig. 1) and (b) can be efficiently applied to high-resolution synthesis of megapixel images.
>> (i) 순수한 변압기 기반 접근 방식(transformer-based approaches)[23, 66]과는 달리(In contrast to), 우리의 방법은 더 우아하게(more graceful) 고차원 데이터(higher dimensional data)로 확장(scales)할 수 있으므로 (a) 이전 작업보다 더 충실하고(faithful) 상세한 재구성(detailed reconstructions)을 제공하는 압축 수준에서 작업할 수 있으며 (see Fig. 1) (b) 메가픽셀 이미지의 고해상도 합성(high-resolution synthesis)에 효율적으로 적용(efficiently applied)할 수 있다.

> (ii) We achieve competitive performance on multiple tasks (unconditional image synthesis(), inpainting, stochastic super-resolution) and datasets while significantly lowering computational costs. Compared to pixel-based diffusion approaches, we also significantly decrease inference costs.
>> (ii) 우리는 계산 비용(computational costs)을 크게(significantly) 낮추면서(lowering) 여러 작업(무조건(unconditional) 이미지 합성, 인페인팅(inpainting), 확률적 초해상도(stochastic super-resolution)) 및 데이터 세트에서 경쟁력(competitive) 있는 성능을 달성한다. 픽셀 기반 확산 접근법(pixel-based diffusion approaches)에 비해(Compared to) 추론 비용도 크게 절감(significantly decrease)한다.

> (iii) We show that, in contrast to previous work [93] which learns both an encoder/decoder architecture and a score-based prior simultaneously, our approach does not require a delicate weighting of reconstruction and generative abilities. This ensures extremely faithful reconstructions and requires very little regularization of the latent space.
>> (iii) 인코더/디코더 아키텍처(encoder/decoder architecture)와 점수 기반 prior(a score-based prior)를 동시에 학습하는 이전 연구[93]와 달리, 우리의 접근 방식은 재구성(reconstruction) 및 생성 능력(generative abilities)에 대한 섬세한(delicate) 가중치를 요구하지 않는다는 것을 보여준다. 이는 매우 충실한(faithful) 재구성(reconstructions)을 보장하고 잠재 공간의 정규화(regularization)를 거의 요구하지 않는다.

> (iv) We find that for densely conditioned tasks such as super-resolution, inpainting and semantic synthesis, our model can be applied in a convolutional fashion and render large, consistent images of $∼ 1024^{2}$ px.
>> (iv) 우리는 초해상도, 인페인팅 및 의미론적 합성(semantic synthesis)과 같은 조밀하게 조건화된 작업(densely conditioned tasks)의 경우, 우리의 모델이 컨볼루션 방식으로 적용될 수 있고 $∼ 1024^{2}$px의 크고 일관된 이미지를 렌더링할 수 있다는 것을 발견했다.

> (v) Moreover, we design a general-purpose conditioning mechanism based on cross-attention, enabling multi-modal training. We use it to train class-conditional, text-to-image and layout-to-image models.
>> (v) 또한 교차 주의(cross-attention)를 기반으로 범용 조건화 메커니즘(general-purpose conditioning mechanism)을 설계하여 다중 모드(multi-modal) 훈련을 가능하게 한다. 우리는 그것을 클래스 조건부(class-conditional), 텍스트 대 이미지(text-to-image) 및 레이아웃 대 이미지(layout-to-image) 모델을 훈련하는 데 사용한다.

> (vi) Finally, we release pretrained latent diffusion and autoencoding models at "https://github.com/CompVis/latent-diffusion" which might be reusable for a various tasks besides training of DMs [81].
>> (vi) 마지막으로, 우리는 DM의 훈련 외에도 다양한 작업에 재사용될 수 있는 "https://github.com/CompVis/lent-diffusion"에서 사전 훈련된 잠재 확산(pretrained latent diffusion) 및 자동 인코딩(autoencoding) 모델을 출시한다[81].

2. Related Work