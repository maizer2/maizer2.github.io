---
layout: post 
title: "(GAN)Analyzing and Improving the Image Quality of StyleGAN"
categories: [1. Computer Engineering]
tags: [1.0. Paper Review, 1.2.2.1. Computer Vision]
---

### [CV Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/paper-of-GAN.html)

# Analyzing and Improving the Image Quality of StyleGAN

## Abstract

> The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven unconditional generative image modeling. 
>> The style-based GAN architecture(StyleGAN)는 data-driven unconditional generative image modeling에서 state-of-the-art 결과를 산출한다. 

> We expose and analyze several of its characteristic artifacts, and propose changes in both model architecture and training methods to address them.
>> 우리는 그것의 characteristic artifacts 중 몇 가지를 노출하고 분석하고, 이를 해결하기 위해 모델 아키텍처와 교육 방법 모두의 변경을 제안한다.

> In particular, we redesign the generator normalization, revisit progressive growing, and regularize the generator to encourage good conditioning in the mapping from latent codes to images. 
>> 특히, 우리는 from latent codes to images의 매핑에서 좋은 조건을 장려하기 위해 generator normalization를 재설계하고, progressive growing을 재검토하고, regularize the generator한다. 

> In addition to improving image quality, this path length regularizer yields the additional benefit that the generator becomes significantly easier to invert. 
>> 이 path length regularizer는 이미지 품질을 향상시킬 뿐만 아니라 generator가 invert(반전)하기가 훨씬 쉬워지는 추가적인 이점을 제공합니다. 

> This makes it possible to reliably attribute a generated image to a particular network. 
>> 이를 통해 생성된 이미지를 특정 네트워크에 확실하게 attribute(포함) 할 수 있다. 

> We furthermore visualize how well the generator utilizes its output resolution, and identify a capacity problem, motivating us to train larger models for additional quality improvements.
>> 우리는 또한 generator가 출력 해상도를 얼마나 잘 활용하는지 시각화하고 capacity(용량) 문제를 식별하여 추가 품질 개선을 위해 더 큰 모델을 훈련하도록 동기를 부여한다.

> Overall, our improved model redefines the state of the art in unconditional image modeling, both in terms of existing distribution quality metrics as well as perceived image quality.
>> 전반적으로, 우리의 개선된 모델은 기존의 distribution quality metrics뿐만 아니라 perceived(관찰된) image quality 측면에서 unconditional image modeling의 state of the art을 재정의한다.

## 1. Introduction

> The resolution and quality of images produced by generative methods, especially generative adversarial networks (GAN) [16], are improving rapidly [23, 31, 5]. 
>> 생성 방법, 특히 생성적 적대 네트워크(GAN)[16]에 의해 생성된 이미지의 해상도와 품질이 빠르게 향상되고 있다[23, 31, 5]. 

> The current state-of-the-art method for high-resolution image synthesis is StyleGAN [24], which has been shown to work reliably on a variety of datasets. 
>> 현재 high-resolution image synthesis을 위한 최첨단 방법은 StyleGAN[24], 이것은 다양한 데이터 세트에서 reliably(안정적)으로 작동하는 것으로 나타났다. 

> Our work focuses on fixing its characteristic artifacts and improving the result quality further.
>> 우리의 작업은 characteristic(특징적인) artifacts를 수정하고 결과 품질을 더욱 향상시키는 데 중점을 둔다.

> The distinguishing feature of StyleGAN [24] is its unconventional generator architecture.
>> StyleGAN[24]의 특징적인 feature는 unconventional(전통적이지 않은) generator architecture이다.

> Instead of feeding the input latent code z ∈ Z only to the beginning of a the network, the mapping network f first transforms it to an intermediate latent code w ∈ W. 
>> mapping network f는 input latent code z ∈ Z를 네트워크의 시작에만 공급하는 대신, 먼저 intermediate latent code w ∈ W로 변환한다. 

> Affine transforms then produce styles that control the layers of the synthesis network g via adaptive instance normalization (AdaIN) [21, 9, 13, 8].
>> 그런 다음 Affine transforms은 adaptive instance normalization(AdaIN) [21, 9, 13, 8]를 통해 synthesis network g의 레이어를 제어하여 스타일을 생성한다.

> Additionally, stochastic variation is facilitated by providing additional random noise maps to the synthesis network. 
>> 또한, stochastic variation은 synthesis network에 additional random noise maps을 제공함으로서 facilitated(촉진된다). 

> It has been demonstrated [24, 38] that this design allows the intermediate latent space W to be much less entangled than the input latent space Z. 
>> 이 설계를 통해 intermediate latent space W가 input latent space Z보다 훨씬 less entangled(덜 얽힐) 수 있음이 [24, 38] has been demonstrated(입증되었다). 

> In this paper, we focus all analysis solely on W, as it is the relevant latent space from the synthesis network’s point of view.
>> 본 논문에서는 synthesis network의 point of view(관점)에서 relevant latent space이기 때문에 모든 분석을 W에만 집중한다.

> Many observers have noticed characteristic artifacts in images generated by StyleGAN [3]. 
>> 많은 관찰자들이 StyleGAN[3]에 의해 생성된 이미지에서 characteristic artifacts를 발견했습니다. 

> We identify two causes for these artifacts, and describe changes in architecture and training methods that eliminate them. 
>> 우리는 이러한 artifacts에 대한 두 가지 원인을 identify(확인)하고, 이를 제거하는 architecture 및 training methods의 변화를 설명한다. 

> First, we investigate the origin of common blob-like artifacts, and find that the generator creates them to circumvent a design flaw in its architecture. 
>> First, 우리는 common blob-like artifacts의 origin(기원)을 investigate(조사)하고, 생성기가 architecture의 설계 flaw(결함)을 circumvent(피하기) 위해 이를 생성한다는 것을 발견한다. 

> In Section 2, we redesign the normalization used in the generator, which removes the artifacts.
>> In Section2, normalization used in the generator를 재설계하여, artifacts를 제거합니다.

> Second, we analyze artifacts related to progressive growing [23] that has been highly successful in stabilizing high-resolution GAN training. 
>> Second, high-resolution GAN training을 stabilizing(안정화)하는 데 has been highly successful(크게 성공한) progressive growing[23]과 관련된 artifacts를 분석한다. 

> We propose an alternative design that achieves the same goal — training starts by focusing on low-resolution images and then progressively shifts focus to higher and higher resolutions — without changing the network topology during training. 
>> 훈련 중 network topology를 변경하지 않고 low-resolution images에 초점을 맞추는 훈련을 시작한 다음 점점 더 higher resolutions로 progressively(점진적으로) shifts focus하는 동일한 목표를 달성하는 alternative(양자 택일 가능한) design를 제안한다. 

> This new design also allows us to reason about the effective resolution of the generated images, which turns out to be lower than expected, motivating a capacity increase (Section 4).
>> 또한 이 새로운 설계를 통해 effective resolution of the generated images에 대해 allows us to reason(추론할 수 있으며), 이는 lower than expected(예상보다 낮은 것)으로 turns out(밝혀져), capacity increase에 motivating(동기를 부여)한다(Section 4).

> Quantitative analysis of the quality of images produced using generative methods continues to be a challenging topic. 
>> Generative methods을 사용하여 생성된 quality of images에 대한 quantitative analysis은 계속해서 어려운 주제이다. 

> Frechet inception distance (FID) [20] measures differences in the density of two distributions in the highdimensional feature space of an InceptionV3 classifier [39].
>> Frechet inception distance (FID) [20]는 inceptionV3 classifier [39]의 highdimensional feature space에서 differences in the density of two distributions(두 분포에서의 밀도 차이)를 measures(측정한다).

> Precision and Recall (P&R) [36, 27] provide additional visibility by explicitly quantifying the percentage of generated images that are similar to training data and the percentage of training data that can be generated, respectively. 
>> Precision and Recall (P&R) [36, 27]은 각각 training data와 유사한 percentage of generated images과  can be generated(생성될 수 있는) percentage of training data을 explicitly(명시적)으로 quantifying(정량화)하여 additional visibility을 제공한다. 

> We use these metrics to quantify the improvements.
>> 우리는 quantify the improvements(개선 사항을 정량화)하기 위해 이러한 메트릭을 사용한다.

> Both FID and P&R are based on classifier networks that have recently been shown to focus on textures rather than shapes [12], and consequently, the metrics do not accurately capture all aspects of image quality. 
>> FID와 P&R 모두 최근에 shapes(모양)보다는 textures(질감)에 초점을 맞춘 것으로 나타난 classifier networks를 based on(기반으로) 하며[12], consequently(결과적으로) metrics는 이미지 품질의 모든 aspects(측면)을 accurately(정확하게) capture하지 못한다. 

> We observe that the perceptual path length (PPL) metric [24], originally introduced as a method for estimating the quality of latent space interpolations, correlates with consistency and stability of shapes. 
>> 우리는 원래 latent space interpolations의 quality를 estimating(추정)하는 방법으로 도입된 perceptual path length (PPL) metric [24]이 shapes의 consistency(일관성) 및 stability(안정성)과 관련이 있음을 관찰한다. 

> Based on this, we regularize the synthesis network to favor smooth mappings (Section 3) and achieve a clear improvement in quality. 
>> 이를 기반으로, 우리는 favor smooth mappings하도록 synthesis network를 regularize하고(Section 3) quality의 clear(명확한) 개선을 달성한다. 

> To counter its computational expense, we also propose executing all regularizations less frequently, observing that this can be done without compromising effectiveness.
>> computational expense(비용)에 To counter(대응하기 위해), all regularizations를 less frequently(덜 자주) executing(실행)하여 효율성을 저하시키지 않고 수행 가능 하다는 것을 관찰할 것을 제안한다.

> Finally, we find that projection of images to the latent space W works significantly better with the new, pathlength regularized StyleGAN2 generator than with the original StyleGAN. 
>> Finally, 우리는 latent space W에 대한 projection of images(이미지 투영)이 original StyleGAN보다 새로운 pathlength regularized StyleGAN2 generator에서 significantly better(특히 잘) 작동한다는 것을 발견했다. 

> This makes it easier to attribute a generated image to its source (Section 5).
>> 이렇게 하면 generated image를 소스에 easier(쉽게) attribute할 수 있습니다(Section 5).

[Figure 1]()

> Figure 1. Instance normalization causes water droplet-like artifacts in StyleGAN images. These are not always obvious in the generated images, but if we look at the activations inside the generator network, the problem is always there, in all feature maps starting from the 64x64 resolution. It is a systemic problem that plagues all StyleGAN images.
>> Figure 1. Instance normalization로 인해 StyleGAN images에서 water droplet-like artifacts가 발생합니다. 생성된 이미지에서 항상 obvious(두드러 지는 것)은 아니지만 generator network 내부의 activations를 보면64x64 resolution에서 시작하는 feature maps에서 항상 문제가 발생합니다. 그것은 모든 StyleGAN images를 괴롭히는 systemic(체계적인) problem이다.

## 2. Removing normalization artifacts

> We begin by observing that most images generated by StyleGAN exhibit characteristic blob-shaped artifacts that resemble water droplets. 
>> 먼저 most images generated by StyleGAN은 water droplets(물방울)과 유사한 characteristic blob-shaped artifacts를 보여줌에서부터 관찰을 시작했다. 

> As shown in Figure 1, even when the droplet may not be obvious in the final image, it is present in the intermediate feature maps of the generator.
>> Figure 1에서 보는 바와 같이, 최종 image에서 droplet(물방울)이 명확하지 않은 경우에도, generator의 intermediate feature maps에 존재한다.

> The anomaly starts to appear around 64×64 resolution, is present in all feature maps, and becomes progressively stronger at higher resolutions. 
>> 이 anomaly(변칙)은 64×64 resolution 부근에서 나타나기 시작하며, all feature maps에 존재하며, higher resolutions에서 progressively stronger해진다. 

> The existence of such a consistent artifact is puzzling, as the discriminator should be able to detect it.
>> 이렇게 such a consistent artifact의 existence(존재)는 discriminator가 그것을 detect(감지)할 수 있어야 하기 as(때문에) is puzzling(곤혹스럽다).

> We pinpoint the problem to the AdaIN operation that normalizes the mean and variance of each feature map separately, thereby potentially destroying any information found in the magnitudes of the features relative to each other.
>> 우리는 each feature map의 mean and variance을 separately(개별적)으로 normalizes하는 AdaIN operation이 문제임을 명시한다, 서로에게 relative(연관된) features의 magnitudes(해아릴 수 없는 양)에서 발견되는 any information를 potentially(잠재적으로) destroying한다.

> We hypothesize that the droplet artifact is a result of the generator intentionally sneaking signal strength information past instance normalization: by creating a strong, localized spike that dominates the statistics, the generator can effectively scale the signal as it likes elsewhere. 
>> 우리는 droplet artifact가 generator intentionally(의도적)로 signal strength(강도) information를 instance normalization를 통해 sneaking(몰래) 얻은 결과라고 가정한다: statistics(통계)를 dominates(지배)하는 강력한 localized spike를 생성함으로서 생성기는 elsewhere(다른 곳)에서 원하는 대로 signal(신호)를 effectively(효과적으로) scale(스케일링)할 수 있다. 

> Our hypothesis is supported by the finding that when the normalization step is removed from the generator, as detailed below, the droplet artifacts disappear completely.
>> 우리의 가설은 아래에 자세히 설명된 바와 같이 normalization step가 generator에서 제거되면 droplet artifacts가 완전히 사라진다는 발견에 의해 뒷받침된다.

> In rare cases (perhaps 0.1% of images) the droplet is missing, leading to severely corrupted images. See Appendix A for details.
>> rare cases(드문 경우)(이미지의 0.1%)에는 droplet이 누락되어 이미지가 심각하게 손상될 수 있습니다. 자세한 내용은 부록 A를 참조하십시오.

### 2.1. Generator architecture revisited 

> We will first revise several details of the StyleGAN generator to better facilitate our redesigned normalization.
>> 우리는 먼저 redesigned normalization를 더 잘 촉진하기 위해 styleGAN 생성기의 몇 가지 세부 사항을 수정할 것이다.

> These changes have either a neutral or small positive effect on their own in terms of quality metrics.
>> 이러한 변화는 in terms of quality metrics(품질 지표 측면)에서 그 자체로 neutral(중립적)이거나 small positive effect을 미친다.

> Figure 2 a shows the original StyleGAN synthesis network g [24], and in Figure 2b we expand the diagram to full detail by showing the weights and biases and breaking the AdaIN operation to its two constituent parts: normalization and modulation. 
>> 그림 2 a는 원래의 styleGAN 합성 네트워크 g[24]를 보여주고, 그림 2b에서 우리는 가중치와 편향을 보여주고 AdaIN 연산을 normalization와 modulation(조절)라는 두 가지 구성 부분으로 분해하여 다이어그램을 완전한 세부 사항으로 확장한다.

> This allows us to re-draw the conceptual gray boxes so that each box indicates the part of the network where one style is active (i.e., “style block”). 
>> 이를 통해 각 상자가 하나의 스타일이 활성화된 네트워크 부분(즉, "스타일 블록")을 나타내도록 conceptual gray boxes를 다시 그릴 수 있습니다. 

> Interestingly, the original StyleGAN applies bias and noise within the style block, causing their relative impact to be inversely proportional to the current style’s magnitudes. 
>> 흥미롭게도, 원래의 styleGAN은 스타일 블록 내에서 편향과 노이즈를 적용하여 상대적인 영향이 현재 스타일의 크기에  inversely proportional(반비례)하도록 한다. 

> We observe that more predictable results are obtained by moving these operations outside the style block, where they operate on normalized data. 
>> 우리는 이러한 작업을 표준화된 데이터에서 작동하는 스타일 블록 외부로 이동함으로써 더 예측 가능한 결과를 얻을 수 있음을 관찰한다. 

> Furthermore, we notice that after this change it is sufficient for the normalization and modulation to operate on the standard deviation alone (i.e., the mean is not needed). 
>> 또한, 우리는 이 변경 후에 standard deviation(표준 편차)에만 대해 정규화와 변조가 작동하기에 충분하다는 것을 알게 된다(즉, mean(평균)은 필요하지 않다). 

> The application of bias, noise, and normalization to the constant input can also be safely removed without observable drawbacks. 
>> constant input(상수 입력)에 대한 편향, 노이즈 및 정규화 적용도 관찰 가능한 drawbacks(단점) 없이 안전하게 제거할 수 있습니다. 

> This variant is shown in Figure 2c, and serves as a starting point for our redesigned normalization.
>> 이 variant(변형)은 그림 2c에 나와 있으며 재설계된 표준화의 starting point 역할을 한다.

### 2.2. Instance normalization revisited

> One of the main strengths of GAN is the ability to control the generated images via style mixing, i.e., by feeding a different latent w to different layers at inference time.
>> StyleGAN의 주요 강점 중 하나는 style mixing을 통해, 즉 추론 시간에 different latent w를 다른 레이어에 공급하여 생성된 이미지를 제어할 수 있는 능력이다.

> In practice, style modulation may amplify certain feature maps by an order of magnitude or more. 
실제로, style modulation(조절)는 certain feature maps을 크기(magnitude) 순서 이상으로 증폭(amplify)할 수 있다. 

> For style mixing to work, we must explicitly counteract this amplification on a per-sample basis — otherwise the subsequent layers would not be able to operate on the data in a meaningful way.
>> style mixing이 작동하려면 샘플별로 이 amplification(증폭)을 명시적으로 대응해야 한다. 그렇지 않으면 subsequent(후속) layers가 meaningful way로 데이터에서 작동할 수 없다.

> If we were willing to sacrifice scale-specific controls (see video), we could simply remove the normalization, thus removing the artifacts and also improving FID slightly [27].
>> scale-specific controls(스케일별 제어)를 sacrifice(희생)할 의향이 있다면(비디오 참조), simply remove the normalization하여  removing the artifacts하고 FID를 slightly(약간) 개선할 수 있습니다[27].

> We will now propose a better alternative that removes the artifacts while retaining full controllability. 
>> 이제 우리는 retaining full controllability을 유지하면서 artifacts를 제거하는 더 나은 alternative(대안)을 제안할 것이다. 

> The main idea is to base normalization on the expected statistics of the incoming feature maps, but without explicit forcing.
>> 주요 아이디어는 들어오는 피쳐 맵의 expected(예상되는) statistics(통계)를 기반으로 정규화하지만 explicit forcing(명시적 강제력)은 없다.

> Recall that a style block in Figure 2c consists of modulation, convolution, and normalization. 
>> 그림 2c의 style block은 modulation, convolution 및 normalization로 구성됩니다. 

> Let us start by considering the effect of a modulation followed by a convolution.
>> modulation 다음에 convolution의 효과를 고려하는 것으로 시작합시다.

> The modulation scales each input feature map of the convolution based on the incoming style, which can alternatively be implemented by scaling the convolution weights:
>> modulation는 incoming style을 기반으로 convolution의 각 input feature map을 스케일링하며, 이는 컨볼루션 가중치를 스케일링하여 alternatively(대신) be implemented(구현)할 수 있다:

![formula 1]()

> where $w$ and $w'$ are the original and modulated weights, respectively, $s_{i}$ is the scale corresponding to the ith input feature map, and $j$ and $k$ enumerate the output feature maps and spatial footprint of the convolution, respectively.
>> 여기서 $w$와 $w'$는 각각 original weights와 modulated weights이며, $s_{i}$는 i번째 입력 feature map에 해당하는 척도이며, $j$와 $k$는 각각 convolution의 output feature maps과 spatial(공간) footprint를 열거한다.

> Now, the purpose of instance normalization is to essentially remove the effect of $s$ from the statistics of the convolution’s output feature maps. 
>> 이제 instance normalization의 목적은 본질적으로 컨볼루션의 output feature maps의 statistics에서 $s$의 효과를 제거하는 것이다. 

> We observe that this goal can be achieved more directly. 
>> 우리는 이 목표가 더 직접적으로 달성될 수 있음을 관찰한다. 

> Let us assume that the input activations are i.i.d. random variables with unit standard deviation. 
>> Input activations가 unit standard deviation(표준편차)가 있는 i.i.d. random variables라고 가정합니다. 

> After modulation and convolution, the output activations have standard deviation of
>> 변조(modulation) 및 convolution 후 output activations는 다음과 같은 standard deviation(표준 편차)를 갖습니다.

![formula 2]()

> i.e., the outputs are scaled by the L2 norm of the corresponding weights. 
>> 즉, 출력은 해당 가중치의  scaled by the L2 norm에 따라 조정됩니다. 

> The subsequent normalization aims to restore the outputs back to unit standard deviation. 
>> subsequent normalization는 출력을 단위 standard deviation로 복원하는 것을 목표로 합니다. 

> Based on Equation 2, this is achieved if we scale (“demodulate”) each output feature map $j$ by $1/σ_{j}$. 
>> Based on Equation 2, 우리가 each output feature map $j$를 $1/σ_{j}$만큼 스케일링("demodulate")하면 이것이 달성된다. 

> Alternatively, we can again bake this into the convolution weights:
>> 또는 이를 다시 컨볼루션 가중치로 사용 할 수 있습니다:

![formula 3]()

> where $\epsilon{}$ is a small constant to avoid numerical issues.
>> 여기서 $\epsilon{}$는 수치 문제를 피하기 위한 작은 상수이다.

> We have now baked the entire style block to a single convolution layer whose weights are adjusted based on s using Equations 1 and 3 (Figure 2d). 

> Compared to instance normalization, our demodulation technique is weaker because it is based on statistical assumptions about the signal instead of actual contents of the feature maps. 

> Similar statistical analysis has been extensively used in modern network initializers [14, 19], but we are not aware of it being previously used as a replacement for data-dependent normalization. 

> Our demodulation is also related to weight normalization [37] that performs the same calculation as a part of reparameterizing the weight tensor. 

> Prior work has identified weight normalization as beneficial in the context of GAN training [43]. 

> Our new design removes the characteristic artifacts (Figure 3) while retaining full controllability, as demonstrated in the accompanying video. 

> FID remains largely unaffected (Table 1, rows A, B), but there is a notable shift from precision to recall. 

> We argue that this is generally desirable, since recall can be traded into precision via truncation, whereas the opposite is not true [27]. 

> In practice our design can be implemented efficiently using grouped convolutions, as detailed in Appendix B. 

> To avoid having to account for the activation function in Equation 3, we scale our activation functions so that they retain the expected signal variance.

## 3. Image quality and generator smoothness

> While GAN metrics such as FID or Precision and Recall (P&R) successfully capture many aspects of the generator, they continue to have somewhat of a blind spot for image quality. 

> For an example, refer to Figures 13 and 14 that contrast generators with identical FID and P&R scores but markedly different overall quality.

> We observe a correlation between perceived image quality and perceptual path length (PPL) [24], a metric that was originally introduced for quantifying the smoothness of the mapping from a latent space to the output image by measuring average LPIPS distances [50] between generated images under small perturbations in latent space. 

> Again consulting Figures 13 and 14, a smaller PPL (smoother generator mapping) appears to correlate with higher overall image quality, whereas other metrics are blind to the change. 

> Figure 4 examines this correlation more closely through per-image PPL scores on LSUN CAT, computed by sampling the latent space around w ∼ f(z). Low scores are indeed indicative of high-quality images, and vice versa. 

> Figure 5a shows the corresponding histogram and reveals the long tail of the distribution. 

> The overall PPL for the model is simply the expected value of these per-image PPL scores. 

> We always compute PPL for the entire image, as opposed to Karras et al. [24] who use a smaller central crop. 

> It is not immediately obvious why a low PPL should correlate with image quality. 

> We hypothesize that during training, as the discriminator penalizes broken images, the most direct way for the generator to improve is to effectively stretch the region of latent space that yields good images. 

> This would lead to the low-quality images being squeezed into small latent space regions of rapid change. 

> While this improves the average output quality in the short term, the accumulating distortions impair the training dynamics and consequently the final image quality. 

> Clearly, we cannot simply encourage minimal PPL since that would guide the generator toward a degenerate solution with zero recall. 

> Instead, we will describe a new regularizer that aims for a smoother generator mapping without this drawback. 

> As the resulting regularization term is somewhat expensive to compute, we first describe a general optimization that applies to any regularization technique.

### 3.1. Lazy regularization

> Typically the main loss function (e.g., logistic loss [16]) and regularization terms (e.g., R1 [30]) are written as a single expression and are thus optimized simultaneously. 

> We observe that the regularization terms can be computed less frequently than the main loss function, thus greatly diminishing their computational cost and the overall memory usage. 

> Table 1, row C shows that no harm is caused when R1 regularization is performed only once every 16 minibatches, and we adopt the same strategy for our new regularizer as well. 

> Appendix B gives implementation details.

### 3.2. Path length regularization

> We would like to encourage that a fixed-size step in W results in a non-zero, fixed-magnitude change in the image. We can measure the deviation from this ideal empirically by stepping into random directions in the image space and observing the corresponding w gradients. 

> These gradients should have close to an equal length regardless of w or the image-space direction, indicating that the mapping from the latent space to image space is well-conditioned [33]. 

> At a single w ∈ W, the local metric scaling properties of the generator mapping g(w) : W 7→ Y are captured by the Jacobian matrix Jw = ∂g(w)/∂w. 

> Motivated by the desire to preserve the expected lengths of vectors regardless of the direction, we formulate our regularizer as

![formula 4]()

> where y are random images with normally distributed pixel intensities, and w ∼ f(z), where z are normally distributed. 

> We show in Appendix C that, in high dimensions, this prior is minimized when Jw is orthogonal (up to a global scale) at any w. 

> An orthogonal matrix preserves lengths and introduces no squeezing along any dimension. 

> To avoid explicit computation of the Jacobian matrix, we use the identity JTwy = ∇w(g(w) · y), which is efficiently computable using standard backpropagation [6].

> The constant a is set dynamically during optimization as the long-running exponential moving average of the lengths kJTwyk2, allowing the optimization to find a suitable global scale by itself. 

> Our regularizer is closely related to the Jacobian clamping regularizer presented by Odena et al. [33]. 

> Practical differences include that we compute the products JTwy analytically whereas they use finite differences for estimating Jwδ with Z 3 δ ∼ N (0, I). 

> It should be noted that spectral normalization [31] of the generator [46] only constrains the largest singular value, posing no constraints on the others and hence not necessarily leading to better conditioning. 

> We find that enabling spectral normalization in addition to our contributions — or instead of them — invariably compromises FID, as detailed in Appendix E. 

> In practice, we notice that path length regularization leads to more reliable and consistently behaving models, making architecture exploration easier. 

> We also observe that the smoother generator is significantly easier to invert (Section 5). 

> Figure 5b shows that path length regularization clearly tightens the distribution of per-image PPL scores, without pushing the mode to zero. 

> However, Table 1, row D points toward a tradeoff between FID and PPL in datasets that are less structured than FFHQ.

## 4. Progressive growing revisited 


> Progressive growing [23] has been very successful in stabilizing high-resolution image synthesis, but it causes its own characteristic artifacts. 

> The key issue is that the progressively grown generator appears to have a strong location preference for details; the accompanying video shows that when features like teeth or eyes should move smoothly over the image, they may instead remain stuck in place before jumping to the next preferred location. 

> Figure 6 shows a related artifact. 

> We believe the problem is that in progressive growing each resolution serves momentarily as the output resolution, forcing it to generate maximal frequency details, which then leads to the trained network to have excessively high frequencies in the intermediate layers, compromising shift invariance [49]. 

> Appendix A shows an example. 

> Thes issues prompt us to search for an alternative formulation that would retain the benefits of progressive growing without the drawbacks.

### 4.1. Alternative network architectures

> While StyleGAN uses simple feedforward designs in the generator (synthesis network) and discriminator, there is a vast body of work dedicated to the study of better network architectures. 

> Skip connections [34, 22], residual networks [18, 17, 31], and hierarchical methods [7, 47, 48] have proven highly successful also in the context of generative methods. 

> As such, we decided to re-evaluate the network design of StyleGAN and search for an architecture that produces high-quality images without progressive growing. 

> Figure 7a shows MSG-GAN [22], which connects the matching resolutions of the generator and discriminator using multiple skip connections. 

> The MSG-GAN generator is modified to output a mipmap [42] instead of an image, and a similar representation is computed for each real image as well. 

> In Figure 7b we simplify this design by upsampling and summing the contributions of RGB outputs corresponding to different resolutions. 

> In the discriminator, we similarly provide the downsampled image to each resolution block of the discriminator. 

> We use bilinear filtering in all up and downsampling operations. 

> In Figure 7c we further modify the design to use residual connections.3 

> This design is similar to LAPGAN [7] without the per-resolution discriminators employed by Denton et al. 

> Table 2 compares three generator and three discriminator architectures: original feedforward networks as used in StyleGAN, skip connections, and residual networks, all trained without progressive growing. 

> FID and PPL are provided for each of the 9 combinations. 

> We can see two broad trends: skip connections in the generator drastically improve PPL in all configurations, and a residual discriminator network is clearly beneficial for FID. 

> The latter is perhaps not surprising since the structure of discriminator resembles classifiers where residual architectures are known to be helpful. 

> However, a residual architecture was harmful in the generator — the lone exception was FID in LSUN CAR when both networks were residual. 

> For the rest of the paper we use a skip generator and a residual discriminator, without progressive growing. 

> This corresponds to configuration E in Table 1, and it significantly improves FID and PPL.

### 4.2. Resolution usage

> The key aspect of progressive growing, which we would like to preserve, is that the generator will initially focus on low-resolution features and then slowly shift its attention to finer details. 

> The architectures in Figure 7 make it possible for the generator to first output low resolution images that are not affected by the higher-resolution layers in a significant way, and later shift the focus to the higher-resolution layers as the training proceeds. 

> Since this is not enforced in any way, the generator will do it only if it is beneficial. 

> To analyze the behavior in practice, we need to quantify how strongly the generator relies on particular resolutions over the course of training. 

> Since the skip generator (Figure 7b) forms the image by explicitly summing RGB values from multiple resolutions, we can estimate the relative importance of the corresponding layers by measuring how much they contribute to the final image. 

> In Figure 8a, we plot the standard deviation of the pixel values produced by each tRGB layer as a function of training time. 

> We calculate the standard deviations over 1024 random samples of w and normalize the values so that they sum to 100%. 

> At the start of training, we can see that the new skip generator behaves similar to progressive growing — now achieved without changing the network topology. 

> It would thus be reasonable to expect the highest resolution to dominate towards the end of the training. 

> The plot, however, shows that this fails to happen in practice, which indicates that the generator may not be able to “fully utilize” the target resolution. 

> To verify this, we inspected the generated images manually and noticed that they generally lack some of the pixel-level detail that is present in the training data — the images could be described as being sharpened versions of 5122 images instead of true 10242 images. 

> This leads us to hypothesize that there is a capacity problem in our networks, which we test by doubling the number of feature maps in the highest-resolution layers of both networks.4 

> This brings the behavior more in line with expectations: Figure 8b shows a significant increase in the contribution of the highest-resolution layers, and Table 1, row F shows that FID and Recall improve markedly. 

> The last row shows that baseline StyleGAN also benefits from additional capacity, but its quality remains far below StyleGAN2. 

> Table 3 compares StyleGAN and StyleGAN2 in four LSUN categories, again showing clear improvements in FID and significant advances in PPL. 

> It is possible that further increases in the size could provide additional benefits.

## 5. Projection of images to latent space

> Inverting the synthesis network g is an interesting problem that has many applications. 

> Manipulating a given image in the latent feature space requires finding a matching latent code w for it first.

> Previous research [1, 10] suggests that instead of finding a common latent code w, the results improve if a separate w is chosen for each layer of the generator. 

> The same approach was used in an early encoder implementation [32]. 

> While extending the latent space in this fashion finds a closer match to a given image, it also enables projecting arbitrary images that should have no latent representation. 

> Instead, we concentrate on finding latent codes in the original, unextended latent space, as these correspond to images that the generator could have produced. 

> Our projection method differs from previous methods in two ways. 

> First, we add ramped-down noise to the latent code during optimization in order to explore the latent space more comprehensively. 

> Second, we also optimize the stochastic noise inputs of the StyleGAN generator, regularizing them to ensure they do not end up carrying coherent signal. 

> The regularization is based on enforcing the autocorrelation coefficients of the noise maps to match those of unit Gaussian noise over multiple scales. 

> Details of our projection method can be found in Appendix D.

### 5.1. Attribution of generated images

> Detection of manipulated or generated images is a very important task. 

> At present, classifier-based methods can quite reliably detect generated images, regardless of their exact origin [29, 45, 40, 51, 41]. 

> However, given the rapid pace of progress in generative methods, this may not be a lasting situation. 

> Besides general detection of fake images, we may also consider a more limited form of the problem: being able to attribute a fake image to its specific source [2]. 

> With StyleGAN, this amounts to checking if there exists a w ∈ W that re-synthesis the image in question. 

> We measure how well the projection succeeds by computing the LPIPS [50] distance between original and resynthesized image as DLPIPS[x, g(˜g−1(x))], where x is the image being analyzed and g˜−1 denotes the approximate projection operation. 

> Figure 10 shows histograms of these distances for LSUN CAR and FFHQ datasets using the original StyleGAN and StyleGAN2, and Figure 9 shows example projections. 

> The images generated using StyleGAN2 can be projected into W so well that they can be almost unambiguously attributed to the generating network. 

> However, with the original StyleGAN, even though it should technically be possible to find a matching latent code, it appears that the mapping from W to images is too complex for this to succeed reliably in practice. 

> We find it encouraging that StyleGAN2 makes source attribution easier even though the image quality has improved significantly.

## 6. Conclusions and future work

> We have identified and fixed several image quality issues in StyleGAN, improving the quality further and considerably advancing the state of the art in several datasets. 

> In some cases the improvements are more clearly seen in motion, as demonstrated in the accompanying video. Appendix A includes further examples of results obtainable using our method. 

> Despite the improved quality, StyleGAN2 makes it easier to attribute a generated image to its source. 

> Training performance has also improved. 

> At 10242 resolution, the original StyleGAN (config A in Table 1) trains at 37 images per second on NVIDIA DGX-1 with 8 Tesla V100 GPUs, while our config E trains 40% faster at 61 img/s. 

> Most of the speedup comes from simplified dataflow due to weight demodulation, lazy regularization, and code optimizations. 

> StyleGAN2 (config F, larger networks) trains at 31 img/s, and is thus only slightly more expensive to train than original StyleGAN. 

> Its total training time was 9 days for FFHQ and 13 days for LSUN CAR. 

> The entire project, including all exploration, consumed 132 MWh of electricity, of which 0.68 MWh went into training the final FFHQ model. 

> In total, we used about 51 single-GPU years of computation (Volta class GPU). A more detailed discussion is available in Appendix F. 

> In the future, it could be fruitful to study further improvements to the path length regularization, e.g., by replacing the pixel-space L2 distance with a data-driven feature-space metric. 

> Considering the practical deployment of GANs, we feel that it will be important to find new ways to reduce the training data requirements. 

> This is especially crucial in applications where it is infeasible to acquire tens of thousands of training samples, and with datasets that include a lot of intrinsic variation.

## Acknowledgements 

> We thank Ming-Yu Liu for an early review, Timo Viitanen for help with the public release, David Luebke for in-depth discussions and helpful comments, and Tero Kuosmanen for technical support with the compute infrastructure.