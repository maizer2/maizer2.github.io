---
layout: post 
title: "(VITON)Toward Characteristic-Preserving Image-based Virtual Try-On Network Translation"
categories: [1. Computer Engineering]
tags: [1.7. Paper Review]
---

### [VITON Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/08/01/paper-of-VITON.html)


### [$$\mathbf{Toward\;Characteristic-Preserving\;Image-based\;Virtual\;Try-On\;Network}$$](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Bochao_Wang_Toward_Characteristic-Preserving_Image-based_ECCV_2018_paper.pdf)

##### $$\mathbf{Bochao\;Wang,\;Huabin\;Zheng,\;Xiaodan\;Liang,\;Yimin\;Chen,\;Liang}$$

##### $$\mathbf{Lin,\;and\;Meng\;Yang}$$

##### $$\mathbf{Sun\;Yat-sen\;University,\;China}$$

##### $$\mathbf{SenseTime\;Group\;Limited}$$

### $\mathbf{Abstract}$

> Image-based virtual try-on systems for fitting a new in-shop clothes into a person image have attracted increasing research attention, yet is still challenging. A desirable pipeline should not only transform the target clothes into the most fitting shape seamlessly but also preserve well the clothes identity in the generated image, that is, the key characteristics (e.g. texture, logo, embroidery) that depict the original clothes. However, previous image-conditioned generation works fail to meet these critical requirements towards the plausible virtual try-on performance since they fail to handle large spatial misalignment between the input image and target clothes. Prior work explicitly tackled spatial deformation using shape context matching, but failed to preserve clothing details due to its coarse-to-fine strategy. In this work, we propose a new fully-learnable Characteristic-Preserving Virtual Try-On Network (CP-VTON) for addressing all real-world challenges in this task. First, CP-VTON learns a thin-plate spline transformation for transforming the in-shop clothes into fitting the body shape of the target person via a new Geometric Matching Module (GMM) rather than computing correspondences of interest points as prior works did. Second, to alleviate boundary artifacts of warped clothes and make the results more realistic, we employ a Try-On Module that learns a composition mask to integrate the warped clothes and the rendered image to ensure smoothness. Extensive experiments on a fashion dataset demonstrate our CP-VTON achieves the state-of-the-art virtual try-on performance both qualitatively and quantitatively.
>> 새로운 매장 내 옷을 사람 이미지에 맞추기 위한 이미지 기반 가상 체험 시스템은 점점 더 많은 연구 관심을 끌었지만 여전히 어려운 과제이다. 바람직한 파이프라인은 대상 옷을 가장 적합한 모양으로 매끄럽게 변형할 뿐만 아니라 생성된 이미지에서 옷의 정체성, 즉 원래 옷을 묘사하는 핵심 특성(예: 질감, 로고, 자수)을 잘 보존해야 한다. 그러나 이전 이미지 조건 생성 작업은 입력 이미지와 대상 옷 사이의 큰 공간 불일치를 처리하지 못하기 때문에 그럴듯한 가상 체험 성능에 대한 이러한 중요한 요구 사항을 충족하지 못한다. 이전 연구는 형태 컨텍스트 매칭을 사용하여 공간 변형을 명시적으로 다루었지만, 거친 전략에서 미세한 전략으로 인해 의류 디테일을 보존하지 못했다. 본 연구에서는 이 작업의 모든 실제 과제를 해결하기 위해 완전히 학습 가능한 새로운 특성 보존 가상 트라이온 네트워크(CP-VTON)를 제안한다. 첫째, CP-VTON은 이전 작업처럼 관심 지점의 대응을 계산하는 대신 새로운 기하학적 매칭 모듈(GMM)을 통해 상점 내 옷을 대상 사람의 체형에 맞게 변형하기 위한 얇은 판 스플라인 변환을 학습한다. 둘째, 뒤틀린 옷의 경계 아티팩트를 완화하고 결과를 보다 사실적으로 만들기 위해 합성 마스크를 학습하는 트라이온 모듈을 사용하여 뒤틀린 옷과 렌더링된 이미지를 통합하여 부드러움을 보장한다. 패션 데이터 세트에 대한 광범위한 실험은 CP-VTON이 질적 및 양적으로 최첨단 가상 트라이온 성능을 달성한다는 것을 보여준다.

> Keywords: Virtual Try-On · Characteristic-Preserving · Thin Plate Spline · Image Alignment

### $\mathbf{1\;Introduction}$

> Online apparel shopping has huge commercial advantages compared to traditional shopping(e.g. time, choice, price) but lacks physical apprehension. To create a shopping environment close to reality, virtual try-on technology has attracted a lot of interests recently by delivering product information similar to that obtained from direct product examination. It allows users to experience themselves wearing different clothes without efforts of changing them physically. This helps users to quickly judge whether they like a garment or not and make buying decisions, and improves sales efficiency of retailers. The traditional pipeline is to use computer graphics to build 3D models and render the output images since graphics methods provide precise control of geometric transformations and physical constraints. But these approaches require plenty of manual labor or expensive devices to collect necessary information for building 3D models and massive computations.
>> 온라인 의류 쇼핑은 기존 쇼핑(예: 시간, 선택, 가격)에 비해 상업적인 이점이 크지만 물리적 이해력은 부족합니다. 가상 체험 기술은 현실에 가까운 쇼핑 환경을 조성하기 위해 최근 직접 제품 검사를 통해 얻은 것과 유사한 제품 정보를 전달해 많은 관심을 끌고 있다. 사용자가 물리적으로 옷을 갈아입는 노력 없이 다른 옷을 입는 경험을 할 수 있게 해준다. 이를 통해 사용자가 의류를 좋아하는지 여부를 신속하게 판단하고 구매 결정을 내릴 수 있으며, 소매업체의 판매 효율성을 높일 수 있다. 전통적인 파이프라인은 그래픽 방법이 기하학적 변환과 물리적 제약 조건을 정확하게 제어하기 때문에 컴퓨터 그래픽을 사용하여 3D 모델을 구축하고 출력 이미지를 렌더링하는 것이다. 그러나 이러한 접근 방식은 3D 모델 구축 및 대규모 컴퓨팅에 필요한 정보를 수집하기 위해 많은 수작업 또는 값비싼 장치가 필요하다.

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-1.PNG)

> Fig. 1. The proposed CP-VTON can generate more realistic image-based virtual tryon results that preserve well key characteristics of the in-shop clothes, compared to the state-of-the-art VITON [10].
>> 그림 1 제안된 CP-VTON은 최첨단 VITON[10]과 비교하여 매장 내 의류의 주요 특성을 잘 보존하는 보다 현실적인 이미지 기반 가상 트라이온 결과를 생성할 수 있다.

> More recently, the image-based virtual try-on system [10] without resorting to 3D information, provides a more economical solution and shows promising results by reformulating it as a conditional image generation problem. Given two images, one of a person and the other of an in-shop clothes, such pipeline aims to synthesize a new image that meets the following requirements:
>> 최근에는 3D 정보에 의존하지 않는 이미지 기반 가상 체험 시스템[10]이 조건부 이미지 생성 문제로 재구성하여 보다 경제적인 솔루션을 제공하고 유망한 결과를 보여준다. 두 개의 이미지, 한 사람과 한 사람의 옷의 다른 이미지를 고려할 때, 이러한 파이프라인은 다음과 같은 요구 사항을 충족하는 새로운 이미지를 합성하는 것을 목표로 한다.

1. > the person is dressed in the new clothes.
    >> 그 사람은 새 옷을 입고 있다.

2. > the original body shape and pose are retained.
    >> 원래 몸매와 포즈는 그대로 유지된다.

3. > the clothing product with high-fidelity is warped smoothly and seamlessly connected with other parts.
    >> 고밀도의 의류제품이 부드럽게 뒤틀려 다른 부품과 매끄럽게 연결됩니다.

4. > the characteristics of clothing product, such as texture, logo and text, are well preserved, without any noticeable artifacts and distortions.
    >> 의류 제품의 텍스처, 로고, 텍스트 등 특색이 잘 보존되어 있어 눈에 띄는 아티팩트와 왜곡이 없습니다.

> Current research and advances in conditional image generation (e.g. image-to-image translation [12, 38, 5, 34, 20, 6]) make it seem to be a natural approach of facilitating this problem. Besides the common pixel-to-pixel losses (e.g. L1 or L2 losses) and perceptual loss [14], an adversarial loss [12] is used to alleviate the blurry issue in some degree, but still misses critical details. Furthermore, these methods can only handle the task with roughly aligned input-output pairs and fail to deal with large transformation cases. Such limitations hinder their application on this challenging virtual try-on task in the wild. One reason is the poor capability in preserving details when facing large geometric changes, e.g. conditioned on unaligned image [23]. The best practice in image-conditional virtual try-on is still a two-stage pipeline VITON [10]. But their performances are far from the plausible and desired generation, as illustrated in Fig. 1. We argue that the main reason lies in the imperfect shape-context matching for aligning clothes and body shape, and the inferior appearance merging strategy.
>> 현재 연구와 조건부 이미지 생성의 발전(예: 이미지 대 이미지 번역 [12, 38, 5, 34, 20, 6])은 이 문제를 용이하게 하는 자연스러운 접근 방식처럼 보인다. 일반적인 픽셀 간 손실(예: L1 또는 L2 손실)과 지각 손실[14] 외에도, 적대적 손실[12]은 어느 정도 흐릿한 문제를 완화하기 위해 사용되지만 여전히 중요한 세부 사항은 누락된다. 또한 이러한 방법은 대략 정렬된 입출력 쌍으로만 작업을 처리할 수 있으며 큰 변환 사례를 처리하지 못한다. 이러한 제한은 야생에서 이 까다로운 가상 체험 작업에 대한 적용을 방해한다. 한 가지 이유는 예를 들어 정렬되지 않은 이미지를 조건으로 한 큰 기하학적 변화에 직면할 때 세부 정보를 보존하는 능력이 부족하기 때문이다[23]. 이미지 조건부 가상 트라이온에서 가장 좋은 방법은 여전히 2단계 파이프라인 VITON[10]이다. 그러나 그들의 성과는 그림 1에서 볼 수 있듯이 그럴듯하고 원하는 세대와는 거리가 멀다. 우리는 주된 이유가 옷과 체형을 맞추기 위한 불완전한 형태-컨텍스트 매칭과 열등한 외모 병합 전략에 있다고 주장한다.

> To address the aforementioned challenges, we present a new image-based method that successfully achieves the plausible try-on image syntheses while preserving cloth characteristics, such as texture, logo, text and so on, named as Characteristic-Preserving Image-based Virtual Try-On Network (CP-VTON). In particular, distinguished from the hand-crafted shape context matching, we propose a new learnable thin-plate spline transformation via a tailored convolutional neural network in order to align well the in-shop clothes with the target person. The network parameters are trained from paired images of in-shop clothes and a wearer, without the need of any explicit correspondences of interest points. Second, our model takes the aligned clothes and clothing-agnostic yet descriptive person representation proposed in [10] as inputs, and generates a pose-coherent image and a composition mask which indicates the details of aligned clothes kept in the synthesized image. The composition mask tends to utilize the information of aligned clothes and balances the smoothness of the synthesized image. Extensive experiments show that the proposed model handles well the large shape and pose transformations and achieves the state-of-art results on the dataset collected by Han et al. [10] in the image-based virtual try-on task.
>> 앞서 언급한 과제를 해결하기 위해 특성 보존 이미지 기반 가상 트라이온 네트워크(CP-VTON)로 명명된 텍스처, 로고, 텍스트 등의 천 특성을 보존하면서 그럴듯한 트라이온 이미지 합성을 성공적으로 달성하는 새로운 이미지 기반 방법을 제시한다.ft 형상 컨텍스트 매칭, 우리는 대상자와 가게 내 옷을 잘 정렬하기 위해 맞춤형 컨볼루션 신경망을 통해 학습 가능한 새로운 박판 스플라인 변환을 제안한다. 네트워크 매개 변수는 관심 지점의 명시적인 대응 없이 상점 내 옷과 착용자의 쌍 이미지에서 훈련된다. 둘째, 우리의 모델은 [10]에서 제안된 정렬된 옷과 옷에 무관하지만 설명적인 인물 표현을 입력으로 사용하고, 합성된 이미지에 유지되는 정렬된 옷의 세부 정보를 나타내는 포즈 일관성 있는 이미지와 합성 마스크를 생성한다. 합성 마스크는 정렬된 옷의 정보를 활용하는 경향이 있으며 합성된 이미지의 부드러움 균형을 맞춘다. 광범위한 실험에 따르면 제안된 모델은 큰 모양과 포즈 변환을 잘 처리하고 한 등이 수집한 데이터 세트에서 최첨단 결과를 달성한다. [10] 이미지 기반 가상 평가판 태스크에 있습니다.

> Our contributions can be summarized as follows:
>> 우리의 기여는 다음과 같이 요약할 수 있습니다.

* > We propose a new Characteristic-Preserving image-based Virtual Try-On Network (CP-VTON) that addresses the characteristic preserving issue when facing large spatial deformation challenge in the realistic virtual try-on task.
    >> 현실적인 가상 트라이온 작업에서 큰 공간 변형 문제에 직면할 때 특성 보존 문제를 해결하는 새로운 특성 보존 이미지 기반 가상 트라이온 네트워크(CP-VTON)를 제안한다.

* > Different from the hand-crafted shape context matching, our CP-VTON incorporates a full learnable thin-plate spline transformation via a new Geometric Matching Module to obtain more robust and powerful alignment.
    >> 수작업으로 만든 형상 컨텍스트 매칭과는 달리, CP-VTON은 새로운 기하학적 매칭 모듈을 통해 완전한 학습이 가능한 박판 스플라인 변환을 통합하여 보다 강력하고 강력한 정렬을 얻는다.

* > Given aligned images, a new Try-On Module is performed to dynamically merge rendered results and warped results.
    >> 정렬된 이미지가 주어지면 렌더링된 결과와 왜곡된 결과를 동적으로 병합하기 위해 새로운 Try-On Module이 수행됩니다.

* > Significant superior performances in image-based virtual try-on task achieved by our CP-VTON have been extensively demonstrated by experiments on the dataset collected by Han et al. [10].
    >> 우리의 CP-VTON이 달성한 이미지 기반 가상 트라이온 작업에서 상당한 우수한 성능은 Han 등이 수집한 데이터 세트에 대한 실험을 통해 광범위하게 입증되었다. [10].

### $\mathbf{2\;Related Work}$

#### $\mathbf{2.1\;Image\;synthesis}$

> Generative adversarial networks(GANs) [9] aim to model the real image distribution by forcing the generated samples to be indistinguishable from the real images. Conditional generative adversarial networks(cGANs) have shown impressive results on image-to-image translation, whose goal is to translate an input image from one domain to another domain [12, 38, 5, 34, 18, 19, 35]. Compared L1/L2 loss, which often leads to blurry images, the adversarial loss has become a popular choice for many image-to-image tasks. Recently, Chen and Koltun [3] suggest that the adversarial loss might be unstable for high-resolution image generation. We find the adversarial loss has little improvement in our model. In image-to-image translation tasks, there exists an implicit assumption that the input and output are roughly aligned with each other and they represent the same underlying structure. However, most of these methods have some problems when dealing with large spatial deformations between the conditioned image and the target one. Most of image-to image translation tasks conditioned on unaligned images [10, 23, 37], adopt a coarse-to-fine manner to enhance the quality of final results. To address the misalignment of conditioned images, Siarohit et al. [31] introduced a deformable skip connections in GAN, using the correspondences of the pose points. VITON [10] computes shape context thin-plate spline(TPS) transofrmation [2] between the mask of in-shop clothes and the predicted foreground mask. Shape context is a hand-craft feature for shape and the matching of two shapes is time-consumed. Besides, the computed TPS transoformations are vulnerable to the predicted mask. Inspired by Rocco et al. [27], we design a convolutional neural network(CNN) to estimate a TPS transformation between in-shop clothes and the target image without any explicit correspondences of interest points.
>> 생성적 적대 네트워크(GAN)[9]는 생성된 샘플을 실제 이미지와 구별할 수 없게 하여 실제 이미지 분포를 모델링하는 것을 목표로 한다. 조건부 생성 적대적 네트워크(cGAN)는 한 도메인에서 다른 도메인으로 입력 이미지를 변환하는 것이 목표인 이미지 대 이미지 변환에서 인상적인 결과를 보여주었다[12, 38, 5, 34, 18, 19, 35]. 종종 흐릿한 이미지로 이어지는 L1/L2 손실과 비교하여, 적대적 손실은 많은 이미지 대 이미지 작업에서 인기 있는 선택이 되었다. 최근, Chen과 Koltun[3]은 고해상도 이미지 생성에 대해 적대적 손실이 불안정할 수 있다고 제안한다. 우리는 적대적 손실이 우리 모델에서 거의 개선되지 않는다는 것을 발견했다. 이미지 대 이미지 변환 작업에서 입력과 출력이 대략적으로 정렬되고 동일한 기본 구조를 나타낸다는 암묵적인 가정이 존재한다. 그러나 이러한 방법의 대부분은 조건부 이미지와 대상 이미지 사이의 큰 공간 변형을 처리할 때 몇 가지 문제가 있다. 정렬되지 않은 이미지[10, 23, 37]에 따라 조정된 대부분의 이미지 대 이미지 변환 작업은 최종 결과의 품질을 향상시키기 위해 거칠고 미세한 방법을 채택한다. 조건부 영상의 정렬 오류 문제를 해결하기 위해 Siarohit 외. [31] 포즈 포인트의 대응성을 사용하여 GAN에 변형 가능한 스킵 연결을 도입했다. VITON[10]은 매장 내 옷의 마스크와 예측된 전경 마스크 사이의 형상 컨텍스트 박판 스플라인(TPS) 변환[2]을 계산한다. 도형 컨텍스트는 도형을 위한 수공예 기능이며 두 도형의 매칭은 시간이 많이 소요됩니다. 게다가, 계산된 TPS 변환은 예측된 마스크에 취약하다. Rocco 외에서 영감을 받았습니다. [27], 우리는 관심 지점의 명시적인 대응 없이 상점 내 옷과 대상 이미지 사이의 TPS 변환을 추정하기 위해 컨볼루션 신경망(CNN)을 설계한다.

#### $\mathbf{2.2\;Person\;Image\;generation}$

> Lassner et al. [17] introduced a generative model that can generate human parsing [8] maps and translate them into persons in clothing. But it is not clear how to control the generated fashion items. Zhao et al. [37] addressed a problem of generating multi-view clothing images based on a given clothing image of a certain view. PG2 [23] synthesizes the person images in arbitrary pose, which explicitly uses the target pose as a condition. Siarohit et al. [31] dealt the same task as PG2, but using the correspondences between the target pose and the pose of conditional image. The generated fashion items in [37, 23, 31], kept consistent with that of the conditional images. FashionGAN [39] changed the fashion items on a person and generated new outfits by text descriptions. The goal of virtual try-on is to synthesize a photo-realistic new image with a new piece of clothing product, while leaving out effects of the old one. Yoo te al. [36] generated in shop clothes conditioned on a person in clothing, rather than the reverse.
>> 라스너 외. [17] 인간 구문 분석[8] 맵을 생성하고 이를 옷을 입은 사람으로 변환할 수 있는 생성 모델을 도입했다. 그러나 생성된 패션 아이템을 어떻게 통제해야 하는지는 명확하지 않다. 자오 외 [37]는 특정 관점의 주어진 옷 이미지를 기반으로 멀티 뷰 옷 이미지를 생성하는 문제를 해결했다. PG2[23]는 목표 포즈를 조건으로 명시적으로 사용하는 임의 포즈로 인물 이미지를 합성한다. 시아로히트 외. [31] PG2와 동일한 작업을 처리했지만 대상 포즈와 조건부 이미지의 포즈 사이의 대응 관계를 사용했다. [37, 23, 31]에서 생성된 패션 아이템은 조건부 이미지의 아이템과 일관성을 유지했다. 패션GAN [39]은 사람의 패션 아이템을 변경하고 텍스트 설명을 통해 새로운 의상을 생성했습니다. 가상 체험의 목표는 기존 의류의 효과를 배제하고 새로운 의복 제품으로 사진처럼 사실적인 새로운 이미지를 합성하는 것이다. 청개구리. [36]은 옷을 입은 사람에게 조건화된 상점 의복에서 생성되며, 그 반대보다.

#### $\mathbf{2.3\;Virtual\;Try-on\;System}$

> Most virtual try-on works are based on graphics models. Sekine et al. [30] introduced a virtual fitting system that captures 3D measurements of body shape. Chen et al. [4] used a SCAPE [1] body model to generate synthetic people. PonsMoll et al. [26] used a 3D scanner to automatically capture real clothing and estimate body shape and pose. Compared to graphics models, image-based generative models are more computationally efficient. Jetchev and Bergmann [13] proposed a conditional analogy GAN to swap fashion articles, without other descriptive person representation. They didn’t take pose variant into consideration, and during inference, they required the paired images of in-shop clothes and a wearer, which limits their practical scenarios. The most related work is VITON [10]. We all aim to synthesize photo-realistic image directly from 2D images. VITON addressed this problem with a coarse-to-fine framework and expected to capture the cloth deformation by a shape context TPS transoformation. We propose an alignment network and a single pass generative framework, which preserving the characteristics of in-shop clothes.
>> 대부분의 가상 체험 작품은 그래픽 모델을 기반으로 합니다. 세키네 외. [30] 체형을 3D로 측정하는 가상 피팅 시스템을 도입했다. 첸 외. [4] 합성 사람을 생성하기 위해 SCAPE[1] 신체 모델을 사용했다. 폰스몰 외. [26]는 3D 스캐너를 사용하여 실제 옷을 자동으로 캡처하고 체형과 포즈를 추정했습니다. 그래픽 모델에 비해 이미지 기반 생성 모델이 더 계산 효율적이다. Jetchev와 Bergmann[13]은 다른 서술적 인물 표현 없이 패션 기사를 교환하기 위한 조건부 유사 GAN을 제안했다. 그들은 포즈 변형을 고려하지 않았고, 추론하는 동안, 그들은 그들의 실용적인 시나리오를 제한하기 위해 매장 내 옷과 착용자의 쌍으로 된 이미지를 필요로 했다. 가장 관련성이 높은 작업은 VITON [10]입니다. 우리는 모두 2D 이미지에서 직접 사실적인 이미지를 합성하는 것을 목표로 한다. VITON은 거친 프레임워크에서 미세한 프레임워크로 이 문제를 해결했으며 형태 컨텍스트 TPS 변환에 의해 천 변형을 포착할 것으로 기대되었다. 우리는 매장 내 옷의 특성을 보존하는 정렬 네트워크와 단일 패스 생성 프레임워크를 제안한다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-2.PNG)

> Fig. 2. An overview of our CP-VTON, containing two main modules. (a) Geometric Matching Module: the in-shop clothes $c$ and input image representation $p$ are aligned via a learnable matching module. (b) Try-On Module: it generates a composition mask $M$ and a rendered person $I_{r}$. The final results $I_{o}$ is composed by warped clothes $\hat{c}$ and the rendered person $I_{r}$ with the composition mask $M$.
>> 그림 2. 두 개의 주요 모듈을 포함하는 CP-VTON의 개요. (a) 기하학적 매칭 모듈: 학습 가능한 매칭 모듈을 통해 매장 내 의류 $c$와 입력 이미지 표현 $p$가 정렬된다. (b) Try-On 모듈: 합성 마스크 $M$과 렌더링된 사람 $I_{r}$을 생성한다. 최종 결과 $I_{o}$는 왜곡된 옷 $\hat{c}$와 합성 마스크 $M$과 함께 렌더링된 사람 $I_{r}$에 의해 구성된다.

### $\mathbf{3\;Characteristic-Preserving\;Virtual\;Try-On\;Network}$

> We address the task of image-based virtual try-on as a conditional image generation problem. Generally, given a reference image $I_{i}$ of a person wearing in clothes $c_{i}$ and a target clothes $c$, the goal of CP-VTON is to synthesize a new image $I_{o}$ of the wearer in the new cloth $c_{o}$, in which the body shape and pose of $I_{i}$ are retained, the characteristics of target clothes $c$ are reserved and the effects of the old clothes $c_{i}$ are eliminated.
>> 우리는 조건부 이미지 생성 문제로 이미지 기반 가상 트라이온 작업을 다룬다. 일반적으로 옷을 입은 사람의 참조 이미지 $I_{i}$와 대상 옷 $c$가 주어지면 CP-VTON의 목표는 $I_{i}$의 체형과 포즈가 유지되는 새 옷 $c_{o}$에서 착용자의 새로운 이미지 $I_{o}$를 합성하는 것이다.hes $c_{i}$는 제거된다.

> Training with sample triplets ($I_{i}$, $c$, $I_{t}$) where $I_{t}$ is the ground truth of $I_{o}$ and $c$ is coupled with $I_{t}$ wearing in clothes $c_{t}$, is straightforward but undesirable in practice. Because these triplets are difficult to collect. It is easier if $I_{i}$ is same as $I_{t}$, which means that $c$, It pairs are enough. These paris are in abundance from shopping websites. But directly training on ($I_{t}$, $c$, $I_{t}$) harms the model generalization ability at testing phase when only decoupled inputs ($I_{i}$, $c$) are available. Prior work [10] addressed this dilemma by constructing a clothingagnostic person representation $p$ to eliminate the effects of source clothing item ci. With ($I_{t}$, $c$, $I_{t}$) transformed into a new triplet form ($p$, $c$, $I_{t}$), training and testing phase are unified. We adopted this representation in our method and further enhance it by eliminating less information from reference person image. Details are described in Sec. 3.1. One of the challenges of image-based virtual try-on lies in the large spatial misalignment between in-shop clothing item and wearer’s body. Existing network architectures for conditional image generation (e.g. FCN [21], UNet [28], ResNet [11]) lack the ability to handle large spatial deformation, leading to blurry try-on results. We proposed a Geometric Matching Module (GMM) to explicitly align the input clothes $c$ with aforementioned person representation $p$ and produce a warped clothes image $\hat{c}$. GMM is a endto-end neural network directly trained using pixel-wise L1 loss. Sec. 3.2 gives the details. Sec. 3.3 completes our virtual try-on pipeline with a characteristicpreserving Try-On Module. The Try-On module synthesizes final try-on results $I_{o}$ by fusing the warped clothes $\hat{c}$ and the rendered person image $I_{r}$. The overall pipeline is depicted in Fig. 2.
>> $I_{t}$가 $I_{o}$의 실측값이고 $c$가 $c_{t}$의 옷을 입은 $I_{t}$와 결합된 샘플 삼중값($I_{i}$, $I_{t}$)을 사용한 훈련은 간단하지만 실제로는 바람직하지 않다. 왜냐하면 이 세 쌍둥이는 모으기가 어렵기 때문입니다. $I_{i}$가 $I_{t}$와 같으면 더 쉽다. 즉, $c$, It 쌍이면 충분하다. 이 파리는 쇼핑 웹사이트들에서 많이 볼 수 있다. 그러나 ($I_{t}$, $c$, $I_{t}$)에 대한 직접 교육은 분리된 입력($I_{i}$, $c$)만 사용할 수 있는 테스트 단계에서 모델 일반화 능력에 해를 끼친다. 이전 연구[10]는 소스 의류 품목 ci의 효과를 제거하기 위해 의류 불가지론자 표현 $p$를 구성하여 이 딜레마를 해결했다. ($I_{t}$, $c$, $I_{t}$)가 새로운 삼중항 형태($p$, $c$, $I_{t}$)로 변환되면 훈련 및 테스트 단계가 통합된다. 우리는 이 표현을 우리의 방법에 채택했고 참조 인물 이미지에서 더 적은 정보를 제거하여 그것을 더욱 향상시켰다. 자세한 내용은 3.1절에 설명되어 있다. 이미지 기반 가상 체험의 어려움 중 하나는 상점 내 의류 품목과 착용자의 신체 사이의 큰 공간적 불일치에 있다. 조건부 이미지 생성을 위한 기존 네트워크 아키텍처(예: FCN [21], UNet [28], ResNet [11])는 큰 공간 변형을 처리할 수 있는 기능이 부족하여 시도 결과가 흐릿하다. 우리는 입력 옷 $c$를 앞서 언급한 인물 표현 $p$와 명시적으로 정렬하고 뒤틀린 옷 이미지 $\hat{c}$를 생성하기 위해 기하학적 매칭 모듈(GMM)을 제안했다. GMM은 픽셀 단위 L1 손실을 사용하여 직접 훈련된 종단 간 신경망이다. 3.2절에 자세한 내용이 나와 있다. 3.3항은 특성 보존 Try-On Module로 가상 Try-On 파이프라인을 완성한다. Try-On 모듈은 뒤틀린 옷 $\hat{c}$와 렌더링된 사람 이미지 $I_{r}$를 융합하여 최종 트라이온 결과 $I_{o}$를 합성한다. 전체 파이프라인은 그림 2에 도시되어 있다.

#### $\mathbf{3.1\;Person\;Representation}$

> The original cloth-agnostic person representation [10] aims at leaving out the effects of old clothes $c_{i}$ like its color, texture and shape, while preserving information of input person $I_{i}$ as much as possible, including the person’s face, hair, body shape and pose. It contains three components:
>> 원래 천에 구애받지 않는 사람 표현[10]은 입력 사람 $I_{i}$의 정보를 가능한 한 보존하면서 색상, 질감 및 모양과 같은 오래된 옷의 효과를 배제하는 것을 목표로 한다. 세 가지 구성 요소로 구성됩니다.

* > Pose heatmap: an 18-channel feature map with each channel corresponding to one human pose keypoint, drawn as an $11 × 11$ white rectangle.
    >> 포즈 히트 맵: 각 채널이 하나의 인간 포즈 키포인트에 해당하는 18채널 피처 맵으로 $11 × 11$ 흰색 직사각형으로 그려진다.

* > Body shape: a 1-channel feature map of a blurred binary mask that roughly covering different parts of human body.
    >> 체형: 인간의 신체 여러 부분을 대략적으로 덮는 흐릿한 이진 마스크의 1채널 특징 지도.

* > Reserved regions: an RGB image that contains the reserved regions to maintain the identity of a person, including face and hair.
    >> 예약 영역: 얼굴과 머리카락을 포함하여 사람의 정체성을 유지하기 위한 예약 영역을 포함하는 RGB 이미지입니다

> These feature maps are all scaled to a fixed resolution 256×192 and concatenated together to form the cloth-agnostic person representation map $p$ of $k$ channels, where $k=18+1+3=22$. We also utilize this representation in both our matching module and try-on module.
>> 이러한 기능 맵은 모두 고정 해상도 256×192로 스케일링되고 함께 연결되어 $k$ 채널의 천에 구애받지 않는 사람 표현 맵 $p$를 형성한다. 여기서 $k=18+1+3=22$ 우리는 또한 매칭 모듈과 트라이온 모듈 모두에서 이 표현을 활용한다.

#### $\mathbf{3.2\;Geometric\;Matching\;Module}$

> The classical approach for the geometry estimation task of image matching consists of three stages: (1) local descriptors (e.g. shape context [2], SIFT [22] ) are extracted from both input images, (2) the descriptors are matched across images form a set of tentative correspondences, (3) these correspondences are used to robustly estimate the parameters of geometric model using RANSAC [7] or Hough voting [16, 22].
>> 이미지 매칭의 지오메트리 추정 작업을 위한 고전적인 접근 방식은 (1) 로컬 설명자(예: 형상 컨텍스트 [2], SIFT [22])가 두 입력 이미지 모두에서 추출된다.

> Rocco et al. [27] mimics this process using differentiable modules so that it can be trainable end-to-end for geometry estimation tasks. Inspired by this work, we design a new Geometric Matching Module (GMM) to transform the target clothes $c$ into warped clothes $\hat{c}$ which is roughly aligned with input person representation $p$. As illustrated in Fig. 2, our GMM consists of four parts:
>> 로코 외. [27]는 지오메트리 추정 작업에 대해 종단 간 훈련할 수 있도록 차별화 가능한 모듈을 사용하여 이 프로세스를 모방한다. 본 연구에서 영감을 받아, 우리는 목표 옷 $c$를 입력 인물 표현 $p$와 대략 정렬된 뒤틀린 옷 $\hat{c}$으로 변환하는 새로운 기하학적 매칭 모듈(GMM)을 설계한다. 그림 2에 나타난 바와 같이, GMM은 네 부분으로 구성되어 있습니다.

1. > two networks for extracting high-level features of $p$ and $c$ respectively.
    >> 각각 $p$와 $c$의 높은 수준의 기능을 추출하기 위한 두 개의 네트워크.

2. > a correlation layer to combine two features into a single tensor as input to the regressor network.
    >> 회귀자 네트워크에 대한 입력으로 두 가지 특징을 단일 텐서로 결합하는 상관 계층.

3. > the regression network for predicting the spatial transformation parameters $θ$.
    >> 공간 변환 매개 변수 $θ$를 예측하기 위한 회귀 네트워크.

4. > a Thin-Plate Spline (TPS) transformation module $T$ for warping an image into the output $\hat{c}=T_{θ}(c)$. 
    >> 이미지를 출력 $\hat{c}=T_{θ}(c)$로 뒤틀리기 위한 얇은 판 조각(TPS) 변환 모듈 $T$.

> The pipeline is end-to-end learnable and trained with sample triplets ($p$, $c$, $c_{t}$), under the pixel-wise L1 loss between the warped result $\hat{c}$ and ground truth $c_{t}$, where  $c_{t}$ is the clothes worn on the target person in It:
>> 파이프라인은 왜곡된 결과 $\hat{c}$와 지상 진리 $c_{t}$ 사이의 픽셀 단위 L1 손실 아래에서 샘플 삼중항 ($p$, $c$, $c_{t}$)로 학습되고 훈련된다. 여기서 $c_{t}$는 대상자에게 입는 옷이다.

$$L_{GMM}(θ)=\vert{}\vert{}\hat{c}−c_{t}\vert{}\vert{}=\vert{}\vert{}T_{θ}(c)−c_{t}\vert{}\vert{}$$

> The key differences between our approach and Rocco et al. [27] are three-fold. First, we trained from scratch rather than using a pretrained VGG network. Second, our training ground truths are acquired from wearer’s real clothes rather than synthesized from simulated warping. Most importantly, our GMM is directly supervised under pixel-wise L1 loss between warping outputs and ground truth.
>> 우리의 접근 방식과 Rocco 외 연구진 사이의 주요 차이점 [27]은 세 겹이다. 첫째, 우리는 사전 훈련된 VGG 네트워크를 사용하기보다는 처음부터 훈련했다. 둘째, 우리의 훈련 배경 진실은 시뮬레이션된 뒤틀림에서 합성되기보다는 착용자의 실제 옷에서 획득된다. 가장 중요한 것은 GMM이 뒤틀린 출력과 실측값 사이의 픽셀 단위 L1 손실에서 직접 감독된다는 것이다.

#### $\mathbf{3.3\;Try-on\;Module}$

> Now that the warped clothes $\hat{c}$ is roughly aligned with the body shape of the target person, the goal of our Try-On module is to fuse $\hat{c}$ with the target person and for synthesizing the final try-on result. One straightforward solution is directly pasting $\hat{c}$ onto target person image $I_{t}$. It has the advantage that the characteristics of warped clothes are fully preserved, but leads to an unnatural appearance at the boundary regions of clothes and undesirable occlusion of some body parts (e.g. hair, arms). Another solution widely adopted in conditional image generation is translating inputs to outputs by a single forward pass of some encoder-decoder networks, such as UNet [28], which is desirable for rendering seamless smooth images. However, It is impossible to perfectly align clothes with target body shape. Lacking explicit spatial deformation ability, even minor misalignment could make the UNet-rendered output blurry.
>> 뒤틀린 옷 $\hat{c}$가 대상 사람의 체형과 대략 정렬되었으므로, 우리의 트라이온 모듈의 목표는 $\hat{c}$를 대상 사람과 융합하고 최종 트라이온 결과를 합성하는 것이다. 한 가지 간단한 해결책은 대상 인물 이미지 $I_{t}$에 $\hat{c}$를 직접 붙여넣는 것이다. 뒤틀린 옷의 특성이 충분히 보존되어 있지만 옷의 경계 부위에 부자연스러운 외관과 일부 신체 부위(예: 머리카락, 팔)의 바람직하지 않은 폐색을 초래한다는 장점이 있다. 조건부 이미지 생성에 널리 채택된 또 다른 솔루션은 UNet[28]과 같은 일부 인코더-디코더 네트워크의 단일 정방향 패스로 입력을 출력으로 변환하는 것인데, 이는 매끄러운 이미지를 렌더링하는 데 바람직하다. 그러나 목표한 체형에 맞춰 옷을 완벽하게 맞추는 것은 불가능하다. 명시적인 공간 변형 능력이 부족하면 사소한 정렬 오류도 UNet이 렌더링한 출력을 흐리게 만들 수 있다.

> Our Try-On Module aims to combine the advantages of both approaches above. As illustrated in Fig. 2, given a concatenated input of person representation $p$ and the warped clothes $\hat{c}$, UNet simultaneously renders a person image $I_{r}$ and predicts a composition mask $M$. The rendered person $I_{r}$ and the warped clothes $\hat{c}$ are then fused together using the composition mask $M$ to synthesize the final try-on result $I_{o}$:
>> Try-On Module은 위의 두 가지 접근 방식의 장점을 결합하는 것을 목표로 합니다. 그림 2에서 보듯이, 인물 표현 $p$와 뒤틀린 옷 $\hat{c}$의 연결된 입력이 주어지면, UNet은 인물 이미지 $I_{r}$를 동시에 렌더링하고 합성 마스크 $M$을 예측한다. 그런 다음 렌더링된 사람 $I_{r}$와 뒤틀린 옷 $\hat{c}$이 합성 마스크 $M$을 사용하여 함께 융합되어 최종 트라이온 결과 $I_{o}$를 합성한다.

$$I_{o}=M⊙\hat{c}+(1−M)⊙I_{r}$$

> where ⊙ represents element-wise matrix multiplication.
>> 여기서 ⊙는 요소별 행렬 곱셈을 나타냅니다.

> At training phase, given the sample triples ($p$, $c$, $I_{t}$), the goal of Try-On Module is to minimize the discrepancy between output $I_{o}$ and ground truth $I_{t}$. We adopted the widely used strategy in conditional image generation problem that using a combination of L1 loss and VGG perceptual loss [14], where the VGG perceptual loss is defined as follows:
>> 교육 단계에서 샘플 트리플 ($p$, $c$, $I_{t}$)가 주어졌을 때, 트라이온 모듈의 목표는 출력 $I_{o}$와 실측값 $I_{t}$ 사이의 불일치를 최소화하는 것이다. 우리는 L1 손실과 VGG 지각 손실[14]의 조합을 사용하는 조건부 이미지 생성 문제에 널리 사용되는 전략을 채택했다. 여기서 VGG 지각 손실은 다음과 같이 정의된다.

$$L_{VGG}(I_{o},I_{t})=\sum_{i=1}^{5}λ_{i}\vert{}\vert{}φ_{i}(I_{o})−φ_{i}(I_{t})\vert{}\vert{}1$$

> where $φ_{i}(I)$ denotes the feature map of image $I$ of the i-th layer in the visual perception network $φ$, which is a VGG19 [32] pre-trained on ImageNet. The layer $i≥1$ stands for ’conv1 2’, ’conv2 2’, ’conv3 2’, ’conv4 2’, ’conv5 2’, respectively. Towards our goal of characteristic-preserving, we bias the composition mask $M$ to select warped clothes as much as possible by applying a L1 regularization $\vert{}\vert{}−M\vert{}\vert{}$ on $M$. The overall loss function for Try-On Module (TOM) is:
>> 여기서 $φ_{i}(I)$는 ImageNet에서 사전 훈련된 VGG19[32]인 시각적 인식 네트워크 $dig$에서 i번째 계층의 이미지 $I$의 기능 맵을 나타낸다. 계층 $φ$은 각각 'conv12', 'conv22', 'conv32', 'conv42', 'conv52'를 나타냅니다. 특성 보존이라는 목표를 향해, 우리는 합성 마스크 $M$을 편향시켜 $M$에 L1 정규화 $\vert{}\vert{}1−M\vert{}\vert{}$를 적용하여 가능한 한 왜곡된 옷을 선택한다. Tri-On Module(TOM)의 전체 손실 함수는 다음과 같다.

$$L_{TOM}=λ_{L1}\vert{}\vert{}I_{o}−I_{t}\vert{}\vert{}+λ_{vgg}L_{VGG}(\hat{I}, I)+λ_{mask}\vert{}\vert{}1−M\vert{}\vert{}.$$

### $\mathbf{4\;Experiments\;and\;Analysis}$

#### $\mathbf{4.1\;Dataset}$

> We conduct our all experiments on the datasets collected by Han et al. [10]. It contains around 19,000 front-view woman and top clothing image pairs. There are 16253 cleaned pairs, which are split into a training set and a validation set with 14221 and 2032 pairs, respectively. We rearrange the images in the validation set into unpaired pairs as the testing set.
>> 우리는 Han 등이 수집한 데이터 세트에 대한 모든 실험을 수행한다. [10. 여기에는 약 19,000개의 프론트 뷰 여성과 톱 의류 이미지 쌍이 포함되어 있습니다. 세척된 쌍은 16253개이며, 각각 14221개 및 2032개 쌍으로 구성된 교육 세트와 검증 세트로 분할됩니다. 검증 세트의 이미지를 테스트 세트로 쌍을 이루지 않은 쌍으로 다시 정렬한다.

#### $\mathbf{4.2\;Quantitative\;Evaluation}$

> We evaluate the quantitative performance of different virtual try-on methods via a human subjective perceptual study. Inception Score (IS) [29] is usually used as to quantitatively evaluate the image synthesis quality, but not suitable for evaluating this task for that it cannot reflect whether the details are preserved as described in [10]. We focus on the clothes with rich details since we are interested in characteristic-preservation, instead of evaluating on the whole testing set. For simplicity, we measure the detail richness of a clothing image by its total variation (TV) norm. It is appropriate for this dataset since the background is in pure color and the TV norm is only contributed by clothes itself, as illustrated in Fig. 3. We extracted 50 testing pairs with largest clothing TV norm named as LARGE to evaluate characteristic-preservation of our methods, and 50 pairs with smallest TV norm named as SMALL to ensure that our methods perform at least as good as previous state-of-the-art methods in simpler cases.
>> 우리는 인간의 주관적 지각 연구를 통해 다양한 가상 체험 방법의 정량적 성능을 평가한다. 초기 점수(IS) [29]는 일반적으로 이미지 합성 품질을 정량적으로 평가하는 데 사용되지만 [10]에 설명된 대로 세부 정보가 보존되는지 여부를 반영할 수 없기 때문에 이 작업을 평가하는 데는 적합하지 않다. 우리는 전체 테스트 세트를 평가하는 대신 특성 보존에 관심이 있기 때문에 디테일이 풍부한 옷에 중점을 둔다. 단순성을 위해, 우리는 총 변형(TV) 규범으로 의류 이미지의 세부 풍부함을 측정한다. 그림 3과 같이 배경은 순수한 색이고 TV 노름은 옷 자체만으로 기여하기 때문에 이 데이터 세트에 적합하다. 우리는 방법의 특성 보존을 평가하기 위해 LARGE로 명명된 가장 큰 의류 TV 규범을 가진 50쌍의 테스트 쌍을 추출하고, 더 단순한 경우에서 우리의 방법이 이전 최첨단 방법만큼 잘 수행되도록 하기 위해 SMARG로 명명된 가장 작은 TV 규범을 가진 50쌍을 추출했다.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-3.PNG)

> Fig. 3. From top to bottom, the TV norm values are increasing. Each line shows some clothes in the same level.
>> 그림 3. 위로부터 아래로 TV 노름 값이 증가하고 있다. 각 선은 같은 레벨의 옷을 보여준다.

> We conducted pairwise A/B tests on Amazon Mechanical Turk (AMT) platform. Specifically, given a person image and a target clothing image, the worker is asked to select the image which is more realistic and preserves more details of the target clothes between two virtual try-on results from different methods. There is no time limited for these jobs, and each job is assigned to 4 different workers. Human evaluation metric is computed in the same way as in [10].
>> Amazon Mechanical Turk(AMT) 플랫폼에서 쌍으로 A/B 테스트를 수행하였습니다. 구체적으로, 사람 이미지와 대상 옷 이미지가 주어지면 작업자는 다른 방법의 두 가상 체험 결과 사이에서 더 사실적이고 대상 옷의 더 자세한 내용을 보존하는 이미지를 선택하도록 요청받는다. 이 작업에는 시간 제한이 없으며 각 작업은 4명의 다른 작업자에게 할당됩니다. 인간 평가 지표는 [10]과 같은 방법으로 계산된다.

#### $\mathbf{4.3\;Implementation\;Details}$

> **Training Setup** In all experiments, we use $λ_{L1}=λ_{vgg}=1$. When composition mask is used, we set $λ_{mask}=1$. We trained both Geometric Matching Module and Try-on Module for 200K steps with batch size 4. We use Adam [15] optimizer with $β_{1}=0.5$ and $β_{2}=0.999$. Learning rate is first fixed at 0.0001 for 100K steps and then linearly decays to zero for the remaining steps. All input images are resized to 256 × 192 and the output images have the same resolution.
>> **교육 설정** 모든 실험에서 $λ_{L1}=λ_{vgg}=1$를 사용합니다. 컴포지션 마스크가 사용되면 $λ_{mask}=1$를 설정합니다. 배치 크기가 4인 200K 단계에 대해 기하학적 매칭 모듈과 트라이온 모듈을 모두 교육했다. Adam [15] 최적화 도구를 $β_{1}=0.5$와 $β_{2}=0.999$와 함께 사용합니다. 학습률은 100K 단계에 대해 먼저 0.0001로 고정된 다음 나머지 단계에 대해 0으로 선형적으로 감소한다. 모든 입력 영상의 크기는 256 × 192로 조정되며 출력 영상의 해상도는 동일합니다.

> **Geometric** Matching Module Feature extraction networks for person representation and clothes have the similar structure, containing four 2-strided downsampling convolutional layers, succeeded by two 1-strided ones, their numbers of filters being 64, 128, 256, 512, 512, respectively. The only difference is the number of input channels. Regression network contains two 2-strided convolutional layers, two 1-strided ones and one fully-connected output layer. The numbers of filters are 512, 256, 128, 64. The fully-connected layer predicts the $x-$ and $y-$ coordinate offsets of TPS anchor points, thus has an output size of $2×5×5=50$.
>> **기하학적인** 인물 표현 및 복장에 대한 기하학적 모듈 특징 추출 네트워크는 유사한 구조를 가지고 있으며, 4개의 2단계 다운샘플링 컨볼루션 레이어를 포함하며, 2개의 1단계 레이어가 이어지며, 필터 수는 각각 64, 128, 256, 512, 512이다. 유일한 차이점은 입력 채널의 수입니다. 회귀 네트워크는 두 개의 2단 컨볼루션 레이어, 두 개의 1단 컨볼루션 레이어 및 한 개의 완전히 연결된 출력 레이어를 포함한다. 필터 수는 512, 256, 128, 64입니다. 완전히 연결된 레이어는 TPS 앵커 포인트의 $x-$ 및 $y-$ 좌표 오프셋을 예측하므로 출력 크기는 $2×5×5=50$이다.

> **Try-On Module** We use a 12-layer UNet with six 2-strided down-sampling convolutional layers and six up-sampling layers. To alleviate so-called “checkerboard artifacts”, we replace 2-strided deconvolutional layers normally used for up-sampling with the combination of nearest-neighbor interpolation layers and 1-strided convolutional layers, as suggested by [25]. The numbers of filters for down-sampling convolutional layers are 64, 128, 256, 512, 512, 512. The numbers of filters for up-sampling convolutional layers are 512, 512, 256, 128, 64, 4. Each convolutional layer is followed by an Instance Normalization layer [33] and Leaky ReLU [24], of which the slope is set to 0.2.
>> **Try-On Module** 우리는 6개의 2단계 다운 샘플링 컨볼루션 레이어와 6개의 업샘플링 레이어가 있는 12계층 UNet을 사용한다. 소위 "체커보드 아티팩트"를 완화하기 위해, 우리는 업샘플링에 일반적으로 사용되는 2단 디콘볼루션 레이어를 [25]에서 제안한 것처럼 가장 가까운 이웃 보간 레이어와 1단 컨볼루션 레이어의 조합으로 대체한다. 컨볼루션 레이어의 다운샘플링을 위한 필터의 수는 64, 128, 256, 512, 512, 512이다. 업샘플링 컨볼루션 레이어를 위한 필터 수는 512, 512, 256, 128, 64, 4이다. 각 컨볼루션 레이어 뒤에는 인스턴스 정규화 레이어 [33]와 Leaky ReLU [24]가 있으며, 그 중 기울기는 0.2로 설정된다.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-4.PNG)

> Fig. 4. Matching results of SCMM and GMM. Warped clothes are directly pasted onto target persons for visual checking. Our method is comparable with SCMM and produces less weird results.
>> 그림 4. SCMM과 GMM의 일치 결과. 뒤틀린 옷은 육안 확인을 위해 대상자에게 직접 붙인다. 우리의 방법은 SCMM과 비슷하고 덜 이상한 결과를 만들어낸다.

#### $\mathbf{4.4\;Comparison\;of\;Warping\;Results}$

> Shape Context Matching Module (SCMM) uses hand-crafted descriptors and explicitly computes their correspondences using an iterative algorithm, which is time-consumed, while GMM runs much faster. In average, processing a sample pair takes GMM 0.06s on GPU, 0.52s on CPU, and takes SCMM 2.01s on CPU.
>> SCMM(Shape Context Matching Module)은 수작업으로 만든 설명자를 사용하고 시간이 많이 걸리는 반복 알고리듬을 사용하여 명시적으로 대응 관계를 계산하지만 GMM은 훨씬 더 빨리 실행된다. 평균적으로 샘플 쌍을 처리하려면 GPU에서 GMM 0.06s, CPU에서 0.52s, CPU에서 SCMM 2.01s가 소요됩니다.

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-5.PNG)

> Fig. 5. Qualitative comparisons of VITON and CP-VTON. Our CP-VTON successfully preserve key details of in-shop clothes.
>> 그림 5. VITON과 CP-VTON의 정성적 비교 저희 CP-VTON은 매장 내 옷의 주요 디테일을 성공적으로 보존하고 있습니다.

> **Qualitative results** Fig. 4 demonstrates a qualitative comparison of SCMM and GMM. It shows that both modules are able to roughly align clothes with target person pose. However, SCMM tends to overly shrink a long sleeve into a “thin band”, as shown in the 6-th column in Fig. 4. This is because SCMM merely relies on matched shape context descriptors on the boundary of cloths shape, while ignores the internal structures. Once there exist incorrect correspondences of descriptors, the warping results will be weird. In contrast, GMM takes full advantages of the learned rich representation of clothes and person images to determinate TPS transformation parameters and more robust for large shape differences.
>> **정적 결과** 그림 4는 SCMM과 GMM을 정성적으로 비교한 것으로, 두 모듈 모두 대상자의 자세에 맞춰 옷을 대략 정렬할 수 있음을 보여준다. 그러나, SCMM은 그림 4의 6번째 열과 같이 긴 소매를 "얇은 밴드"로 지나치게 수축시키는 경향이 있다. 이는 SCMM이 내부 구조를 무시하면서 옷 모양 경계에서 일치하는 모양 컨텍스트 설명자에만 의존하기 때문이다. 설명자의 잘못된 대응이 존재하면 뒤틀림 결과가 이상해집니다. 대조적으로, GMM은 TPS 변환 매개 변수를 결정하기 위해 옷과 사람 이미지의 학습된 풍부한 표현을 최대한 활용하고 큰 형상 차이에 대해 더 강력하다.

> **Quantitative results** It is difficult to evaluate directly the quantitative performance of matching modules due to the lack of ground truth in the testing phase. Nevertheless, we can simply paste the warped clothes onto the original person image as a non-parametric warped synthesis method in [10]. We conduct a perceptual user study following the protocol described in Sec. 4.2, for these two warped synthesis methods. The synthesized by GMM are rated more realistic in 49.5% and 42.0% for LARGE and SMALL, which indicates that GMM is comparable to SCMM for shape alignment.
>> **정량적 결과** 시험단계에서 접지실증이 부족하여 매칭모듈의 정량적 성능을 직접 평가하기는 어렵다. 그럼에도 불구하고 [10]에서 비모수 왜곡 합성 방법으로 왜곡된 옷을 원래 인물 이미지에 간단히 붙여넣을 수 있다. 우리는 이 두 가지 왜곡된 합성 방법에 대해 4.2절에 설명된 프로토콜을 따라 지각적 사용자 연구를 수행한다. GMM에 의해 합성된 결과는 LARGE와 SMART의 경우 49.5%, LARGE와 42.0%로 보다 사실적으로 평가되며, 이는 GMM이 형상 정렬에 있어 SCMM과 유사하다는 것을 보여준다.

#### $\mathbf{4.5\;Comparison\;of\;Try-on\;Results}$

> **Qualitative results** Fig. 2 shows that our pipeline performs roughly the same as VITON when the patterns of target clothes are simpler. However, our pipeline preserves sharp and intact characteristic on clothes with rich details (e.g. texture, logo, embroidery) while VITON produces blurry results.
>> **정적 결과** 그림 2는 대상 옷의 패턴이 단순할 때 파이프라인이 VITON과 거의 동일한 성능을 발휘함을 보여준다. 그러나 우리의 파이프라인은 풍부한 디테일(예: 텍스처, 로고, 자수)이 있는 옷에 날카롭고 온전한 특성을 보존하는 반면, VITON은 흐릿한 결과를 생성한다.

> We argue that the failure of VITON lies in its coarse-to-fine strategy and the imperfect matching module. Precisely, VITON learns to synthesis a coarse person image at first, then to align the clothes with target person with shape context matching, then to produce a composition mask for fusing UNet rendered person with warped clothes and finally producing a refined result. After extensive training, the rendered person image has already a small VGG perceptual loss with respect to ground truth. On the other hand, the imperfect matching module introduces unavoidable minor misalignment between the warped clothes and ground truth, making the warped clothes unfavorable to perceptual loss. Taken together, when further refined by truncated perceptual loss, the composition mask will be biased towards selecting rendered person image rather than warped clothes, despite the regularization of the composition mask(Eq. 4). The VITON’s “ragged” masks shown in Fig. 6 confirm this argument.
>> 우리는 VITON의 실패는 거칠고 미세한 전략과 불완전한 매칭 모듈에 있다고 주장한다. 정확하게, VITON은 처음에 거친 사람 이미지를 합성한 다음, 모양 컨텍스트 매칭으로 대상 사람과 옷을 정렬한 다음 UNet 렌더링된 사람과 뒤틀린 옷을 융합하고 최종적으로 정제된 결과를 생성하는 합성 마스크를 생성하는 방법을 배운다. 광범위한 훈련을 거친 후 렌더링된 사람 이미지는 이미 지상 진실에 대한 작은 VGG 지각 손실을 가지고 있다. 반면, 불완전한 매칭 모듈은 왜곡된 옷과 지상 진실 사이에 피할 수 없는 사소한 오정렬을 도입하여 왜곡된 옷을 지각 손실에 불리하게 만든다. 종합해 보면, 잘린 지각 손실에 의해 더욱 정제될 때 합성 마스크는 합성 마스크의 정규화(Eq.4)에도 불구하고 뒤틀린 옷보다는 렌더링된 사람 이미지를 선택하는 데 편향될 것이다. 그림 6에 표시된 바이톤의 "raged" 마스크는 이러한 주장을 뒷받침한다.

![Table1, Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-6.PNG)

> Table 1. Results of pairwise comparisons of images synthesized with LARGE and SMALL clothes by different models. Each column compares our approach with one of the baselines. Higher is better. The random chance is at 50%.
>> 표 1. 서로 다른 모델에 의해 LARGE 및 SLARGE 의상으로 합성된 이미지의 쌍별 비교 결과. 각 열은 우리의 접근 방식을 기준 중 하나와 비교한다. 높을수록 좋다. 무작위 확률은 50%입니다.

> Fig. 6. An example of VITON stage II. The composition mask tends to ignore the details of coarsely aligned clothes.
>> 그림 6. VITON 단계 II의 예. 컴포지션 마스크는 거칠게 정렬된 옷의 디테일을 무시하는 경향이 있다.

> Our pipeline doesn’t address the aforementioned issue by improving matching results, but rather sidesteps it by simultaneously learning to produce a UNet rendered person image and a composition mask. Before the rendered person image becomes favorable to loss function, the central clothing region of composition mask is biased towards warped clothes because it agrees more with ground truth in the early training stage. It is now the warped clothes rather than the rendered person image that takes the early advantage in the competition of mask selection. After that, the UNet learns to adaptively expose regions where UNet rendering is more suitable than directly pasting. Once the regions of hair and arms are exposed, rendered and seamlessly fused with warped clothes.
>> 우리의 파이프라인은 매칭 결과를 개선하여 앞서 언급한 문제를 해결하지 않고, UNet 렌더링된 인물 이미지와 합성 마스크를 동시에 생성하는 방법을 학습함으로써 이를 회피한다. 렌더링된 인물 이미지가 손실 기능에 유리해지기 전에 구성 마스크의 중앙 의복 영역은 초기 훈련 단계에서 지상 사실과 더 일치하기 때문에 왜곡된 의복에 치우친다. 마스크 선택 경쟁에서 일찌감치 우위를 점하는 것은 렌더링된 인물 이미지보다는 뒤틀린 옷이다. 그 후, UNet은 직접 붙여넣기보다 UNet 렌더링이 더 적합한 영역을 적응적으로 노출하는 방법을 학습한다. 일단 머리카락과 팔의 영역이 노출되면, 변형된 옷과 함께 렌더링되고 매끄럽게 융합됩니다.

> **Quantitative results** The first column of Table 1 shows that our pipeline surpasses VITON in the preserving the details of clothes using identical person representation. According to the table, our approach performs better than other methods, when dealing with rich details clothes.
>> **정량적 결과* 표 1의 첫 번째 열은 동일 인물 표현을 사용하여 옷의 디테일을 보존하는 데 있어 우리의 파이프라인이 VITON을 능가한다는 것을 보여준다. 표에 따르면, 우리의 접근 방식은 풍부한 디테일 의상을 다룰 때 다른 방법보다 더 잘 수행된다.

#### $\mathbf{4.6\;Discussion\;and\;Ablation\;Studies}$

> Effects of composition mask To empirically justify the design of composition mask and mask L1 regularization (Eq. 4) in our pipeline, we compare it with two variants for ablation studies: (1): mask composition is also removed and the final results are directly rendered by UNet as CP-VTON(w/o mask). (2): the mask composition is used but the mask L1 regularization is removed as CP-VTON(w/o L1 Loss);
>> 합성 마스크의 효과 파이프라인에서 합성 마스크와 마스크 L1 정규화(Eq.4)의 설계를 경험적으로 정당화하기 위해 절제 연구를 위한 두 가지 변형과 비교한다. (1) 마스크 합성도 제거되고 최종 결과는 UNet에 의해 CP-VTON(마스크 없음)으로 직접 렌더링된다. (2): 마스크 조성물이 사용되지만 마스크 L1 정규화는 CP-VTON(L1 손실 없음)으로 제거됩니다.

> As shown in Fig. 6, even though the warped clothes are roughly aligned with target person, CP-VTON(w/o mask) still loses characteristic details and produces blurry results. This verifies that encoder-decoder network architecture like UNet fails to handle even minor spatial deformation.
>> 그림 6과 같이 뒤틀린 옷이 대상자와 대략 정렬되어 있음에도 불구하고 CP-VTON(마스크 미포함)은 여전히 특징적인 디테일을 잃고 흐릿한 결과가 나타난다. 이는 UNet과 같은 인코더-디코더 네트워크 아키텍처가 사소한 공간 변형도 처리하지 못한다는 것을 검증한다.

![Figure 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-7.PNG)

> Fig. 7. Ablation studies on composition mask and mask L1 loss. Without mask composition, UNet cannot handle well even minor misalignment and produces undesirable try-on results. Without L1 regularization on mask, it tends to select UNet-rendered person, leading to blurry results as well.
>> 그림 7. 합성 마스크 및 마스크 L1 손실에 대한 절제 연구 마스크 구성이 없으면 UNet은 사소한 정렬 오류도 잘 처리할 수 없으며 바람직하지 않은 트라이온 결과를 생성합니다. 마스크에서 L1 정규화를 하지 않으면 UNet-rendered를 선택하는 경향이 있어 결과도 흐리게 된다.

> Though integrated with mask composition, CP-VTON(no L1) performs as poorly as variant CP-VTON(w/o mask. Fig. 7 shows that composition mask tends to select rendered person image without L1 regularization. This verifies that even minor misalignment introduces large perceptual disagreement between warped clothes and ground truth.
>> 마스크 구성과 통합되지만 CP-VTON(No L1)은 변형 CP-VTON(마스크 없음)만큼 성능이 떨어진다. 그림 7은 합성 마스크가 L1 정규화 없이 렌더링된 인물 이미지를 선택하는 경향이 있음을 보여준다. 이것은 심지어 사소한 정렬 오류도 뒤틀린 옷과 지상 진실 사이에 큰 지각적 불일치를 초래한다는 것을 검증한다.

> **Robustness against** minor misalignment In Sec. 4.5 we argue that VITON is vulnerable to minor misalignment due to its coarse-to-fine strategy, while our pipeline sidesteps imperfect alignment by simultaneously producing rendered person and composition mask. This is further clarified below in a controlled condition with simulated warped clothes. 
>> **Robustness against** 경미한 정렬 오류에 대한 견고성 4.5절에서 우리는 VITON이 거친 대 미세 전략으로 인해 경미한 정렬에 취약하다고 주장하는 반면, 파이프라인 쪽은 렌더링된 사람과 합성 마스크를 동시에 생산하여 불완전한 정렬을 밟는다. 이는 시뮬레이션된 뒤틀린 의복으로 제어된 조건에서 아래에서 더욱 명확히 설명된다.

> Specifically, rather than real warped clothes produced by matching module, we use the wore clothes collected from person images to simulate perfect alignment results. We then train VITON stage II, our proposed variant CPVTON(w/o mask) and our pipeline. For VITON stage II, we synthesize coarse person image with its source code and released model checkpoint.
>> 구체적으로 매칭 모듈에서 제작한 실제 뒤틀린 옷보다는 인물 이미지에서 수집한 착용 옷을 사용하여 완벽한 정렬 결과를 시뮬레이션한다. 그런 다음 VITON 단계 II, 제안된 변형 CPVTON(마스크 없음) 및 파이프 라인을 훈련한다. VITON 단계 II의 경우, 우리는 소스 코드와 공개된 모델 체크포인트로 거친 인물 이미지를 합성한다.

> It is predictable that with this “perfect matching module”, all the three methods could achieve excellent performance in training and validation phase, where input samples are paired. Next is the interesting part: what if the perfect alignment is randomly perturbed within a range of N pixels, to simulate an imperfect matching module? With the perturbation getting greater (N=0, 5, 10, 15, 20) , how fast will the try-on performance decay?
>> 이 "완벽한 매칭 모듈"을 사용하면 세 가지 방법 모두 입력 샘플이 쌍을 이루는 훈련 및 검증 단계에서 우수한 성능을 달성할 수 있을 것으로 예측된다. 다음은 흥미로운 부분입니다. 완벽한 정렬이 N 픽셀 범위 내에서 무작위로 교란되면 불완전한 일치 모듈을 시뮬레이션할 수 있습니다. 섭동이 커짐에 따라 (N=0, 5, 10, 15, 20) 평가판 성능이 얼마나 빨리 저하될까요?
![Figure 8](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-8.PNG)

> Fig. 8. Comparisons on the robustness of three methods against minor misalignment simulated by random shift within radius N. As N increasing, results of CP-VTON decays more slightly than other methods.
>> 그림 8. 반경 N 내의 무작위 이동에 의해 시뮬레이션된 사소한 정렬 오류에 대한 세 가지 방법의 견고성 비교 N이 증가할수록 CP-VTON의 결과는 다른 방법보다 약간 감소한다.

![Figure 9](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-07-25-(VITON)Toward-Characteristic-Preserving-Image-based-Virtual-Try-On-Network/Figure-9.PNG)

> Fig. 9. Some failure cases of our CP-VTON.
>> 그림 9. CP-VTON의 몇 가지 실패 사례

> These questions are answered in Fig. 8. As we applying greater perturbation, the performance of both VITON stage II and CP-VTON(w/o mask) decays quickly. In contrast, our pipeline shows robustness against perturbation and manages to preserve detailed characteristic.
>> 이러한 질문에 대한 답은 그림 8에 나와 있다. 더 큰 섭동을 적용함에 따라 VITON 단계 II와 CP-VTON(마스크 미포함)의 성능이 모두 빠르게 저하된다. 대조적으로, 우리의 파이프라인은 섭동에 대한 견고성을 보여주며 세부적인 특성을 보존한다.

> **Failure cases** Fig. 9 shows three failure cases of our CP-VTON method caused by (1) improperly preserved shape information of old clothes, (2) rare poses and (3) inner side of the clothes undistinguishable from the outer side, respectively.
>> **고장사례** 그림 9는 (1) 헌옷의 형상정보가 부적절하게 보존되어 있는 점, (2) 희귀한 포즈, (3) 옷의 안쪽과 바깥쪽을 구별할 수 없는 점으로 인해 발생한 CP-VTON 방법의 3가지 고장사례이다.

### $\mathbf{5\;Conclusions}$

> In this paper, we propose a fully learnable image-based virtual try-on pipeline towards the characteristic-preserving image generation, named as CP-VTON, including a new geometric matching module and a try-on module with the new merging strategy. The geometric matching module aims at aligning in-shop clothes and target person body with large spatial displacement. Given aligned clothes, the try-on module learns to preserve well the detailed characteristic of clothes. Extensive experiments show the overall CP-VTON pipeline produces high-fidelity virtual try-on results that retain well key characteristics of in-shop clothes. Our CP-VTON achieves state-of-the-art performance on the dataset collected by Han et al. [10] both qualitatively and quantitatively.
>> 본 논문에서는 새로운 기하학적 매칭 모듈과 새로운 병합 전략을 사용한 트라이온 모듈을 포함하여 CP-VTON이라는 특성 보존 이미지 생성을 위한 완전히 학습 가능한 이미지 기반 가상 트라이온 파이프라인을 제안한다. 기하학적 매칭 모듈은 공간 변위가 큰 매장 내 옷과 대상 인체를 정렬하는 것을 목표로 한다. 정렬된 옷이 주어지면, 트라이온 모듈은 옷의 세부 특성을 잘 보존하는 법을 배운다. 광범위한 실험에 따르면 전체 CP-VTON 파이프라인은 매장 내 옷의 핵심 특성을 잘 유지하는 충실도가 높은 가상 체험 결과를 생성한다. 우리의 CP-VTON은 Han 등이 수집한 데이터 세트에서 최첨단 성능을 달성한다. [10] 질적으로나 양적으로나

---

### $\mathbf{References}$

<a href="#footnote_1_2" name="footnote_1_1">[1]</a> Anguelov, D., Srinivasan, P., Koller, D., Thrun, S., Rodgers, J., Davis, J.: Scape: shape completion and animation of people. In: ACM transactions on graphics (TOG). vol. 24, pp. 408–416. ACM (2005)

<a href="#footnote_2_2" name="footnote_2_1">[2]</a> Belongie, S., Malik, J., Puzicha, J.: Shape matching and object recognition using shape contexts. IEEE transactions on pattern analysis and machine intelligence 24(4), 509–522 (2002)

<a href="#footnote_3_2" name="footnote_3_1">[3]</a> Chen, Q., Koltun, V.: Photographic image synthesis with cascaded refinement networks. In: The IEEE International Conference on Computer Vision (ICCV). vol. 1 (2017)

<a href="#footnote_4_2" name="footnote_4_1">[4]</a> Chen, W., Wang, H., Li, Y., Su, H., Wang, Z., Tu, C., Lischinski, D., Cohen-Or, D., Chen, B.: Synthesizing training images for boosting human 3d pose estimation. In: 3D Vision (3DV), 2016 Fourth International Conference on. pp. 479–488. IEEE (2016)

<a href="#footnote_5_2" name="footnote_5_1">[5]</a> Choi, Y., Choi, M., Kim, M., Ha, J.W., Kim, S., Choo, J.: Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. arXiv preprint arXiv:1711.09020 (2017)

<a href="#footnote_6_2" name="footnote_6_1">[6]</a> Deng, Z., Zhang, H., Liang, X., Yang, L., Xu, S., Zhu, J., Xing, E.P.: Structured generative adversarial networks. In: Advances in Neural Information Processing Systems. pp. 3899–3909 (2017)

<a href="#footnote_7_2" name="footnote_7_1">[7]</a> Fischler, M.A., Bolles, R.C.: Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. In: Readings in computer vision, pp. 726–740. Elsevier (1987)

<a href="#footnote_8_2" name="footnote_8_1">[8]</a> Gong, K., Liang, X., Shen, X., Lin, L.: Look into person: Self-supervised structuresensitive learning and a new benchmark for human parsing. arXiv preprint arXiv:1703.05446 (2017)

<a href="#footnote_9_2" name="footnote_9_1">[9]</a> Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: Advances in neural information processing systems. pp. 2672–2680 (2014)

<a href="#footnote_10_2" name="footnote_10_1">[10]</a> Han, X., Wu, Z., Wu, Z., Yu, R., Davis, L.S.: Viton: An image-based virtual try-on network. arXiv preprint arXiv:1711.08447 (2017)

<a href="#footnote_11_2" name="footnote_11_1">[11]</a> He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR. pp. 770–778 (2016)

<a href="#footnote_12_2" name="footnote_12_1">[12]</a> Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. arXiv preprint (2017)

<a href="#footnote_13_2" name="footnote_13_1">[13]</a> Jetchev, N., Bergmann, U.: The conditional analogy gan: Swapping fashion articles on people images. arXiv preprint arXiv:1709.04695 (2017)

<a href="#footnote_14_2" name="footnote_14_1">[14]</a> Johnson, J., Alahi, A., Fei-Fei, L.: Perceptual losses for real-time style transfer and super-resolution. In: ECCV. pp. 694–711 (2016)

<a href="#footnote_15_2" name="footnote_15_1">[15]</a> Kinga, D., Adam, J.B.: A method for stochastic optimization. In: International Conference on Learning Representations (ICLR) (2015)

<a href="#footnote_16_2" name="footnote_16_1">[16]</a> Lamdan, Y., Schwartz, J.T., Wolfson, H.J.: Object recognition by affine invariant matching. In: Computer Vision and Pattern Recognition, 1988. Proceedings CVPR’88., Computer Society Conference on. pp. 335–344. IEEE (1988)

<a href="#footnote_17_2" name="footnote_17_1">[17]</a> Lassner, C., Pons-Moll, G., Gehler, P.V.: A generative model of people in clothing. arXiv preprint arXiv:1705.04098 (2017)

<a href="#footnote_18_2" name="footnote_18_1">[18]</a> Li, J., Liang, X., Wei, Y., Xu, T., Feng, J., Yan, S.: Perceptual generative adversarial networks for small object detection. In: IEEE CVPR (2017)

<a href="#footnote_19_2" name="footnote_19_1">[19]</a> Liang, X., Lee, L., Dai, W., Xing, E.P.: Dual motion gan for future-flow embedded video prediction. In: IEEE International Conference on Computer Vision (ICCV). vol. 1 (2017)

<a href="#footnote_20_2" name="footnote_20_1">[20]</a> Liang, X., Zhang, H., Xing, E.P.: Generative semantic manipulation with contrasting gan. arXiv preprint arXiv:1708.00315 (2017)

<a href="#footnote_21_2" name="footnote_21_1">[21]</a> Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: CVPR. pp. 3431–3440 (2015)

<a href="#footnote_22_2" name="footnote_22_1">[22]</a> Lowe, D.G.: Distinctive image features from scale-invariant keypoints. International journal of computer vision 60(2), 91–110 (2004)

<a href="#footnote_23_2" name="footnote_23_1">[23]</a> Ma, L., Jia, X., Sun, Q., Schiele, B., Tuytelaars, T., Van Gool, L.: Pose guided person image generation. In: Advances in Neural Information Processing Systems. pp. 405–415 (2017)

<a href="#footnote_24_2" name="footnote_24_1">[24]</a> Maas, A.L., Hannun, A.Y., Ng, A.Y.: Rectifier nonlinearities improve neural network acoustic models. In: Proc. icml. vol. 30, $p$. 3 (2013)

<a href="#footnote_25_2" name="footnote_25_1">[25]</a> Odena, A., Dumoulin, V., Olah, C.: Deconvolution and checkerboard artifacts. Distill 1(10), e3 (2016)

<a href="#footnote_26_2" name="footnote_26_1">[26]</a> Pons-Moll, G., Pujades, S., Hu, S., Black, M.J.: Clothcap: Seamless 4d clothing capture and retargeting. ACM Transactions on Graphics (TOG) 36(4), 73 (2017)

<a href="#footnote_27_2" name="footnote_27_1">[27]</a> Rocco, I., Arandjelovic, R., Sivic, J.: Convolutional neural network architecture for geometric matching. In: Proc. CVPR. vol. 2 (2017)

<a href="#footnote_28_2" name="footnote_28_1">[28]</a> Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

<a href="#footnote_29_2" name="footnote_29_1">[29]</a> Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.: Improved techniques for training gans. In: NIPS. pp. 2234–2242 (2016)

<a href="#footnote_30_2" name="footnote_30_1">[30]</a> Sekine, M., Sugita, K., Perbet, F., Stenger, B., Nishiyama, M.: Virtual fitting by single-shot body shape estimation. In: Int. Conf. on 3D Body Scanning Technologies. pp. 406–413. Citeseer (2014)

<a href="#footnote_31_2" name="footnote_31_1">[31]</a> Siarohin, A., Sangineto, E., Lathuiliere, S., Sebe, N.: Deformable gans for posebased human image generation. arXiv preprint arXiv:1801.00055 (2017) 

<a href="#footnote_32_2" name="footnote_32_1">[32]</a> Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)

<a href="#footnote_33_2" name="footnote_33_1">[33]</a> Ulyanov, D., Vedaldi, A., Lempitsky, V.: Improved texture networks: Maximizing quality and diversity in feed-forward stylization and texture synthesis. In: Proc. CVPR (2017)\

<a href="#footnote_34_2" name="footnote_34_1">[34]</a> Wang, T.C., Liu, M.Y., Zhu, J.Y., Tao, A., Kautz, J., Catanzaro, B.: Highresolution image synthesis and semantic manipulation with conditional gans. arXiv preprint arXiv:1711.11585 (2017)

<a href="#footnote_35_2" name="footnote_35_1">[35]</a> Yang, L., Liang, X., Xing, E.: Unsupervised real-to-virtual domain unification for end-to-end highway driving. arXiv preprint arXiv:1801.03458 (2018)

<a href="#footnote_36_2" name="footnote_36_1">[36]</a> Yoo, D., Kim, N., Park, S., Paek, A.S., Kweon, I.S.: Pixel-level domain transfer. In: European Conference on Computer Vision. pp. 517–532. Springer (2016)

<a href="#footnote_37_2" name="footnote_37_1">[37]</a> Zhao, B., Wu, X., Cheng, Z.Q., Liu, H., Feng, J.: Multi-view image generation from a single-view. arXiv preprint arXiv:1704.04886 (2017)

<a href="#footnote_38_2" name="footnote_38_1">[38]</a> Zhu, J.Y., Park, T., Isola, P., Efros, A.A.: Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint arXiv:1703.10593 (2017)

<a href="#footnote_39_2" name="footnote_39_1">[39]</a> Zhu, S., Fidler, S., Urtasun, R., Lin, D., Loy, C.C.: Be your own prada: Fashion synthesis with structural coherence. arXiv preprint arXiv:1710.07346 (2017)