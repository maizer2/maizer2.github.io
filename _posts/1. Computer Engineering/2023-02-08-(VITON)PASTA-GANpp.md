---
layout: post
title: "(VITON)PASTA-GAN++: A Versatile Framework for High-Resolution Unpaired Virtual Try-on"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.7. Paper Review, 1.2.2.8. Virtual Try-on]
---

### [CV Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/08/01/paper-of-VITON.html)

# PASTA-GAN++: A Versatile Framework for High-Resolution Unpaired Virtual Try-on

## Abstract

> Image-based virtual try-on is one of the most promising applications of human-centric image generation due to its tremendous real-world potential. 
>> 이미지 기반 가상 트라이온은 엄청난 실제 잠재력으로 인해 인간 중심 이미지 생성의 가장 유망한 응용 프로그램 중 하나이다.이미지 기반 가상 트라이온은 엄청난 실제 잠재력(tremendous)으로 인해 인간 중심(human-centric) 이미지 생성의 가장 유망한 응용 프로그램 중 하나이다.

> In this work, we take a step forwards to explore versatile virtual try-on solutions, which we argue should possess three main properties, namely, they should support unsupervised training, arbitrary garment categories, and controllable garment editing.
>> 이 연구에서, 우리는 다목적(versatile) 가상 트라이온 솔루션을 탐구하기 위해 한 걸음 더 나아간다(take a step forwards to). 우리는 세 가지 주요 속성, 즉 감독되지 않은 훈련(unsupervised training), 임의의 의류 범주(arbitrary garment categories) 및 제어 가능한 의류 편집(controllable garment editing)을 지원해야 한다고 주장한다.

> To this end, we propose a characteristic-preserving end-to-end network, the PAtch-routed SpaTially-Adaptive GAN++ (PASTA-GAN++), to achieve a versatile system for high-resolution unpaired virtual try-on. 
>> 이를 위해, 우리는 고해상도 짝을 이루지 않은 가상 트라이온을 위한 다목적 시스템을 달성하기 위해 특성 보존 종단 간 네트워크(a characteristic-preserving end-to-end network)인 PAtch-routed SpaTially-Adaptive GAN++ (PASTA-GAN++)를 제안한다.

> Specifically, our PASTA-GAN++ consists of an innovative patch-routed disentanglement module to decouple the intact garment into normalized patches, which is capable of retaining garment style information while eliminating the garment spatial information, thus alleviating the overfitting issue during unsupervised training.
>> 특히, 우리의 PASTA-GAN++는 손상되지 않은 의복(intact garment)을 정규화된 패치(normalized patches)로 분리하기 위한(to decouple) 혁신적인(innovative) 패치 라우팅(patch-routed) 분리 모듈(disentanglement module)로 구성되어(consists) 있으며, 이 모듈은 의복 스타일 정보를 유지하면서(retaining garment style information) 의복 공간 정보를 제거(eliminating the garment spatial information)할 수 있으므로 감독되지 않은 훈련(unsupervised training) 중 과적합 문제(overfitting issue)를 완화(alleviating)할 수 있다.

> Furthermore, PASTA-GAN++ introduces a patch-based garment representation and a patch-guided parsing synthesis block, allowing it to handle arbitrary garment categories and support local garment editing. 
>> 또한 PASTA-GAN++는 패치 기반 의류 표현(a patch-based garment representation)과 패치 가이드 구문 분석 합성 블록(a patch-guided parsing synthesis block)을 도입하여 임의의 의류 범주(arbitrary garment categories)를 처리하고 로컬 의류 편집(local garment editing)을 지원할 수 있다.

> Finally, to obtain try-on results with realistic texture details, PASTA-GAN++ incorporates a novel spatially-adaptive residual module to inject the coarse warped garment feature into the generator. 
>> 마지막으로, 사실적인 질감 세부 정보와 함께 트라이온 결과를 얻기 위해, PASTA-GAN++는 새로운 공간 적응형 잔류 모듈(a novel spatially-adaptive residual module)을 통합하여(incorporates) 거친 뒤틀린 의복 기능(the coarse warped garment feature)을 생성기에 주입한다.

> Extensive experiments on our newly collected UnPaired virtual Try-on (UPT) dataset demonstrate the superiority of PASTA-GAN++ over existing SOTAs and its ability for controllable garment editing.
>> 새로 수집된 UPT(Unpaired Virtual Try-on) 데이터 세트에 대한 광범위한 실험은 기존 SOTA에 비해 PASTA-GAN++의 우수성과 제어 가능한 의복 편집 능력을 보여준다.

## 1. Introduction

> IMAGE-BASED virtual try-on, the process of computationally transferring a garment onto a particular person in a query image, is one of the most promising applications of human-centric image generation with the potential to revolutionize shopping experiences and reduce purchase returns.
>> 쿼리(query) 이미지에서 특정 사람에게 의복을 계산적으로 전송하는 프로세스인 IMAGE-BASED virtual try-on은 쇼핑 경험을 혁신하고 구매 수익을 줄일 수 있는 잠재력을 가진 human-centric 이미지 생성의 가장 유망한 응용 프로그램 중 하나이다.

> The computer vision community has recently witnessed the increasing development of image-based virtual try-on, and numerous impressive works [1]–[4], [6], [7], [12], [13], [16]–[18] have been proposed to synthesize photo-realistic try-on results on the publicly available benchmarks [2], [5], [19], [20]. 
>> 컴퓨터 비전 커뮤니티는 최근 이미지 기반 가상 트라이온의 발전을 목격했으며(witnessed), 공개적으로 사용 가능한 벤치마크 [2], [5], [19], [20]에서 사실적인 트라이온 결과를 합성하기 위해 수많은 인상적인 작업[1]–[4], [6], [7], [12], [13], [16]–[18]이 제안되었다.

> However, to fully exploit its potential, versatile virtual try-on solutions are required, which possess the following three properties: 
>> 그러나 잠재력을 최대한 활용하려면 다음과 같은 세 가지 특성을 가진 다목적(versatile) 가상 트라이온 솔루션이 필요합니다:

> First, the training procedure of the virtual try-on network should make full use of the easily accessible unpaired fashion model images from e-commerce websites, which means that the network should be easily trainable in an unsupervised manner. 
>> 첫째, 가상 트라이온 네트워크의 훈련 절차는 전자 상거래 웹 사이트에서 쉽게 액세스할 수 있는 짝을 이루지 않은(unpaired) 패션 모델 이미지를 최대한 활용해야 하며, 이는 네트워크가 감독되지 않은 방식(unsupervised manner)으로 쉽게 훈련될 수 있어야 한다는 것을 의미한다.

> Second, a versatile virtual try-on solution should be capable of handling arbitrary garment categories (e.g., long/short sleeve shirt, vest, sling, pants, shorts, skirts, dresses, etc.) within a single pre-trained model.
>> 둘째, 다목적 가상 시험 솔루션(versatile virtual try-on solution)은 사전 훈련된 단일 모델(single pre-trained model) 내에서 임의의 의류 범주(예: 긴/짧은 소매 셔츠, 조끼, 슬링, 바지, 반바지, 치마, 드레스 등)를 처리할 수 있어야 한다.

> Third, it needs to support auxiliary applications (e.g., dressing order controlling, shape/texture editing, local shape editing, etc.) to fulfill various requirements of dressing styles or garment editing in real-world scenarios. 
>> 셋째, 실제 시나리오에서 복장 스타일 또는 의복 편집의 다양한 요구 사항을 충족하기 위해 보조 애플리케이션(예: 복장 순서 제어(dressing order controlling), 모양/ 질감 편집(shape/texture editing), 로컬 모양 편집(local shape editing) 등)을 지원해야 한다.

> Unfortunately, as shown in Table 1, to date, few methods can achieve all of the above three requirements simultaneously. 
>> 불행하게도, 표 1에 나타난 바와 같이, 현재까지 위의 세 가지 요구사항을 모두 동시에 달성할 수 있는 방법은 거의 없다.

> Most existing methods [2]–[10], [12], [13] rely on paired training data from curated academic datasets [2], [5], [19], [20], in which each data pair is composed of a person image and its corresponding in-shop garment or of images of the same person captured in different body poses, resulting in laborious data-collection and post-processing processes.
>> 대부분의 기존 방법 [2]–[10], [12], [13]은 큐레이티드 학술 데이터 세트 [2], [5], [19], [20]의 쌍을 이룬 훈련 데이터(paired training data)에 의존한다. 각 데이터 쌍은 개인 이미지(person image)와 해당 매장 내 의류(corresponding in-shop garment) 또는 다른 신체 자세로 캡처된 동일인의 이미지(images of the same person captured in different body poses)로 구성되어 있어(composed of) 힘든(laborious) 데이터 수집 및 후 처리 과정을 초래한다.

> Besides, most existing methods [2]–[7] only focus on upper body virtual try-on and fail to support arbitrary categories within a single pre-trained model. Furthermore, most existing methods [8], [9], [12], [14], [15] neglect the dressing styles and lack the ability to edit the garment texture/shape, which largely limits their application potential.
>> 또한 대부분의 기존 방법 [2]–[7]은 상체(upper) 가상 트라이온에만 초점을 맞추고 사전 훈련된 단일 모델 내에서 임의 범주를 지원하지 못한다. 또한, 대부분의 기존 방법[8], [9], [12], [14], [15]은 드레싱 스타일을 무시하고 의복 질감/모양을 편집하는 능력이 부족하여 적용 가능성이 크게 제한된다.

> While unpaired solutions have recently started to emerge, performing virtual try-on in an unsupervised setting is extremely challenging and tends to affect the visual quality of the try-on results. 
>> 최근 쌍을 이루지 않은(unpaired solutions) 솔루션이 등장하기 시작했지만, 감독되지 않은 환경(unsupervised setting)에서 가상 트라이온을 수행하는 것은 매우 어렵고 트라이온 결과의 시각적 품질에 영향을 미치는 경향이 있다(tends to affect). 

> Specifically, without access to the paired data, these models are usually trained by reconstructing the same person image, which is prone to over-fitting, and they thus  underperform when handling garment transfer during testing. 
>> 특히, 쌍을 이룬 데이터에 액세스하지 않으면 이러한 모델은 대개 과적합(over-fitting)되기 쉬운(prone) 동일 인물 이미지를 재구성하여 훈련되므로 테스트 중에 의복 전송을 처리할 때 성능이 떨어진다. 

> The performance discrepancy is mainly reflected in the garment synthesis results, in particular the shape and texture, which we argue is caused by the entanglement of the garment style (i.e., color, category) and spatial (i.e.,the location, orientation, and relative size of the garment in the model image) representations in the  synthesis network during the reconstruction process.
>> 성능 불일치(performance discrepancy)는 주로 의복 합성 결과, 특히 모양과 질감에 반영되는데, 우리는 재구성 과정에서 의복 스타일(즉, 색상, 범주)와 공간(즉, 모델 이미지에서 의복의 위치, 방향 및 상대적 크기) 표현이 합성 네트워크에서 얽힘으로써 발생한다고 주장한다.

> While traditional paired try-on approaches, such as the warping-based methods [3], [4], [6], [10], [12], [13], [16] avoid the problem and preserve the garment characteristics by utilizing a supervised warping network to deform the garment into target shape, this is not possible in the unpaired setting due to the lack of the warped ground truth. 
>> 워핑 기반 방법(warping-based methods)[3], [4], [6], [10], [12], [13], [16]과 같은 traditional paired try-on approaches은 문제를 피하고 supervised warping network를 사용하여 의복을 대상 모양으로 변형하여 의복 특성을 보존하지만, warped ground truth가 없기 때문에 unpaired setting에서는 불가능하다. 

> Similarly, warping-free methods [9], [11], [21], [22], which choose to circumvent this problem by using person images in various poses as training data and taking pose transfer as pretext task to disentangle the garment feature from the intrinsic body pose, also require a laborious data-collection process for paired data, largely limiting the scalability of network training. 
>> 마찬가지로, 뒤틀림이 없는 방법[9], [11], [21], [22]은 다양한 포즈의 사람 이미지를 훈련 데이터로 사용하고 의복 특징을 본질적인 신체 포즈에서 분리하기 위한 구실 작업으로 포즈 전송을 취함으로써 이 문제를 피하기로 선택하며, 또한 쌍을 이룬 데이터에 대한 힘든 데이터 수집 프로세스가 필요하며 네트워크 교육의 확장성을 크게 제한한다.

> The few works [14], [15] that attempt to achieve unpaired virtual try-on train an unsupervised try-on network and then exploit extensive online optimization procedures to obtain fine-grained details of the original garments, harming the inference efficiency. 
>> 짝을 이루지 않은 가상 트라이온(unpaired virtual try-on)을 달성하려는 몇 안 되는 연구[14], [15]는 감독되지 않은 트라이온 네트워크를 훈련한 다음 광범위한 온라인 최적화 절차를 활용하여 원래 의류의 세분화된 세부 정보를 얻음으로써 추론 효율성에 해를 끼친다.

> Furthermore, none of the existing unpaired try-on methods consider the problem of coupled style and spatial garment information directly, which is crucial to obtain accurate garment transfer results in the unpaired and unsupervised virtual try-on scenario.
>> 또한 기존의 짝을 이루지 않은 트라이온 방법 중 어떤 것도 짝을 이루지 않은 가상 트라이온 시나리오에서 정확한 의류 전송 결과를 얻는 데 중요한 커플링 스타일 및 공간 의류 정보의 문제를 직접 고려하지 않는다.

> The few works [14], [15] that attempt to achieve unpaired virtual try-on train an unsupervised try-on network and then exploit extensive online optimization procedures to obtain fine-grained details of the original garments, harming the inference efficiency. 
>> 짝을 이루지 않은 가상 트라이온을 달성하려는 몇 안 되는 연구[14], [15]는 감독되지 않은 트라이온 네트워크를 훈련한 다음 광범위한 온라인 최적화 절차를 활용하여 원래 의류의 세분화된 세부 정보를 얻음으로써 추론 효율성에 해를 끼친다.

> Furthermore, none of the existing unpaired try-on methods consider the problem of coupled style and spatial garment information directly, which is crucial to obtain accurate garment transfer results in the unpaired and unsupervised virtual try-on scenario.
>> 또한 기존의 짝을 이루지 않은 트라이온 방법 중 어떤 것도 짝을 이루지 않은 가상 트라이온 시나리오에서 정확한 의류 전송 결과를 얻는 데 중요한 커플링 스타일 및 공간 의류 정보의 문제를 직접 고려하지 않는다.

> On the other hand, the choice of garment representation is crucial in devising a controllable virtual try-on network.
>> 반면, 의류 표현의 선택은 제어 가능한 가상 트라이온 네트워크를 고안하는 데 중요하다.

> A flexible garment representation can empower the network to accomplish fine-grained garment editing. 
>> 유연한 의복 표현은 네트워크가 세분화된 의복 편집을 수행할 수 있도록 지원할 수 있다.

> Nevertheless, most existing methods [2]–[4], [6], [7] take intact garments as inputs and only focus on garment transfer, failing to conduct controllable garment editing. 
>> 그럼에도 불구하고, 대부분의 기존 방법 [2]–[4], [6], [7]은 손상되지 않은 의복을 입력으로 취하고 의복 전달에만 초점을 맞추고 제어 가능한 의복 편집을 수행하지 못했다.

> Some methods [9], [10] introduce human parsing into the try-on network and support fundamental editing like shape/texture transfer between two garments.
>> 일부 방법 [9], [10]은 트라이온 네트워크에 인간 파싱을 도입하고 두 의복 사이의 모양/ 질감 전송과 같은 기본 편집을 지원한다.

> However, since the human parsing is defined on an object level, it is unable to distinguish different regions (e.g., upper/lower sleeve region, upper/lower pants region, torso region, etc.) within the garment, which makes it challenging to conduct more fine-grained garment editing like changing the shape of a particular local region. 
>> 그러나, 인간 파싱은 객체 수준에서 정의되기 때문에, 의복 내에서 서로 다른 영역(예: 상/하부 소매 영역, 상/하부 바지 영역, 몸통 영역 등)을 구별할 수 없으므로, 특정 국소 영역의 모양을 변경하는 것과 같은 보다 세분화된 의복 편집을 수행하기가 어렵다.

> Some advanced methods [8], [11], [12] exploit the pre-defined body segmentation from the 3D model [23], [24] to divide the garment into body-related patches, which enables the network to conduct editing at a patch-level.
>> 일부 고급 방법 [8], [11], [12]는 3D 모델 [23], [24]의 사전 정의된 신체 분할을 활용하여 의복을 신체 관련 패치로 분할하여 네트워크가 패치 수준에서 편집을 수행할 수 있도록 한다.

> However, since the 3D model [23], [24] can only represent the human body without clothing, garment patches derived from the 3D model can not guarantee the completeness of the original
garment, especially for loose garments.
>> 그러나, 3D 모델[23], [24]은 의류 없이 인체만을 나타낼 수 있기 때문에, 3D 모델에서 파생된 의류 패치는 특히 헐렁한 의류의 경우 원래 의류의 완전성을 보장할 수 없다.

> Therefore, in order to fulfill controllable editing during the virtual try-on process, a feasible garment representation is required, which not only separates the garment into fine-grained patches but also maintains the garment’s completeness.
>> 따라서 가상 트라이온 프로세스 중에 제어 가능한 편집을 수행하기 위해서는 의복을 미세한 패치로 분리할 뿐만 아니라 의복의 완전성을 유지하는 실현 가능한 의복 표현이 필요하다.

> In this paper, to tackle the essential challenges mentioned above and resolve the deficiency of prior approaches, we propose a novel PAtch-routed SpaTially-Adaptive GAN++ (PASTA-GAN++), a versatile solution to the high-resolution unpaired virtual try-on task. 
>> 본 논문에서는 위에서 언급한 필수 과제를 해결하고 이전 접근 방식의 부족을 해결하기 위해 고해상도 짝을 이루지 않은 가상 트라이온 작업에 대한 다목적 솔루션인 새로운 패치 라우팅 SpaTially-Adaptive GAN++(PASTA-GAN++)를 제안한다.

> Our PASTA-GAN++ can precisely synthesize garment shape and texture by introducing a patchrouted disentanglement module that decouples the garment style and spatial features, a patch-guided parsing synthesis block to generate correct garment shapes complying with specific body poses, as well as a novel spatially-adaptive residual module to mitigate the problem of feature misalignment. 
>> 우리의 PASTA-GAN++는 의복 스타일과 공간적 특징을 분리하는 패치 라우팅 분리 모듈, 특정 신체 자세를 준수하는 올바른 의복 모양을 생성하는 패치 안내 구문 분석 합성 블록, 완화하기 위한 새로운 공간 적응형 잔류 모듈을 도입하여 의복 모양과 질감을 정밀하게 합성할 수 있다 형상 정렬 불량 문제.

> Besides, due to the well-designed garment patches, our PASTA-GAN++ can handle arbitrary garment categories (e.g., shirt, vest, sling, pants, skirts, dresses, etc.) through a single pre-trained model and further supports garment editing (e.g. dressing order controlling, shape/texture transfer, local shape editing, etc.) during the try-on procedure (see Fig. 1).
>> 또한 잘 설계된 의류 패치로 인해 당사의 파스타-간++는 사전 훈련된 단일 모델을 통해 임의의 의류 범주(예: 셔츠, 조끼, 슬링, 바지, 스커트, 드레스 등)를 처리할 수 있으며 트라이온 절차(예: 드레싱 순서 제어, 모양/ 텍스처 전송, 로컬 모양 편집 등) 동안 의류 편집을 추가로 지원한다그림 1)을 참조하십시오.

> The innovation of our PASTA-GAN++ includes four aspects: 
>> 우리의 PASTA-GAN++의 혁신은 다음과 같은 네 가지 측면을 포함합니다:

> First, by separating the garments into normalized patches with the inherent spatial information largely reduced, the patch-routed disentanglement module encourages the style encoder to learn spatial-agnostic garment features.
>> 첫째, 고유한 공간 정보가 크게 줄어든 상태에서 의복을 정규화된 패치로 분리함으로써 패치 라우팅 분리 모듈은 스타일 인코더가 공간에 구애받지 않는 의복 특징을 학습하도록 장려한다.

> These features enable the synthesis network to generate images with accurate garment style regardless of varying spatial garment information. 
>> 이러한 기능을 통해 합성 네트워크는 다양한 공간 의류 정보에 관계없이 정확한 의류 스타일로 이미지를 생성할 수 있습니다.

> Second, the well-designed garment patches provide a fine-grained and unified representation for garments from various categories, which is leveraged by the patch-guided parsing synthesis block to generate correct target shape for arbitrary garment category within a single pre-trained model, and enables the model to conduct finegrained garment editing. 
>> 둘째, 잘 디자인된 의류 패치는 다양한 범주의 의류에 대해 세분화되고 통합된 표현을 제공하는데, 이는 패치 안내 구문 분석 합성 블록에 의해 활용되어 사전 훈련된 단일 모델 내에서 임의의 의류 범주에 대한 올바른 대상 모양을 생성하고 모델이 미세하게 바운 의류 편집을 수행할 수 있게 한다.

> Third, given the target human pose, the normalized patches can be easily reconstructed to the warped garment complying with the target shape, without requiring a warping network or a 3D human model. 
>> 셋째, 대상 인간 자세를 고려할 때, 정규화된 패치는 워핑 네트워크나 3D 인간 모델을 필요로 하지 않고 대상 모양을 따라 워핑된 의복으로 쉽게 재구성할 수 있다.

> Finally, the spatially-adaptive residual module extracts the warped garment feature and adaptively inpaints the region that is misaligned with the target garment shape. 
>> 마지막으로, 공간 적응형 잔류 모듈은 뒤틀린 의복 특징을 추출하고 대상 의복 형태와 잘못 정렬된 영역을 적응적으로 색칠합니다.

> Thereafter, the inpainted warped garment features are embedded into the intermediate layer of the synthesis network, guiding the network to generate try-on results with realistic garment texture.
>> 그 후, 페인트로 칠해진 뒤틀린 의복 특징이 합성 네트워크의 중간 계층에 내장되어 네트워크가 사실적인 의복 질감으로 시험 결과를 생성하도록 안내한다.

> To explore the proposed high-resolution unpaired virtual try-on algorithm, we collect a large number of high-quality images and combine them with a subset of the existing virtual try-on benchmarks [19], [20] to construct a scalable UnPaired virtual Try-on (UPT) dataset, which contains more than 100k high-resolution front-view fashion model images wearing a large variety of garments, e.g., long/short sleeve shirt, sling, vest, sling, pants, shorts, skirts, dresses, etc. 
>> 제안된 고해상도 짝을 이루지 않은 가상 트라이온 알고리듬을 탐색하기 위해 다수의 고품질 이미지를 수집하고 기존 가상 트라이온 벤치마크의 하위 집합[19], [20]과 결합하여 100,000개 이상의 고해상도 프론트 뷰 패션 모델 이미지를 포함하는 확장 가능한 UnPaired 가상 트라이온(UPT) 데이터 세트를 구성한다긴/짧은 소매 셔츠, 슬링, 조끼, 슬링, 바지, 반바지, 치마, 드레스 등 다양한 종류의 의류를 착용하는 것.

> Extensive experiment results on the UPT dataset demonstrate that our unsupervised PASTA-GAN++ outperforms the previous virtual try-on approaches and can obtain impressive results for controllable garment editing.
>> UPT 데이터 세트에 대한 광범위한 실험 결과는 감독되지 않은 PASTA-GAN++가 이전의 가상 트라이온 접근 방식을 능가하고 제어 가능한 의복 편집에 대한 인상적인 결과를 얻을 수 있음을 보여준다.

## 2. Related work

### 2.1 Human-centric Image Synthesis

> 