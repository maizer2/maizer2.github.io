---
layout: post 
title: "(GAN)InsetGAN for Full-Body Image Generation Translation"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review, 1.2.2.5. GAN]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

### [$$\mathbf{InsetGAN\;for\;Full-Body\;Image\;Generation}$$](https://arxiv.org/pdf/2203.07293.pdf)

##### $$\mathbf{Anna\;Fruhstuck,\;Krishna\;Kumar\;Singh,\;Eli Shechtman}$$

##### $$\mathbf{Niloy\;J.\;Mitra,\;Peter\;Wonka,\;Jingwan\;Lu}$$

##### $$\mathbf{KAUST,\;Adobe\;Research,\;University\;College\;London}$$

![Figure 1]()

> Figure 1. **InsetGAN applications.** Our full-body human generator is able to generate reasonable bodies at state-of-the-art resolution (1024×1024px) (a). However, some artifacts appear in the synthesized results, most visibly in extremities and faces. We make use of a second, specialized generator to seamlessly improve the face region (b). We can also use a given face as an input for unconditional generation of bodies (c). Furthermore, we can select both specific faces and bodies and compose them in a seamlessly merged output (d).
>> 그림 1. **InsetGAN 응용 프로그램.** 우리의 전신 인간 생성기는 최첨단 해상도(1024×1024px)로 합리적인 신체를 생성할 수 있다(a). 그러나 일부 인공물은 합성 결과에서 가장 눈에 띄게 사지와 얼굴에서 나타난다. 얼굴 영역을 원활하게 개선하기 위해 두 번째 전문 생성기를 사용한다(b). 또한 주어진 얼굴을 무조건적인 신체 생성을 위한 입력으로 사용할 수 있다(c). 또한 특정 얼굴과 몸을 모두 선택하여 매끄럽게 병합된 출력(d)으로 구성할 수 있다.

### $\mathbf{Abstract}$

> While GANs can produce photo-realistic images in ideal conditions for certain domains, the generation of full-body human images remains difficult due to the diversity of identities, hairstyles, clothing, and the variance in pose. Instead of modeling this complex domain with a single GAN, we propose a novel method to combine multiple pretrained GANs, where one GAN generates a global canvas (e.g., human body) and a set of specialized GANs, or insets, focus on different parts (e.g., faces, shoes) that can be seamlessly inserted onto the global canvas. We model the problem as jointly exploring the respective latent spaces such that the generated images can be combined, by inserting the parts from the specialized generators onto the global canvas, without introducing seams. We demonstrate the setup by combining a full body GAN with a dedicated high-quality face GAN to produce plausible-looking humans. We evaluate our results with quantitative metrics and user studies.
>> GAN은 특정 도메인에 이상적인 조건에서 사진 사실적인 이미지를 생성할 수 있지만, 정체성, 헤어스타일, 의상 및 포즈의 다양성으로 인해 전신 인간 이미지 생성은 여전히 어렵다. 단일 GAN으로 이 복잡한 도메인을 모델링하는 대신, 우리는 사전 훈련된 여러 GAN을 결합하는 새로운 방법을 제안한다. 여기서 한 GAN은 글로벌 캔버스(예: 인체)와 전문화된 GAN 세트를 생성하고 글로벌 캔버스에 원활하게 삽입할 수 있는 다른 부분(예: 얼굴, 신발)에 초점을 맞춘다. 우리는 문제를 솔기를 도입하지 않고 전문 발전기의 부품을 글로벌 캔버스에 삽입하여 생성된 이미지가 결합될 수 있도록 각각의 잠재 공간을 공동으로 탐색하는 것으로 모델링한다. 우리는 그럴듯해 보이는 인간을 만들기 위해 전신 GAN과 전용 고품질 얼굴 GAN을 결합하여 설정을 보여준다. 정량적 측정 기준과 사용자 연구를 통해 결과를 평가한다.

### $\mathbf{1.\;Introduction}$

> Generative adversarial networks (GANs) have emerged as a very successful image generation paradigm. For example, "StyleGAN" [14] is now the method of choice for creating near photorealistic images for multiple classes (e.g., human faces, cars, landscapes). However, for classes that exhibit complex variations, creating very high quality results becomes harder. For example, full-body human generation still remains an open challenge, given the high variability of human pose, shape, and appearance.
>> 생성적 적대 네트워크(GAN)는 매우 성공적인 이미지 생성 패러다임으로 떠올랐다. 예를 들어, 스타일GAN[14]은 이제 여러 클래스(예: 사람 얼굴, 자동차, 풍경)에 대해 거의 사실적인 이미지를 만드는 선택 방법이다. 그러나 복잡한 변동을 보이는 클래스의 경우 매우 높은 품질의 결과를 만드는 것이 어려워집니다. 예를 들어, 인간의 자세, 모양 및 외형의 높은 가변성을 고려할 때, 전신 인간 세대는 여전히 열린 과제로 남아 있다.

> How can we generate results at both high resolution and high quality? One approach is to break the target image into tiles and train a GAN to sequentially produce them [7]. Such methods, however, are unsuited for cases where the coupling between the (object) parts are nonlocal and/or cannot easily be statistically modeled. An alternate approach is to aim for collecting very high resolution images and train a single GAN, at full resolution. However, this makes the data collection and training tasks very expensive, and variations in object configuration/poses cause further challenges. To the best of our knowledge, neither such a high resolution dataset, nor a corresponding high resolution GAN architecture has been published.
>> 고해상도 및 고품질 모두에서 결과를 생성할 수 있는 방법은 무엇입니까? 한 가지 접근 방식은 대상 이미지를 타일로 분할하고 순차적으로 생성하도록 GAN을 훈련시키는 것이다[7]. 그러나 이러한 방법은 (물체) 부품 간의 결합이 국부적이지 않거나 통계적으로 쉽게 모델링할 수 없는 경우에는 적합하지 않다. 다른 접근 방식은 고해상도 이미지를 수집하고 단일 GAN을 완전한 해상도로 훈련하는 것을 목표로 한다. 그러나 이는 데이터 수집 및 교육 작업을 매우 비싸게 만들고, 개체 구성/포즈의 다양성은 추가적인 문제를 야기한다. 우리가 아는 한, 그러한 고해상도 데이터 세트나 해당 고해상도 GAN 아키텍처는 출판되지 않았다.

> We propose InsetGAN towards solving the above problems. Specifically, we propose to combine a generator to provide the global context in the form of a canvas, and a set of specialized part generators that provide details for different regions of interest. The specialized results are then pasted, as insets, on to the canvas to produce a final generation. Such an approach has multiple advantages: (I) the canvas GAN can be trained on medium quality data, where the object parts are not necessarily aligned. Although this results in the individual parts in the canvas being somewhat blurry (e.g., fuzzy/distorted faces in case of human bodies), this is sufficient to provide global coordination for later specialized parts to be inserted; (II) the specialized parts can be trained on part-specific data, where consistent alignment can be more easily achieved; and (III) different canvas/part GANs can be trained at different resolutions, thus lowering the data (quality) requirements. CollageGAN [20] has explored a similar idea in a conditional setting. Given a semantic map which provides useful shape and alignment hints, they create a collage using an ensemble of outputs from class-specific GANs [20]. In contrast, our work focuses on the unconditional setting, which is more challenging since our multiple generators need to collaborate with one another to generate a coherent shape and appearance together without access to a semantic map for hints.
>> 위의 문제를 해결하기 위해 InsetGAN을 제안한다. 구체적으로, 우리는 글로벌 컨텍스트를 캔버스 형태로 제공하기 위한 생성기와 다양한 관심 영역에 대한 세부 정보를 제공하는 전문 부품 생성기 세트를 결합할 것을 제안한다. 그런 다음 전문화된 결과를 캔버스에 삽입하여 붙여넣어 최종 세대를 만듭니다. 이러한 접근 방식은 여러 가지 장점이 있다. (I) 객체 부품이 반드시 정렬되지 않는 중간 품질의 데이터에 대해 캔버스 GAN을 훈련할 수 있다. 이로 인해 캔버스의 개별 부품(예: 인체의 경우 퍼지/왜곡된 얼굴)이 다소 흐릿해지지만, 이는 나중에 특수 부품을 삽입할 수 있도록 글로벌 조정을 제공하기에 충분하다. (II) 특수 부품을 일관된 정렬을 보다 쉽게 달성할 수 있는 부품 특정 데이터에 대해 훈련할 수 있다.d; 및 (III) 서로 다른 캔버스/부품 GAN을 서로 다른 해상도로 훈련할 수 있으므로 데이터(품질) 요구 사항을 낮출 수 있다. 콜라주GAN[20]은 조건부 환경에서 유사한 아이디어를 탐구했다. 유용한 모양과 정렬 힌트를 제공하는 의미 맵이 주어지면, 클래스별 GAN의 출력 앙상블을 사용하여 콜라주를 생성한다[20]. 대조적으로, 우리의 작업은 무조건적인 설정에 중점을 두는데, 이는 힌트를 위한 의미 맵에 액세스하지 않고 우리의 다중 생성기가 일관된 모양과 모양을 함께 생성하기 위해 서로 협력해야 하기 때문에 더 어렵다.

![Figure 2]()

> Figure 2. InsetGAN Pipeline. Given two latents $w_{A}$ and $w_{B}$, along with pretrained generators $G_{A}$ and GB, that generate two images $I_{A}:=G_{A}(w_{A})$ and $I_{B}:=G_{B}(w_{B})$, respectively, we design a pipeline that can optimize either only $w_{A}$ (a), or iteratively optimize both $w_{A}$ and $w_{B}$ (b) in order to achieve a seamless output composition of face and body. We use a set of losses Lcoarse and Lborder to describe the conditions we want to minimize during optimization. On the right, we show that given an input body, mere copy and pasting of a target face yields boundary artifacts. We show an application of one-way optimization (top right) and two-way optimization (bottom right) to create a seamlessly merged result. Note that when the algorithm can optimize in both inset-face and canvas-body generator spaces, it produces more natural results at the seam boundary – notice how the hair and skin tone blend from the head to the body region. The joint optimization is challenging as the bounding box B(I_{A}) is conditioned on the variable w_{A}.
>> 그림 2. GAN 파이프라인을 삽입합니다. 사전 훈련된 생성기 $G_{A}$ 및 GB와 함께 두 개의 이미지 I_{A}:=G_{A}(w_{A}) 및 I_{B}:=G_{B}(w_{B})를 생성하는 두 개의 잠재 $w_{A}$ 및 w_{B}를 고려할 때, 우리는 얼굴과 몸의 원활한 출력 구성을 달성하기 위해 w_{A}와 w_{B}(b)만 최적화하거나 반복적으로 최적화할 수 있는 파이프라인을 설계한다. 우리는 최적화 중에 최소화하고자 하는 조건을 설명하기 위해 Lcorse와 Lborder의 집합을 사용한다. 오른쪽에서는 입력 본문이 주어지면 대상 면의 복사 및 붙여넣기만으로 경계 아티팩트가 생성된다는 것을 보여준다. 우리는 매끄럽게 병합된 결과를 만들기 위해 단방향 최적화(오른쪽 위)와 양방향 최적화(오른쪽 아래)의 응용 프로그램을 보여준다. 알고리즘이 인서트 페이스와 캔버스 바디 발생기 공간 모두에서 최적화할 수 있는 경우, 솔기 경계에서 더 자연스러운 결과를 생성한다는 점에 유의하십시오. 즉, 머리부터 신체 부위까지 머리카락과 피부 톤이 어떻게 혼합되는지 주목하십시오. 경계 상자 B(IA)가 변수 w_{A}에 대해 조건화되기 때문에 공동 최적화는 어렵다.

> The remaining problem is how to coordinate the canvas and the part GANs, such that adding the insets to the canvas does not reveal seam artifacts at the inset boundaries. This aspect is particularly challenging when boundary conditions are nontrivial and the inset boundaries themselves are unknown. For example, a face, when added to the body, should have consistent skin tone, clothing boundaries, and hair flow. We solve the problem by jointly seeking latent codes in (pretrained) canvas and part GANs such that the final image, formed by inserting the part insets on the canvas, does not exhibit any seams. In this paper, we investigate this problem in the context of human body generation, where the human faces are created by a face-specific GAN.
>> 남은 문제는 캔버스와 부품 GAN을 조정하는 방법인데, 이렇게 하면 캔버스에 인셋을 추가하면 인셋 경계에서 심 아티팩트가 드러나지 않는다. 이러한 측면은 경계 조건이 중요하지 않고 삽입 경계 자체를 알 수 없는 경우 특히 어렵다. 예를 들어, 얼굴은 신체에 추가되었을 때, 일관된 피부색, 옷의 경계, 그리고 머리카락의 흐름을 가져야 한다. 우리는 (사전 훈련된) 캔버스 및 부품 GAN에서 잠재 코드를 공동으로 검색하여 캔버스에 부품을 삽입하여 형성된 최종 이미지가 솔기를 나타내지 않도록 문제를 해결한다. 본 논문에서는 얼굴별 GAN에 의해 인간의 얼굴이 생성되는 인체 생성의 맥락에서 이 문제를 조사한다.

> We evaluate InsetGAN on a custom dataset, compare with alternative approaches, and evaluate the quality of the results with quantitative metrics and user studies. Fig. 1 shows human body generation applications highlighting both seamless results, across face insets, as well as having diversity of solutions across face insertion boundaries.
>> 우리는 사용자 지정 데이터 세트에서 InsetGAN을 평가하고, 대안적 접근 방식과 비교하며, 정량적 메트릭과 사용자 연구를 통해 결과의 품질을 평가한다. 그림 1은 얼굴 삽입 경계에 걸친 솔루션의 다양성뿐만 아니라 얼굴 삽입 전체에 걸쳐 완벽한 결과를 강조하는 인체 생성 응용 프로그램을 보여준다.

> **Contributions. (1)** We propose a multi-GAN optimization framework that jointly optimizes the latent codes of two or more collaborative generators such that the overall composed result is coherent and free of boundary artifacts when the generated parts are inserted as insets into the generated canvas. (2) We demonstrate our framework on the highly challenging full-body human generation task and propose the first viable pipeline to generate plausible-looking humans unconditionally at 1024×1024px resolution.
>> **기여. (1)** 우리는 생성된 부품을 생성된 캔버스에 삽입할 때 전체 구성 결과가 일관되고 경계 아티팩트가 없도록 두 개 이상의 협업 생성기의 잠재 코드를 공동으로 최적화하는 다중 GAN 최적화 프레임워크를 제안한다. (2) 우리는 높은 값에 대한 프레임워크를 보여준다. (2)우리는 매우 어려운 전신 인간 생성 작업에 대한 우리의 프레임워크를 시연하고 1024×1024px 해상도로 그럴듯해 보이는 인간을 무조건 생성할 수 있는 최초의 실행 가능한 파이프라인을 제안한다.



### $\mathbf{2.\;Related Work}$

> Unconditional Image Generation via Generative Adversarial Networks (GANs) [8] has shown a lot of promise in recent years. In this context, the "StyleGAN" architecture was developed over a sequence of papers [11–14] and is widely considered the state of the art for synthesizing individual object classes. For class-conditional image generation on the ImageNet dataset, "BigGAN" [5] is often the architecture of choice. In our work, we are building on "StyleGAN2-ADA", since this architecture yields better FID [9] and Precision&Recall [17] scores on our domain when compared to "StyleGAN3". In addition, generating complete human body images using "StyleGAN2" is a baseline we would like to improve upon in our work.
>> 생성적 적대 네트워크(GAN)를 통한 무조건적 이미지 생성[8]은 최근 몇 년 동안 많은 가능성을 보여주었다. 이러한 맥락에서, "StyleGAN" 아키텍처는 일련의 논문에 걸쳐 개발되었으며 [11-14] 개별 객체 클래스를 합성하는 최첨단 아키텍처로 널리 간주된다. ImageNet 데이터 세트에 대한 클래스 조건부 이미지 생성의 경우 "BigGAN"[5]이 종종 선택되는 아키텍처이다. 우리의 연구에서, 우리는 "StyleGAN2-ADA"를 기반으로 하고 있는데, 이 아키텍처가 "StyleGAN3"에 비해 도메인에서 더 나은 FID[9] 및 Precision&Recall[17] 점수를 산출하기 때문이다. 또한 "StyleGAN2"를 사용하여 완전한 인체 이미지를 생성하는 것은 우리의 작업에서 개선하고자 하는 기준이다.

![Figure 3]()

> Figure 3. Unconditional generation results. Examples are created using our adaptive truncation approach (described in the supplementary material) and are cropped horizontally. At first glance, the results look realistic, but face regions show noticeable artifacts (please zoom).
>> 그림 3. 무조건 생성 결과. 예제는 적응형 잘라내기 접근 방식(보조 자료에 설명되어 있음)을 사용하여 생성되고 수평으로 잘린다. 언뜻 보기에는 결과가 사실적으로 보이지만 얼굴 영역에는 눈에 띄는 아티팩트가 표시됩니다(확대/축소).

> **Image Outpainting** refers to image completion problems where the missing pixels are not surrounded by available pixels. Recent papers build on the ideas to use generative adversarial networks [29] and the explicit modeling of structure [19, 34]. Though these two papers specialize in human bodies, we find that the GAN architecture CoModGAN [33] has even more impressive results for image outpainting (see the comparison in Section 5). 
>> **Image Outpainting**은 누락된 픽셀이 사용 가능한 픽셀로 둘러싸여 있지 않은 이미지 완성 문제를 말합니다. 최근 논문은 생성적 적대 네트워크[29]와 구조의 명시적 모델링[19, 34]을 사용하는 아이디어를 기반으로 한다. 이 두 논문은 인체를 전문으로 하지만, 우리는 GAN 아키텍처 CoModGAN[33]이 이미지 아웃페인팅에 대해 훨씬 더 인상적인 결과를 가지고 있다는 것을 발견했다(섹션 5의 비교 참조).

> **Conditional Generation of Full-Body Humans** has two possible advantages. First, the conditional generation enables more control. Second, conditional generation can help in controlling the variability and improve the visual quality. In the context of humans, a natural idea is to condition the generation on the human pose [4, 15, 16, 22, 23, 26, 28] or segmentation information [10].
>> **조건부 전신 인간 생성**에는 두 가지 장점이 있습니다. 첫째, 조건부 생성은 더 많은 제어를 가능하게 한다. 둘째, 조건부 생성은 가변성을 제어하고 시각적 품질을 향상시키는 데 도움이 될 수 있다. 인간의 맥락에서 자연스러운 아이디어는 인간 자세[4, 15, 16, 22, 23, 26, 28] 또는 분할 정보[10]에 따라 세대를 조건화하는 것이다.

> As many conditional architectures are not able to handle the same high resolution (1024×1024px) of unconditional StyleGAN, an alternative to developing new architectures is conditional embedding into an unconditional generator’s latent space. Two approaches used in this context are "StyleGAN" embedding using optimization [1, 2] or "StyleGAN" embedding using an encoder architecture [3, 25, 30]. Our work also makes use of embedding algorithms.
>> 많은 조건부 아키텍처가 무조건적인 StyleGAN의 동일한 고해상도(1024×1024px)를 처리할 수 없기 때문에, 새로운 아키텍처를 개발하는 대안은 무조건적인 생성기의 잠재 공간에 조건부 임베딩이다. 이러한 맥락에서 사용되는 두 가지 접근 방식은 최적화[1, 2]를 사용하는 "StyleGAN" 임베딩 또는 인코더 아키텍처[3, 25, 30]를 사용하는 "StyleGAN" 임베딩이다. 또한 우리의 작업은 임베딩 알고리듬을 사용한다.

### $\mathbf{3.\;Methodology}$

> We propose a method for the unconditional generation of full-body human images using one or more independent pretrained unconditional generator networks. Depending on the desired application and output configuration, we describe different ways to coordinate the multiple generators. 
>> 하나 이상의 독립적인 사전 훈련된 무조건 생성기 네트워크를 사용하여 전신 인간 이미지의 무조건 생성을 위한 방법을 제안한다. 원하는 응용 프로그램 및 출력 구성에 따라 여러 생성기를 조정하는 다양한 방법을 설명합니다.

#### $\mathbf{3.1.\;Full-Body\;GAN}$

> The naive approach to generate a full-body human image is to use a single generator trained on tens of thousands of example humans (see Section 4 about the dataset). We adopt the state-of-the-art "StyleGAN2" architecture proposed by Karras et al. [11]. Most previous full-body generation or editing work [4, 18, 20, 35] generate images at 256×256px or 512×512px resolution. We made the first attempt to unconditionally generate full-body humans at 1024×1024px resolution. Due to the complex nature of our target domain, the results generated by a single GAN sometimes exhibit artifacts such as weirdly-shaped body parts and nonphotorealistic appearance. These artifacts are most visible in faces and extremities, as shown in Fig. 1(a). Due to the vast diversity of human poses and appearances and the associated alignment difficulty, hands and feet appear in many possible locations in the training images, making them harder for a single generator to learn. Faces are especially hard since we humans are ultra-sensitive to artifacts in these areas. They therefore deserve dedicated networks and special treatment. Fig. 3 shows a variety of unconditional generation results. Our results exhibit correct human body proportions, consistent skin tones across face and body, interesting garment variations and plausible-looking accessories (e.g. handbags and sunglasses) whereas artifacts can be present when viewed in detail.
>> 전신 인간 이미지를 생성하는 순진한 접근 방식은 수만 명의 예시 인간에 대해 훈련된 단일 생성기를 사용하는 것이다(데이터 세트에 대한 섹션 4 참조). 우리는 Karas 등이 제안한 최첨단 "StyleGAN2" 아키텍처를 채택한다. [11. 대부분의 이전 전신 생성 또는 편집 작업[4, 18, 20, 35]은 256x256px 또는 512x512px 해상도로 이미지를 생성합니다. 우리는 1024×1024px 해상도에서 무조건 전신 인간을 생성하려는 첫 번째 시도를 했다. 대상 도메인의 복잡한 특성으로 인해 단일 GAN에 의해 생성된 결과는 때때로 이상한 모양의 신체 부위 및 비사실적 외관과 같은 아티팩트를 보여준다. 이러한 아티팩트는 그림 1(a)과 같이 얼굴과 사지에서 가장 잘 보입니다. 인간의 자세와 외관의 엄청난 다양성과 관련된 정렬 어려움으로 인해 손과 발이 훈련 이미지의 많은 가능한 위치에 나타나 단일 생성자가 배우기가 더 어렵다. 우리 인간은 특히 이 지역의 공예품에 매우 민감하기 때문에 얼굴은 특히 힘들다. 그러므로 그들은 전용 네트워크와 특별 대우를 받을 자격이 있다. 그림 3은 다양한 무조건적인 생성 결과를 보여준다. 우리의 결과는 정확한 인체 비율, 얼굴과 신체에 걸쳐 일관된 피부 색조, 흥미로운 의류 변형 및 그럴듯해 보이는 액세서리(예: 핸드백과 선글라스)를 보여주지만, 아티팩트는 자세히 볼 때 존재할 수 있다.

#### $\mathbf{3.2.\;Multi-GAN\;Optimization}$

> To improve the problematic regions generated by the full-body GAN, we use other generators trained on images of specific body regions to generate pixels to be pasted, as insets, into the full-body GAN result. The base full-body GAN and the dedicated body part GANs can be trained using the same or different datasets. In either case, the additional network capacity contained in the multiple GANs can better model the complex appearance and variability of the human bodies.
>> 전신 GAN에서 생성된 문제 영역을 개선하기 위해 특정 신체 영역의 이미지에 대해 훈련된 다른 생성기를 사용하여 전신 GAN 결과에 삽입하여 붙여넣을 픽셀을 생성한다. 기본 전신 GAN 및 전용 신체 부위 GAN은 동일하거나 다른 데이터 세트를 사용하여 훈련될 수 있다. 두 경우 모두 여러 GAN에 포함된 추가 네트워크 용량은 인체의 복잡한 외관과 가변성을 더 잘 모델링할 수 있다.

> As a proof of concept, we show that a face GAN trained with the face regions cropped from our full-body training images can be used to improve the appearance of the body GAN results. Alternatively, we can also leverage a face generator trained on other datasets such as FFHQ [14] for face enhancement as well. Similarly, specialized hands or feet generators can also be used in our framework to improve other regions of the body. We show that we can also use multiple part generators together in a multi-optimization process, as depicted in Fig. 4.
>> 개념 증명으로, 전신 훈련 이미지에서 잘라낸 얼굴 영역으로 훈련된 얼굴 GAN을 사용하여 신체 GAN 결과의 외관을 개선할 수 있음을 보여준다. 또는 FFHQ[14]와 같은 다른 데이터 세트에 대해 훈련된 얼굴 생성기를 활용하여 얼굴 개선도 할 수 있다. 마찬가지로, 특수화된 손 또는 발 생성기도 우리 프레임워크에서 신체의 다른 부위를 개선하는 데 사용될 수 있다. 우리는 또한 그림 4와 같이 다중 최적화 프로세스에서 여러 개의 부품 생성기를 함께 사용할 수 있음을 보여준다.

> The main challenge is how to coordinate multiple unconditional GANs to produce pixels that are coherent with one another. In our application, we have alathat generates the full-body human where $I_{A}:=G_{A}(w_{A})$ and another $G_{B}$ that generates a sub-region or inset within the human body where $I_{B}:=G_{B}(w_{B})$. In order to coordinate the specialized part GAN with the global/canvas GAN, we need a bounding box detector to identify the region of $I_{A}$ that corresponds to the region our part GAN generates. We crop $I_{A}$ with the detected bounding box and denote the cropped pixels as $B(I_{A})$. The problem of inserting a separatelygenerated part $I_{B}$ into the canvas $I_{A}$ is equivalent to finding a latent code pair $(w_{A},w_{B})$ such that the respective images $I_{A}$ and $I_{B}$ can be combined without noticeable seams in the boundary regions of $B(I_{A})$ and $I_{B}$. To generate the final result, we directly replace the original pixels inside the bounding box $B(I_{A})$ with the generated pixels from $I_{B}$,

$$$$

![Figure 4]()

> Figure 4. Two Insets. These results are improved using a dedicated shoe generator trained on shoe crops from our full-body humans, and also using our face generator. All three generators (full-body canvas and two insets) are jointly optimized to produce a seamless coherent output. The circular closeups show the shoes before (top) and after (bottom) improvement (please zoom).

> where, Ω := B(GA(w_{A})) and, with slight abuse of notation, L captures the loss both along the boundary of Ω measuring seam quality and inside the region Ω measuring similarity of $I_{A}$ and $I_{B}$ inside the respective faces. The full optimization is complex as the region of interest Ω depends on w_{A}.

> Our multi-GAN optimization framework can support various human generation and editing applications. Depending on the application scenario, we optimize either $w_{A}$ or $w_{B}$ or jointly optimize both for the best results.

> Optimization Objectives. When optimizing the latent codes $w_{A}$,w_{B} or both, we consider multiple objectives: (I) the face regions generated by the face GAN and body GAN should have similar appearance at a coarse scale so that when the pixels generated by the face GAN are pasted onto the body GAN canvas, attributes match (e.g., the skin tone of the face matches that of the neck); (II) the boundary pixels around the face crops match up so that a simple copyand-paste operation does not result in visible seams; and (III) the final composed result looks realistic. To match the face appearance, we downsample the face regions and calculate a combination of L1 and perceptual loss [32] Llpips:

$$$$

> where I↓A = D64(B(I_{A})) and I↓B = D64(I_{B}) and D64 refers to downsampling the image to 64×64px resolution. 
> Figure 5. Face Refinement. Given generated humans, we use a dedicated face model trained on the same dataset to improve the quality of the face region. We jointly optimize both the face and the human latent codes so that the two generators coordinate with each other to produce coherent results. The two inset face crops show the initial face generated by the body GAN (bottom) and the final face improved by dedicated face GAN (top).

> For the boundary matching, we also apply a L1 and perceptual loss to the border pixels at full resolution:

$$\begin {split} \lB := \weight _3\Lone (\border _8(\bbox (\iA )), \border _8(\iB )) + \\ \weight _4\percep (\border _8(\bbox (\iA )), \border _8(\iB )) \end {split}$$

![Figure 5]()

> Figure 5. Face Refinement. Given generated humans, we use a dedicated face model trained on the same dataset to improve the quality of the face region. We jointly optimize both the face and the human latent codes so that the two generators coordinate with each other to produce coherent results. The two inset face crops show the initial face generated by the body GAN (bottom) and the final face improved by dedicated face GAN (top).

> where Ex(I) is the border region of I of width x pixels.

> To maintain realism during the optimization, we also add two regularization terms:

$$\vspace *{-.1in} \lR := \weight _{r1}\|\w ^* - \w_{A}vg \| + \weight _{r2} \sum _{i}\|\delta _i\|$$

> The first term prevents the optimized latent code from deviating too far from the average latent. We compute wavg by randomly sampling a large number of latents in Z space, mapping them to W space, and computing the average. The second term is to regularize the latent code in w+ latent space. During "StyleGAN2" inference, the same 512 dimensional latent code w is fed into each of the n generator layers (n is dependent on the output resolution). Many GAN inversion methods optimize in this n×512 dimensional w+ latent space [31] instead of the 512 dimensional w latent space. We follow recent work to decompose the w+ latent into a single base w∗ latent and n offset latents δi . The latent used for layer i is w +δi . We use the L2 norm as regularizer to ensure that the δis remain small. Based on our visual analysis of the results, we use larger weights for the body generator than the face generator for this regularizer. 

> We mix and match the various losses depending on the specific application at hand. 

> **Face Refinement versus Face Swap.** Given a randomly generated human body G_{A}(w_{A}), we can keep $w_{A}$ fixed and optimize for $w_{B}$ such that G_{B}(w_{B}) looks similar to B(GA(w_{A})) at a coarse scale and matches the boundary pixels at a fine scale (Fig. 2 top right). We have:

$$$$

> While this almost produces satisfactory results, boundary discontinuities show up at times. For further improvement, we can optimize both $w_{A}$ and $w_{B}$ so that both generators coordinate with each other to generate a coherent image free of blending artifacts (Fig. 2 bottom right). To keep the body appearance unchanged during the optimization of $w_{A}$,we introduce an additional loss term:

$$$$

> where Iref is the input reference body generated by $G_{A}$ that should remain unchanged during the optimization, RO defines the body region outside of the face bounding box. We also use the mean latent regularization term Lreg to prevent generating artifacts. The final objective function becomes:

$$$$

> Fig. 1(b) and Fig. 5 show face refinement results using a dedicated face model trained on faces cropped from the same data used for training the body generator. Our refinement results when using the pretrained FFHQ face model exhibit similar visual quality (see supplementary material). 

> Body Generation for an Existing Face. Given a real face or a randomly-generated face G_{B}(w_{B}), we can keep $w_{B}$ fixed and optimize for $w_{A}$ such that G_{A}(w_{A}) produces a body that looks compatible with the input face in terms of pose, skin tone, gender, hair style, etc. In practice, we find that to best maintain boundary continuity, especially when generating bodies to match faces of complex hair styles, it is often to discourage large changes in $w_{B}$, such that the face identity is mostly preserved but the boundary and background pixels can be slightly adjusted to make the optimization of $w_{A}$ easier. To preserve the face identity during the optimization, we use an additional face reconstruction loss:

$$$$

> where RI defines the interior region of the face crop and Iref denotes the referenced input face. For more precise control, face segmentation can be used instead of bounding boxes. 

> Our objective function becomes:

$$$$

> With different initialization for $w_{A}$,we can generate multiple results per face as shown in Fig. 6. Note that our model can generate diverse body appearances compatible with the input face. The generated body skin tone generally match the input face skin tone (e.g., the women of African descent in the top and bottom rows of Fig. 6). Fig. 1(c) shows another example.

![Figure 6]()

> Figure 6. Multimodal Body Generation for an Existing Face. For each face generated by the pretrained FFHQ model (middle column), we use joint optimization to generate three different bodies while maintaining the facial identities from the input faces.

> **Face Body Montage.** We can combine any real or generated face with any generated body to produce a photo montage. With a real face, we need to first encode it into the latent space of $G_{B}$ as $w_{B}$ using an off-the-shell encoder [30]. Similarly, a real body could be encoded into the latent space of GA, but due to the high variability of human bodies it is difficult to achieve a low reconstruction error. All montage results are created from synthetic bodies generated by GB. We use the following objective function:

$$$$

> Fig. 7 shows the result of combining faces (top row) generated by the pretrained FFHQ model with bodies (leftmost column) generated by our full-body generator $G_{A}$. With minor adjustments of both the face and body latent codes, we achieve composition results that are coherent and identitypreserving. While we do not have any explicit loss encouraging skin tone coherence, given faces with different skin tones, our joint optimization slightly adjusts the skin tone of the body’s neck and hand pixels to minimize appearance incoherence and boundary discrepancy in the final results. Fig. 1(d) shows two more examples. Our joint optimization is able to slightly adjust the shoulder region of the lady to extend her hair to naturally rest on her right shoulder. The rightmost column in Fig. 2 shows the improvement the joint optimization makes to the final result quality (bottom) compared to only optimizing $w_{B}$ given an input body (top).

![Figure 7]()

> Figure 7. Face Body Montage. Given target faces (top row) generated by the pretrained FFHQ model and bodies(leftmost column) generated by our full-body human generator, we apply joint latent optimization to find compatible face and human latent codes that can be combined to produce coherent full-body humans. Notice how face and skin colors get synchronized, and zoom in to observe the (lack of) seams around face insets.

> **Optimization Details.** While the difference is subtle, we observe a slightly better visual performance when using L1 over L2 losses. We apply many of our losses to downsampled versions of the images D64(B(I_{A})) and D64(I_{B}) to allow for more flexibility during optimization and to reduce the risk of overfitting to artifacts from the source image (e.g., the body GAN’s face region, which lacks realistic high-frequency details) in a strategy similar to PULSE [24].

![Figure 8]()

> Figure 8. Multimodal Face Improvement. To improve humans generated by a full-body model trained on DeepFashion, we use the pretrained FFHQ model to synthesize a variety of seamlessly merged result faces that all look compatible with the input body.

> One challenge in the joint optimization of $w_{A}$ and $w_{B}$ is that the boundary condition Ω depends on the variable w_{A}. We address this by alternately optimizing for $w_{A}$ and $w_{B}$, and reevaluating the boundary after each update of w_{A}. We stop the process when the updates converge.

> **Optimization Initialization.** The default choice of initialization for either $w_{A}$ or $w_{B}$ is their corresponding average latent vector wavg. This typically leads to reasonable results quickly. However, it is desirable to generate a variety of results for applications like finding matching bodies $I_{A}$ for an input face $I_{B}$. In this case, we start from truncated latent codes wtrunc = wrand ∗ (1 − α) + wavg ∗α. Due to the introduced randomness and the interpolation with the average latent code, we can generate diverse yet realistic results (see Fig. 6). In Fig. 8, given humans generated by our full-body model trained on DeepFashion, we use the pretrained FFHQ face model to swap in multiple better looking faces. Different initialization of $w_{B}$ yields different results. In the cases where either the face region or the body region should remain fixed during the joint optimization of both latent codes, we initialize the optimization with the latent code initially used to generate the synthetic reference image or the latent code encoded from a real image.

### $\mathbf{4.\;Dataset\;and\;Implementation}$

> We curate a proprietary dataset of 83,972 high-quality full-body human photographs at 1024×1024px resolution. These images stem from a dataset of 100,718 diverse photographs in the wild purchased from a third-party data vendor. The dataset includes hand-labeled ground-truth segmentation masks. We apply a human pose-detection network [6] on the original images and filter out those that contain extreme poses causing pose detection results to have low confidence. Fig. 9 shows some sample training images.
![Figure 9]()

> Figure 9. Full-body Human Dataset. We create a dataset from photographs of humans in the wild. The images are automatically preprocessed, aligned, and cropped to 1024×1024px resolution using the ground-truth segmentation masks and detected pose skeletons.

> Feature alignment plays an important role in high-quality image generation, as can be seen in the qualitative difference of models trained on FFHQ data vs. other face datasets. Therefore, we carefully align the humans using their pose skeletons. We define an upper body axis based on the position of the neck and hip joints. We position humans so that the upper body axis is aligned in the center of the image. As the variance in perspective and pose is very large, choosing the appropriate scale for each person within the context of their image frame is challenging. We scale the humans based on their upper-body length and then evaluate the extent of the face region as defined by the segmentation mask. If the face length is smaller (larger) than a given minimum (maximum) value, we rescale so that the face length is equal to the minimum (maximum).

> Lastly, we enlarge the backgrounds using reflection padding and heavily blur them using a Gaussian kernel of size 27 to focus the generator capacity on modeling only the foreground humans. The huge variation in background appearance in these in-the-wild photos poses extreme challenges for the GAN, especially with limited data quantity.

> We also considered completely removing the background, but did not do it for two reasons: (1) Humanlabeled segmentation masks are still imperfect around boundaries and (2) We observe that current GAN architectures do not handle large areas of uniform color well. 

> We also show our method on DeepFashion [21], which consists of 66,607 fashion photographs, including garment pieces and garments on humans. Using the same alignment strategy as above, we extract 10,145 full-body images at 1024×768 resolution. Since the backgrounds are already uniform, we do not blur them.

> **Training Details** We trained our main human body generator network at 1024×1024px resolution using the StyleGAN2-ADA architecture using all augmentation schemes proposed in the paper [11] for 28 days and 18 hours on 4 Titan V GPUs, using a batch size of 4, processing a total of 42M images. After experimenting with different R1 γ values between 0.1 and 20, we chose a value of 13. Similarly, we trained our DeepFashion human generator network at 1024×768px resolution for 9 days on 4 v100 GPUs, using a batch size of 8, processing a total of 18M images. We use a pretrained FaceNet [27] to detect and align the bounding boxes of the face regions in our generated bodies and faces. The running time of our optimization algorithm for jointly optimizing two generator latents at 1024×1024px output resolution is about 75 seconds on a Titan RTX GPU. If $G_{B}$ has a smaller resolution of 256×256px, the optimization time decreases to around 60 seconds.

### $\mathbf{5.\;Evaluation\;and\;Discussion}$

> **Quantitative Evaluation.** We follow the standard practice to calculate FID (Frechet Inception Distance) to measure how closely our generated full-body results follow the training distribution. Many previous papers including CoModGAN point out that FID statistics are noisy and do not correlate well with human perception about visual quality. We also observe that FID is more sensitive to result diversity than quality and increases significantly as we truncate the generated results, which reduces variation but is crucial for generating natural looking images with fewer artifacts. While the FID for untruncated results is 13.96, it rises to 26.67 for t=0.7 and 71.90 for t=0.4 (more truncation). We compare FID values of several alternative approaches for our face refinement application. We use two different truncation settings, t=0.7 and t=0.4 and evaluate on both the full-body images and image crops that include the refined face and the boundary pixels after copy&pasting.

![Table]()

> The differences in FID are small. This indicates that the face refinement using joint optimization does not modify the distribution learned by the unconditional generator and therefore does not decrease the result diversity. However, large differences in perceptual quality are still possible despite similar FID values as demonstrated in our user study.

> **Baseline Comparison.** To the best of our knowledge, no other prior work generates full-body humans unconditionally or inpaints/outpaints humans at 1024×1024px resolution without requiring conditioning other than reference pixels of the known regions. Previous works have attempted to generate plausible human bodies, but they require segmentation masks as input [19, 20]. The best state-of-theart method that can be repurposed for our body generation and face refinement applications is CoModGAN [33]. Fig. 10 shows that our "InsetGAN" (top right) outperforms CoModGAN (bottom right) in replacing the initial face generated by our body generator (left). We trained CoModGAN with square (with small random offsets for generalization) holes around faces using the official implementation, training data and default parameters for two weeks on four V100 GPUs. Similarly, we train CoModGAN with rectangular holes around the bodies to compare with our "InsetGAN" for the body generation task. In Fig. 11, we show the two best results of CoModGAN obtained by using several random initializations per input face. Compared to our results in Fig. 6, CoModGAN produces less realistic and diverse image completions.

> **User Study.** We performed a user study to better evaluate the perceptual quality of our method. We aggregated 500 generated humans from our full-body generator and 500 random training images. We then applied either our joint optimization method or CoModGAN to replace the face regions in the generated samples. We showed several sets of image pairs to volunteer participants on Amazon Mechanical Turk and asked them to pick “in which of the two images the person looks more plausible and real”. Per image pair, 5 votes were collected and aggregated towards the majority votes. The study shows that in 12.4% of image pairs, users prefer our unrefined results over the training images when given only 1 second to look at each image. This shows that our results get the basic human proportion and pose right, being able to confuse people about being real. In 98% of cases, users prefer our joint optimization results over the unrefined images. In contrast, only 7% of the CoModGAN samples were picked over the unrefined images, which is consistent with our observation from Fig. 10. 

> **Limitations.** Our work has multiple limitations that could benefit from improvements. First, the joint optimization approach may change details such as hair style, neckline or clothing details. In many cases the changes are minor, but in some cases changes can be larger, e.g. the woman’s hair in the middle row of Fig. 6 and the man’s collar in the top row of Fig. 7. Second, as shown in Fig. 3, our full-body GAN has other problems not improved by "InsetGAN": symmetry, e.g. noticeable in the hands and feet and on outfits (the woman with a light blue shirt), and the consistency of the fabric used for the clothing (the rightmost person). Third, the generated results have limited variations in body type and pose as discussed next in more detail. The vast majority of our generated results have a slender body type due to the training data distribution. Dataset Bias and Societal Impact. Both DeepFashion and our proprietary dataset contain biases. DeepFashion contains a limited number of unique identities. The same models appear in multiple images. An overwhelming majority of the images are female (around 9:1) and the range of age, ethnicity and body shape does not represent the real human population. As a result, models trained on it can only generate limited range of identities (mostly young white females) as shown in Fig. 8. We made our best attempt to look for a diverse dataset and purchased one from a data vendor in East Asia but noticed the over-representation of young Asian females in the images. Also, many images appear to portray slim street fashion models; as a result the vast majority of them contain slender body type and formal attires. Models trained on biased datasets tend to learn biased representations of the human body. Due to the over-representation of Asians, our results on other ethnicities contain more artifacts in the face region (see the rightmost four results in Fig. 3). We encourage future research efforts to diversify the training dataset to better serve our diverse society. The unconditional nature of our generation process combined with the limitation of some optimization losses operating at a low resolution implies that our generated human outputs might not preserve the attributes in the input image exactly. In addition, since the age distribution in our data is almost exclusively adult humans, we are not able to faithfully produce bodies for faces of children. Similar to other human domain generative models, our approach can be exploited by malicious users to produce deep fakes. However, as we have seen in the user study, even in a short second, users identify most of our generated results as fake compared to real human images. As we further improve the result qualities, we hope, and encourage other researchers, to investigate deep fake detection algorithms.

![Figure 11]()

> Figure 11. Body Generation with CoModGAN [33]. We show results generated by CoModGAN trained to fill a rectangular hole covering the body in a given image. Inputs with holes are shown in insets. We generate several results per input and show the best looking two here. Please refer to Fig. 6 for our results on the same input faces. We observe that CoModGAN creates seamless content, but worse visual quality compared to ours.

### $\mathbf{6.\;Conclusion}$

We presented "InsetGAN", the first viable framework to generate plausible-looking human images unconditionally at 1024×1024px resolution. The main technical contribution of "InsetGAN" is to introduce a multi-GAN optimization framework that jointly optimizes the latent codes of two or more collaborative generators. In future work, we propose to extend the multi-generator idea to 3D shape representations, such as 3D GANs or auto-regressive models based on transformers. We also plan to demonstrate the "InsetGAN" framework on other image domains and investigate coordinated latent editing in the multi-GAN setup.


---

### $\mathbf{References}$

<a href="#footnote_1_2" name="footnote_1_1">[1]</a> Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2StyleGAN: How to embed images into the "StyleGAN" latent space? In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), pages 4432–4441, 2019. 3

<a href="#footnote_2_2" name="footnote_2_1">[2]</a> Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2StyleGAN++: How to edit the embedded images? In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), pages 8296–8305, 2020. 3

<a href="#footnote_3_2" name="footnote_3_1">[3]</a> Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. Restyle: A residual-based "StyleGAN" encoder via iterative refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), pages 6711–6720, 2021. 3

<a href="#footnote_4_2" name="footnote_4_1">[4]</a> Badour AlBahar, Jingwan Lu, Jimei Yang, Zhixin Shu, Eli Shechtman, and Jia-Bin Huang. Pose with Style: Detailpreserving pose-guided image synthesis with conditional stylegan. ACM Transactions on Graphics, 2021. 3

<a href="#footnote_5_2" name="footnote_5_1">[5]</a> Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In International Conference on Learning Representations (ICLR), 2019. 2 

<a href="#footnote_6_2" name="footnote_6_1">[6]</a> Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh. OpenPose: Realtime multi-person 2D pose estimation using part affinity fields. IEEE Transactions on Pattern Analysis & Machine Intelligence, 43(01):172–186, 2021. 6

<a href="#footnote_7_2" name="footnote_7_1">[7]</a> Anna Fruhst ¨ uck, Ibraheem Alhashim, and Peter Wonka. Ti- leGAN: Synthesis of large-scale non-homogeneous textures. ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH), 38(4):58:1–58:11, 2019. 1

<a href="#footnote_8_2" name="footnote_8_1">[8]</a> Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2

<a href="#footnote_9_2" name="footnote_9_1">[9]</a> Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 2

<a href="#footnote_10_2" name="footnote_10_1">[10]</a> Jinfeng Jiang, Guiqing Li, Shihao Wu, Huiqian Zhang, and Yongwei Nie. BPA-GAN: Human motion transfer using body-part-aware generative adversarial networks. Graphical Models, 115:101107, 2021. 3

<a href="#footnote_11_2" name="footnote_11_1">[11]</a> Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In Proceedings of the IEEE Conference on Neural Information Processing Systems (NeurIPS), 2020. 2, 3, 7

<a href="#footnote_12_2" name="footnote_12_1">[12]</a> Tero Karras, Miika Aittala, Samuli Laine, Erik Hark ¨ onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In Proc. NeurIPS, 2021. 2

<a href="#footnote_13_2" name="footnote_13_1">[13]</a> Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), pages 4401–4410, 2019. 2

<a href="#footnote_14_2" name="footnote_14_1">[14]</a> Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), pages 8107–8116, 2020. 1, 2, 3

<a href="#footnote_15_2" name="footnote_15_1">[15]</a> Markus Knoche, Istvan Sarandi, and Bastian Leibe. Reposing humans by warping 3D features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2020. 3

<a href="#footnote_16_2" name="footnote_16_1">[16]</a> Sheela Raju Kurupathi, Pramod Murthy, and Didier Stricker. Generation of human images with clothing using advanced conditional generative adversarial networks. In DeLTA, 2020. 3

<a href="#footnote_17_2" name="footnote_17_1">[17]</a> Tuomas Kynka¨anniemi, Tero Karras, Samuli Laine, Jaakko ¨ Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in Neural Information Processing Systems, 32, 2019. 2

<a href="#footnote_18_2" name="footnote_18_1">[18]</a> Kathleen M Lewis, Srivatsan Varadharajan, and Ira Kemelmacher-Shlizerman. TryOnGAN: Body-aware try-on via layered interpolation. ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH), 40(4), 2021. 3

<a href="#footnote_19_2" name="footnote_19_1">[19]</a> Yijun Li, Lu Jiang, and Ming-Hsuan Yang. Controllable and progressive image extrapolation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2140–2149, January 2021. 3, 7

<a href="#footnote_20_2" name="footnote_20_1">[20]</a> Yuheng Li, Yijun Li, Jingwan Lu, Eli Shechtman, Yong Jae Lee, and Krishna Kumar Singh. Collaging class-specific GANs for semantic image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021. 2, 3, 7

<a href="#footnote_21_2" name="footnote_21_1">[21]</a> Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. DeepFashion: Powering robust clothes recognition and retrieval with rich annotations. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), June 2016. 7

<a href="#footnote_22_2" name="footnote_22_1">[22]</a> Liqian Ma, Xu Jia, Qianru Sun, Bernt Schiele, Tinne Tuytelaars, and Luc Van Gool. Pose guided person image generation. In Advances in Neural Information Processing Systems, pages 405–415, 2017. 3

<a href="#footnote_23_2" name="footnote_23_1">[23]</a> Yifang Men, Yiming Mao, Yuning Jiang, Wei-Ying Ma, and Zhouhui Lian. Controllable person image synthesis with attribute-decomposed GAN. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 3

<a href="#footnote_24_2" name="footnote_24_1">[24]</a> Sachit Menon, Alexandru Damian, Shijia Hu, Nikhil Ravi, and Cynthia Rudin. PULSE: Self-supervised photo upsampling via latent space exploration of generative models. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), pages 2437–2445, 2020. 6

<a href="#footnote_25_2" name="footnote_25_1">[25]</a> Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in Style: a "StyleGAN" encoder for image-to-image translation. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), June 2021. 3

<a href="#footnote_26_2" name="footnote_26_1">[26]</a> Soubhik Sanyal, Alex Vorobiov, Timo Bolkart, Matthew Loper, Betty Mohler, Larry S. Davis, Javier Romero, and Michael J. Black. Learning realistic human reposing using cyclic self-supervision with 3D shape, pose, and appearance consistency. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11138–11147, October 2021. 3

<a href="#footnote_27_2" name="footnote_27_1">[27]</a> Florian Schroff, Dmitry Kalenichenko, and James Philbin. FaceNet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), pages 815–823, June 2015. 7

<a href="#footnote_28_2" name="footnote_28_1">[28]</a> Aliaksandr Siarohin, Stephane Lathuili ´ ere, Enver Sangineto, ` and Nicu Sebe. Appearance and Pose-Conditioned human image generation using deformable GANs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019. 3

<a href="#footnote_29_2" name="footnote_29_1">[29]</a> Piotr Teterwak, Aaron Sarna, Dilip Krishnan, Aaron Maschinot, David Belanger, Ce Liu, and William T. Freeman. Boundless: Generative adversarial networks for image extension. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019. 3

<a href="#footnote_30_2" name="footnote_30_1">[30]</a> Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for "StyleGAN" image manipulation. ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH), 40(4), jul 2021. 3, 5

<a href="#footnote_31_2" name="footnote_31_1">[31]</a> Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. GAN inversion: A survey. CoRR, abs/2101.05278, 2021. 4

<a href="#footnote_32_2" name="footnote_32_1">[32]</a> Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 4

<a href="#footnote_33_2" name="footnote_33_1">[33]</a> Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I Chang, and Yan Xu. Large scale image completion via Co-Modulated generative adversarial networks. In International Conference on Learning Representations (ICLR), 2021. 3, 7, 8

<a href="#footnote_34_2" name="footnote_34_1">[34]</a> Zibo Zhao, Wen Liu, Yanyu Xu, Xianing Chen, Weixin Luo, Lei Jin, Bohui Zhu, Tong Liu, Binqiang Zhao, and Shenghua Gao. Prior based human completion. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), pages 7951–7961, June 2021. 3

<a href="#footnote_35_2" name="footnote_35_1">[35]</a> Qiang Zhou, Shiyin Wang, Yitong Wang, Zilong Huang, and Xinggang Wang. Human de-occlusion: Invisible perception and recovery for humans. CoRR, abs/2103.11597, 2021.