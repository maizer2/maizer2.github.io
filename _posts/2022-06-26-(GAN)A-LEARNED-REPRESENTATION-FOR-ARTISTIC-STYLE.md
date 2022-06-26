---
layout: post 
title: "(GAN)A LEARNED REPRESENTATION FOR ARTISTIC STYLE"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review, 1.2.2.5. GAN]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

### [$$\mathbf{A\;LEARNED\;REPRESENTATION\;FOR\;ARTISTIC\;STYLE}$$](https://arxiv.org/pdf/1610.07629.pdf)

#### $$\mathbf{Vincent\;Dumoulin\;,\;Jonathon\;Shlens,\;Manjunath\;Kudlur}$$

### $\mathbf{Abstract}$

> The diversity of painting styles represents a rich visual vocabulary for the construction of an image. The degree to which one may learn and parsimoniously capture this visual vocabulary measures our understanding of the higher level features of paintings, if not images in general. In this work we investigate the construction of a single, scalable deep network that can parsimoniously capture the artistic style of a diversity of paintings. We demonstrate that such a network generalizes across a diversity of artistic styles by reducing a painting to a point in an embedding space. Importantly, this model permits a user to explore new painting styles by arbitrarily combining the styles learned from individual paintings. We hope that this work provides a useful step towards building rich models of paintings and offers a window on to the structure of the learned representation of artistic style.
>> 그림 스타일의 다양성은 이미지 구성을 위한 풍부한 시각적 어휘를 나타냅니다. 이 시각적 어휘를 배우고 인색하게 포착할 수 있는 정도는 일반적으로는 아니지만 그림의 더 높은 수준의 특징에 대한 우리의 이해를 측정합니다. 이 작품에서 우리는 다양한 그림의 예술 스타일을 인색하게 포착할 수 있는 확장 가능한 단일 심층 네트워크의 구성을 조사합니다. 우리는 그러한 네트워크가 그림을 임베딩 공간의 한 점으로 줄임으로써 다양한 예술 스타일에 걸쳐 일반화된다는 것을 보여줍니다. 중요한 것은, 이 모델은 사용자가 개별 그림에서 배운 스타일을 임의로 결합하여 새로운 그림 스타일을 탐색할 수 있도록 한다는 것입니다. 우리는 이 작품이 풍부한 그림 모델을 구축하기 위한 유용한 단계를 제공하고 예술적 스타일의 학습된 표현 구조에 대한 창구를 제공하기를 바랍니다.

### $\mathbf{1.\;Introduction}$

> A pastiche is an artistic work that imitates the style of another one. Computer vision and more recently machine learning have a history of trying to automate pastiche, that is, render an image in the style of another one. This task is called style transfer, and is closely related to the texture synthesis task. While the latter tries to capture the statistical relationship between the pixels of a source image which is assumed to have a stationary distribution at some scale, the former does so while also attempting to preserve some notion of content.
>> 페이시체는 다른 것의 스타일을 모방한 예술 작품입니다. 컴퓨터 비전과 더 최근의 기계 학습은 페이시체를 자동화하려고 시도한 역사를 가지고 있습니다. 즉, 이미지를 다른 스타일의 이미지로 렌더링합니다. 이 작업을 스타일 전송이라고 하며 텍스처 합성 작업과 밀접한 관련이 있습니다. 후자는 일정한 규모의 정지 분포를 갖는 것으로 가정되는 소스 이미지의 픽셀들 사이의 통계적 관계를 포착하려고 하는 반면, 전자는 콘텐츠의 일부 개념을 보존하려고 시도합니다.

> On the computer vision side, Efros & Leung (1999) and Wei & Levoy (2000) attempt to “grow” textures one pixel at a time using non-parametric sampling of pixels in an examplar image. Efros & Freeman (2001) and Liang et al. (2001) extend this idea to “growing” textures one patch at a time, and Efros & Freeman (2001) uses the approach to implement “texture transfer”, i.e. transfering the texture of an object onto another one. Kwatra et al. (2005) approaches the texture synthesis problem from an energy minimization perspective, progressively refining the texture using an EMlike algorithm. Hertzmann et al. (2001) introduces the concept of “image analogies”: given a pair of “unfiltered” and “filtered” versions of an examplar image, a target image is processed to create an analogous “filtered” result. More recently, Frigo et al. (2016) treats style transfer as a local texture transfer (using an adaptive patch partition) followed by a global color transfer, and Elad & Milanfar (2016) extends Kwatra’s energy-based method into a style transfer algorithm by taking content similarity into account.
>> 컴퓨터 비전 측면에서, Efros & Leung (1999)과 Wei & Levoy (2000)는 예제 이미지에서 픽셀의 비모수 샘플링을 사용하여 한 번에 한 픽셀의 텍스처를 "성장"하려고 시도합니다. Efros & Freeman (2001)과 Liang 외 연구진입니다. (2001)은 이 아이디어를 한 번에 하나의 패치로 "성장"하는 텍스처로 확장하고, 에프로스 & 프리먼 (2001)은 " 텍스처 전송"을 구현하기 위해 접근 방식을 사용합니다. 즉, 객체의 텍스처를 다른 패치로 옮기는 것입니다. 콰트라 외입니다. (2005)는 에너지 최소화 관점에서 텍스처 합성 문제에 접근하여 EM 라이크 알고리즘을 사용하여 텍스처를 점진적으로 개선합니다. 헤르츠만 외입니다. (2001)은 "이미지 유사성"의 개념을 도입했습니다: 샘플 이미지의 "필터링된" 버전과 "필터링된" 버전의 한 쌍이 주어지면, 대상 이미지가 유사한 "유사한" 결과를 생성하도록 처리됩니다. 보다 최근에는 Frigo 외 연구진(2016)은 스타일 전송을 (적응 패치 파티션을 사용하여) 로컬 텍스처 전송에 이어 글로벌 컬러 전송으로 취급하고, Elad & Milanfar(2016)는 콘텐츠 유사성을 고려하여 Kwatra의 에너지 기반 방법을 스타일 전송 알고리듬으로 확장합니다.

> On the machine learning side, it has been shown that a trained classifier can be used as a feature extractor to drive texture synthesis and style transfer. Gatys et al. (2015a) uses the VGG-19 network (Simonyan & Zisserman, 2014) to extract features from a texture image and a synthesized texture. The two sets of features are compared and the synthesized texture is modified by gradient descent so that the two sets of features are as close as possible. Gatys et al. (2015b) extends this idea to style transfer by adding the constraint that the synthesized image also be close to a content image with respect to another set of features extracted by the trained VGG-19 classifier.
>> 기계 학습 측면에서, 훈련된 분류기가 텍스처 합성 및 스타일 전송을 유도하는 특징 추출기로 사용될 수 있는 것으로 나타났습니다. Gatys 등(2015a)은 VGG-19 네트워크(Simonyan & Zisserman, 2014)를 사용하여 텍스처 이미지와 합성 텍스처로부터 특징을 추출합니다. 두 피쳐 세트가 비교되고 합성된 텍스처가 그라데이션 강하로 수정되어 두 피쳐 세트가 최대한 가깝게 됩니다. Gatys 외 연구진(2015b)은 훈련된 VGG-19 분류기에 의해 추출된 다른 기능 세트에 관해서도 합성 이미지가 콘텐츠 이미지에 가깝다는 제약 조건을 추가하여 이 아이디어를 스타일 전송으로 확장합니다.

> While very flexible, this algorithm is expensive to run due to the optimization loop being carried. Ulyanov et al. (2016a), Li & Wand (2016) and Johnson et al. (2016) tackle this problem by introducing a feedforward style transfer network, which is trained to go from content to pastiche image in one pass. However, in doing so some of the flexibility of the original algorithm is lost: the style transfer network is tied to a single style, which means that separate networks have to be trained for every style being modeled. Subsequent work has brought some performance improvements to style transfer networks, e.g. with respect to color preservation (Gatys et al., 2016a) or style transfer quality (Ulyanov et al., 2016b), but to our knowledge the problem of the single-purpose nature of style transfer networks remains untackled.
>> 이 알고리즘은 매우 유연하지만, 실행 중인 최적화 루프 때문에 비용이 많이 듭니다. Ulyanov 등(2016a), Li & Wand(2016), Johnson 등(2016)은 콘텐츠에서 페이시 이미지로 한 번에 이동하도록 훈련된 피드포워드 스타일 전송 네트워크를 도입하여 이 문제를 해결합니다. 그러나 그렇게 함으로써 원래 알고리듬의 유연성이 일부 손실됩니다. 스타일 전송 네트워크는 단일 스타일에 묶여 있으며, 이는 모델링되는 모든 스타일에 대해 별도의 네트워크를 훈련해야 한다는 것을 의미합니다. 후속 작업으로 색상 보존(Gatys et al., 2016a) 또는 스타일 전송 품질(Ulyanov et al., 2016b)과 관련하여 스타일 전송 네트워크에 일부 성능 향상을 가져왔지만, 우리가 아는 한 스타일 전송 네트워크의 단일 목적 특성에 대한 문제는 해결되지 않았습니다.

![Figure 1a](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-26-(GAN)A-LEARNED-REPRESENTATION-FOR-ARTISTIC-STYLE/Figure-1a.JPG)

> (a) With conditional instance normalization, a single style transfer network can capture 32 styles at the same time, five of which are shown here. All 32 styles in this single model are in the Appendix. Golden Gate Bridge photograph by Rich Niewiroski Jr.
>> (a) 조건부 인스턴스 정규화를 사용하면 단일 스타일 전송 네트워크가 동시에 32개의 스타일을 캡처할 수 있으며, 그 중 5개는 여기에 나와 있습니다. 이 단일 모델의 32가지 스타일은 모두 부록에 나와 있습니다. 금문교 사진: 리치 니에이로스키 주니어입니다.

![Figure 1b](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-26-(GAN)A-LEARNED-REPRESENTATION-FOR-ARTISTIC-STYLE/Figure-1b.JPG)

> (b) The style representation learned via conditional instance normalization permits the arbitrary combination of artistic styles. Each pastiche in the sequence corresponds to a different step in interpolating between the $\gamma{}$ and $\beta{}$ values associated with two styles the model was trained on.
>> (b) The style representation learned via conditional instance normalization permits the arbitrary combination of artistic styles. Each pastiche in the sequence corresponds to a different step in interpolating between the $\gamma{}$ and $\beta{}$ values associated with two styles the model was trained on.

> Figure 1: Pastiches produced by a style transfer network trained on 32 styles chosen for their variety.
>> 그림 1: 스타일 전송 네트워크에 의해 생성된 페이스트(Pastches)는 32가지 스타일을 선택하여 학습합니다.

> We think this is an important problem that, if solved, would have both scientific and practical importance. First, style transfer has already found use in mobile applications, for which on-device processing is contingent upon the models having a reasonable memory footprint. More broadly, building a separate network for each style ignores the fact that individual paintings share many common visual elements and a true model that captures artistic style would be able to exploit and learn from such regularities. Furthermore, the degree to which an artistic styling model might generalize across painting styles would directly measure our ability to build systems that parsimoniously capture the higher level features and statistics of photographs and images (Simoncelli & Olshausen, 2001).
>> 우리는 이것이 해결된다면, 과학적이고 실제적인 중요성을 둘 다 가질 수 있는 중요한 문제라고 생각합니다. 첫째, 스타일 전송은 합리적인 메모리 공간을 가진 모델에 따라 장치 내 처리가 좌우되는 모바일 애플리케이션에서 이미 사용되고 있습니다. 더 넓게 말하면, 각각의 스타일을 위해 별도의 네트워크를 구축하는 것은 개별 그림들이 많은 공통적인 시각적 요소들을 공유하고 있고 예술적인 스타일을 포착하는 진정한 모델이 그러한 규칙성을 이용하고 배울 수 있다는 사실을 무시합니다. 게다가, 예술적 스타일 모델이 그림 스타일에 걸쳐 일반화 될 수 있는 정도는 사진과 이미지의 더 높은 수준의 특징과 통계를 인색하게 포착하는 시스템을 구축하는 우리의 능력을 직접 측정할 것입니다(Simoncelli & Olshausen, 2001).

> In this work, we show that a simple modification of the style transfer network, namely the introduction of conditional instance normalization, allows it to learn multiple styles (Figure 1a).We demonstrate that this approach is flexible yet comparable to single-purpose style transfer networks, both qualitatively and in terms of convergence properties. This model reduces each style image into a point in an embedding space. Furthermore, this model provides a generic representation for artistic styles that seems flexible enough to capture new artistic styles much faster than a single-purpose network. Finally, we show that the embeddding space representation permits one to arbitrarily combine artistic styles in novel ways not previously observed (Figure 1b).
>> 본 연구에서는 스타일 전송 네트워크의 간단한 수정, 즉 조건부 인스턴스 정규화의 도입으로 여러 스타일을 학습할 수 있음을 보여줍니다(그림 1a).우리는 이 접근 방식이 유연하지만 질적으로나 수렴 속성 측면에서 단일 목적 스타일 전송 네트워크에 필적한다는 것을 보여줍니다. 이 모델은 각 스타일 이미지를 포함 공간의 한 점으로 축소합니다. 또한 이 모델은 단일 목적 네트워크보다 훨씬 빠르게 새로운 예술 스타일을 포착할 수 있을 정도로 유연해 보이는 예술 스타일에 대한 일반적인 표현을 제공합니다. 마지막으로, 임베딩 공간 표현을 통해 이전에 관찰되지 않은 새로운 방식으로 예술 스타일을 임의로 결합할 수 있음을 보여줍니다(그림 1b).

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-26-(GAN)A-LEARNED-REPRESENTATION-FOR-ARTISTIC-STYLE/Figure-2.JPG)

> Figure 2: Style transfer network training diagram (Johnson et al., 2016; Ulyanov et al., 2016a). A pastiche image is produced by feeding a content image through the style transfer network. The two images, along with a style image, are passed through a trained classifier, and the resulting intermediate representations are used to compute the content loss$L_{c}$and style loss $L_{s}$. The parameters of the classifier are kept fixed throughout training.
>> 그림 2: 스타일 전송 네트워크 교육 다이어그램(Johnson et al., 2016; Ulyanov et al., 2016a)입니다. 스타일 전송 네트워크를 통해 컨텐츠 이미지를 공급하여 페이시 이미지를 생성합니다. 두 이미지는 스타일 이미지와 함께 훈련된 분류기를 통과하며, 그 결과 중간 표현은 콘텐츠 손실 $L_{c}$ 및 스타일 손실 $L_{s}$을 계산하는 데 사용됩니다. 분류기의 매개 변수는 교육 내내 고정됩니다.

### $\mathbf{2\;STYLE\;TRANSFER\;WITH\;DEEP\;NETWORKS}$

> Style transfer can be defined as finding a pastiche image $p$ whose content is similar to that of a content image c but whose style is similar to that of a style image s. This objective is by nature vaguely defined, because similarity in content and style are themselves vaguely defined.
>> 스타일 전송은 컨텐츠 이미지 c의 내용과 비슷하지만 스타일이 스타일 이미지와 유사한 페이시 이미지 $p$를 찾는 것으로 정의할 수 있습니다. 이 목표는 본질적으로 모호하게 정의됩니다. 왜냐하면 내용과 스타일의 유사성 자체가 모호하게 정의되기 때문입니다.

> The neural algorithm of artistic style proposes the following definitions:
>> 예술 스타일의 신경 알고리즘은 다음과 같은 정의를 제안합니다.

* > Two images are similar in content if their high-level features as extracted by a trained classifier are close in Euclidian distance.
    >> 훈련된 분류기에 의해 추출된 높은 수준의 기능이 유클리드 거리에 가까울 경우 두 이미지는 내용이 유사합니다.

* > Two images are similar in style if their low-level features as extracted by a trained classifier share the same statistics or, more concretely, if the difference between the features’ Gram matrices has a small Frobenius norm.
    >> 훈련된 분류기에 의해 추출된 낮은 수준의 형상이 동일한 통계를 공유하는 경우 또는 형상의 그램 매트릭스 간의 차이가 작은 Frobenius 규범을 갖는 경우 두 이미지는 스타일이 유사합니다.

> The first point is motivated by the empirical observation that high-level features in classifiers tend to correspond to higher levels of abstractions (see Zeiler & Fergus (2014) for visualizations; see Johnson et al. (2016) for style transfer features). The second point is motivated by the observation that the artistic style of a painting may be interpreted as a visual texture (Gatys et al., 2015a). A visual texture is conjectured to be spatially homogenous and consist of repeated structural motifs whose minimal sufficient statistics are captured by lower order statistical measurements (Julesz, 1962; Portilla & Simoncelli, 1999).
>> 첫 번째 포인트는 분류기의 고급 기능이 더 높은 수준의 추상화에 대응하는 경향이 있다는 경험적 관찰에 의해 동기가 부여됩니다(시각화는 Zeiler & Fergus(2014년) 참조, 스타일 전송 기능은 Johnson 등(2016년) 참조). 두 번째 점은 그림의 예술적 스타일이 시각적 질감으로 해석될 수 있다는 관찰에 의해 동기가 부여됩니다(Gatys et al., 2015a). 시각적 텍스처는 공간적으로 균질하며 저차 통계 측정에 의해 최소 충분한 통계량을 포착하는 반복적인 구조적 모티프로 구성됩니다(Julesz, 1962; Portilla & Simoncelli, 1999).

> In its original formulation, the neural algorithm of artistic style proceeds as follows: starting from some initialization of $p$ (e.g. c, or some random initialization), the algorithm adapts $p$ to minimize the loss function
>> 원래 공식에서 예술 스타일의 신경 알고리듬은 다음과 같이 진행됩니다. $p$의 일부 초기화(예: c 또는 일부 무작위 초기화)에서 시작하여 알고리듬은 손실 함수를 최소화하기 위해 $p$를 조정한다.

$$L(s,c,p)=λ_{s}L_{s}(p)λ_{c}L_{c}(p),$$

> where $L_{s}(p)$ is the style loss,$L_{c}(p)$ is the content loss and $λ_{s}$, $λ_{c}$ are scaling hyperparameters. Given a set of “style layers” $S$ and a set of “content layers” $C$, the style and content losses are themselves defined as
>> 여기서 $L_{s}(p)$는 스타일 손실이고 $L_{c}(p)$는 콘텐츠 손실이며 $__{s}$, $__{c}$는 스케일링 하이퍼 매개 변수입니다. "스타일 레이어" $S$ 집합과 "콘텐츠 레이어" $C$ 집합이 주어지면 스타일 및 콘텐츠 손실 자체는 다음과 같이 정의됩니다.

$$L_{s}(p)=\sum_{i\in{S}}\frac{1}{U_{i}}\vert{}\vert{}G(φ_{i}(p))-G(φ_{i}(s))\vert{}\vert{}_{F}^{2}$$

$$L_{c}(p)=\sum_{i\in{C}}\frac{1}{U_{j}}\vert{}\vert{}φ_{j}(p)−φ_{j}(c)\vert{}\vert{}_{2}^{2}$$

> where $φ_{l}(x)$ are the classifier activations at layer $l$, $U_{l}$ is the total number of units at layer l and $G(φ_{l}(x))$ is the Gram matrix associated with the layer l activations. In practice, we set $λ_{c}=1.0$ and and leave $λ_{s}$ as a free hyper-parameter.
>> 여기서 $θ_{l}(x)$는 계층 $l$의 분류자 활성화이고, $U_{l}$는 계층 l의 총 단위 수이며 $G(θ_{l}(x))$는 계층 활성화와 연결된 Gram 행렬입니다. 실제로 $syslog_{c}=1.0$을 설정하고 $syslog_{s}$를 자유 하이퍼 매개 변수로 남깁니다.

> In order to speed up the procedure outlined above, a feed-forward convolutional network, termed a style transfer network $T$, is introduced to learn the transformation (Johnson et al., 2016; Li & Wand, 2016; Ulyanov et al., 2016a). It takes as input a content image c and outputs the pastiche image $p$ directly (Figure 2). The network is trained on many content images (Deng et al., 2009) using the same loss function as above, i.e.
>> 위에 설명된 절차를 가속화하기 위해 스타일 전송 네트워크 $T$라고 하는 피드포워드 컨볼루션 네트워크를 도입하여 변환을 학습합니다(Johnson et al., 2016; Li & Wand, 2016; Ulyanov et al., 2016a). 컨텐츠 이미지 c를 입력으로 사용하고 페이시 이미지 $p$를 직접 출력합니다(그림 2). 네트워크는 위와 같은 손실 함수를 사용하여 많은 콘텐츠 이미지(Deng et al., 2009)에 대해 훈련됩니다.

$$L(s,c)=λ_{s}L_{s}(T(c))+λ_{c}L_{c}(T(c)).$$

> While feedforward style transfer networks solve the problem of speed at test-time, they also suffer from the fact that the network $T$ is tied to one specific painting style. This means that a separate network $T$ has to be trained for every style to be imitated. The real-world impact of this limitation is that it becomes prohibitive to implement a style transfer application on a memory-limited device, such as a smartphone.
>> 피드포워드 스타일 전송 네트워크는 테스트 시 속도 문제를 해결하지만, 네트워크 $T$가 하나의 특정 그림 스타일에 묶여 있다는 사실 때문에 어려움을 겪기도 합니다. 이는 모방할 모든 스타일에 대해 별도의 네트워크 $T$를 훈련해야 한다는 것을 의미합니다. 이러한 제한이 실제로 미치는 영향은 스마트폰과 같은 메모리 제한 장치에 스타일 전송 응용 프로그램을 구현하는 것이 금지된다는 것입니다.

#### $\mathbf{2.1\;N-STYLES\;FEEDFORWARD\;STYLE\;TRANSFER\;NETWORKS}$

> Our work stems from the intuition that many styles probably share some degree of computation, and that this sharing is thrown away by training $N$ networks from scratch when building an Nstyles style transfer system. For instance, many impressionist paintings share similar paint strokes but differ in the color palette being used. In that case, it seems very wasteful to treat a set of $N$ impressionist paintings as completely separate styles.
>> 우리의 작업은 많은 스타일이 어느 정도의 계산을 공유할 수 있으며, 이러한 공유는 Nstyles 스타일 전송 시스템을 구축할 때 $N$ 네트워크를 처음부터 훈련시킴으로써 버려진다는 직관에서 비롯됩니다. 예를 들어, 많은 인상주의 그림들은 비슷한 화법을 공유하지만 사용되는 색 팔레트에 차이가 있습니다. 그런 경우, $N$ 인상파 그림 세트를 완전히 별개의 스타일로 취급하는 것은 매우 낭비적으로 보입니다.

> To take this into account, we propose to train a single conditional style transfer network $T(c,s)$ for  $N$ styles. The conditional network is given both a content image and the identity of the style to apply and produces a pastiche corresponding to that style. While the idea is straightforward on paper, there remains the open question of how conditioning should be done. In exploring this question, we found a very surprising fact about the role of normalization in style transfer networks: to model a style, it is sufficient to specialize scaling and shifting parameters after normalization to each specific style. In other words, all convolutional weights of a style transfer network can be shared across many styles, and it is sufficient to tune parameters for an affine transformation after normalization for each style.
>> 이를 고려하여 $N$ 스타일에 대해 단일 조건부 스타일 전송 네트워크 $T(c,s)$를 훈련할 것을 제안한다. 조건부 네트워크에는 적용할 스타일의 이미지와 ID가 모두 제공되고 해당 스타일에 해당하는 파티쉐가 생성됩니다. 아이디어는 서류상으로는 간단하지만, 컨디셔닝이 어떻게 이루어져야 하는지에 대한 미해결 문제가 남아 있습니다. 이 질문을 탐색하면서 스타일 전송 네트워크에서 정규화의 역할에 대한 매우 놀라운 사실을 발견했습니다. 스타일을 모델링하려면 각 특정 스타일에 정규화 후 매개 변수를 전문화하면 충분합니다. 즉, 스타일 전송 네트워크의 모든 컨볼루션 가중치는 많은 스타일에서 공유될 수 있으며, 각 스타일에 대한 정규화 후 아핀 변환에 대한 매개 변수를 조정하기에 충분합니다.

> We call this approach conditional instance normalization. The goal of the procedure is transform a layer’s activations $x$ into a normalized activation $z$ specific to painting style $s$. Building off the instance normalization technique proposed in Ulyanov et al. (2016b), we augment the $\gamma{}$ and $\beta{}$ parameters so that they’re $N\times{}C$ matrices, where $N$ is the number of styles being modeled and $C$ is the number of output feature maps. Conditioning on a style is achieved as follows:
>> 우리는 이 접근법을 조건부 인스턴스 정규화라고 부릅니다. 이 절차의 목표는 레이어의 활성화 $x$를 페인트 스타일 $s$에 특정한 정규화된 활성화 $z$로 변환하는 것입니다. Ulyanov 등(2016b)에서 제안된 인스턴스 정규화 기법을 바탕으로 $α$ 및 $\beta{}$ 매개 변수를 $N\times{}C$ 행렬이 되도록 증강한다. 여기서 $N$은 모델링되는 스타일 수이고 $C$는 출력 기능 맵 수이다. 스타일에 대한 조건화는 다음과 같이 수행됩니다.

$$z=γ_{s}\frac{x−µ}{σ}+β_{s}$$

> where $µ$ and $σ$ are $x$’s mean and standard deviation taken across spatial axes and $λ_{s}$ and $β_{s}$ are obtained by selecting the row corresponding to s in the $\gamma{}$ and $\beta{}$ matrices (Figure 3). One added benefit of this approach is that one can stylize a single image into $N$ painting styles with a single feed forward pass of the network with a batch size of $N$. In constrast, a single-style network requires $N$ feed forward passes to perform $N$ style transfers (Johnson et al., 2016; Li & Wand, 2016; Ulyanov et al., 2016a).
>> 여기서 $µ$와 $σ$는 공간 축에서 취한 $x$의 평균 및 표준 편차이며 $λ_{s}$와 $β_{s}$는 $\gamma{}$ 및 $\beta{}$ 행렬에서 s에 해당하는 행을 선택하여 얻습니다(그림 3). 이 접근 방식의 한 가지 추가적인 이점은 배치 크기가 $N$인 네트워크의 단일 피드 포워드 패스로 단일 이미지를 $N$ 페인팅 스타일로 스타일링할 수 있다는 것입니다. 대조적으로, 단일 스타일 네트워크는 $N$ 스타일 전송을 수행하기 위해 $N$ 피드 포워드 패스가 필요합니다(Johnson 등, 2016; Li & Wand, 2016; Ulyanov 등, 2016a).

> Because conditional instance normalization only acts on the scaling and shifting parameters, training a style transfer network on $N$ styles requires fewer parameters than the naive approach of training $N$ separate networks. In a typical network setup, the model consists of roughly 1.6M parameters, only around 3K (or 0.2%) of which specify individual artistic styles. In fact, because the size of $\gamma{}$ and $\beta{}$ grows linearly with respect to the number of feature maps in the network, this approach requires $O(N\times{}L)$ parameters, where $L$ is the total number of feature maps in the network.
>> 조건부 인스턴스 정규화는 확장 및 이동 매개 변수에만 작용하기 때문에 $N$ 스타일에서 스타일 전송 네트워크를 훈련하는 것은 $N$ 개별 네트워크를 훈련하는 순진한 접근 방식보다 매개 변수가 더 적게 필요합니다. 일반적인 네트워크 설정에서 모델은 약 160만 개의 매개 변수로 구성되며, 이 매개 변수 중 약 3K(또는 0.2%)만 개별 예술 스타일을 지정합니다. 실제로, $\gamma{}$ 및 $\beta{}$의 크기는 네트워크의 기능 맵 수에 따라 선형적으로 증가하기 때문에, 이 접근 방식에는 $O(N\times{}L)$ 매개 변수가 필요합니다. 여기서 $L$은 네트워크에 있는 기능 맵의 총 수입니다.

> In addition, as is discussed in subsection 3.4, conditional instance normalization presents the advantage that integrating an $N+1th$ style to the network is cheap because of the very small number of parameters to train
>> 또한, 서브섹션 3.4에서 논의된 바와 같이, 조건부 인스턴스 정규화는 훈련할 매개 변수의 수가 매우 적기 때문에 $N+1th$ 스타일을 네트워크에 통합하는 것이 저렴하다는 이점을 제공합니다.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-26-(GAN)A-LEARNED-REPRESENTATION-FOR-ARTISTIC-STYLE/Figure-3.JPG)

> Figure 3: Conditional instance normalization. The input activation $x$ is normalized across both spatial dimensions and subsequently scaled and shifted using style-dependent parameter vectors $λ_{s},β_{s}$ where s indexes the style label.
>> 그림 3: 조건부 예를 정상화다.이 입력은 활성화달러 $x$은차원 공간을 가로질러 그리고 계속하여 기어 올라가고 사용하는 옮겨 정상화되style-dependent 매개 변수 벡터달러 $λ_{s},β_{s}$ 어디 가서는 스타일 라벨을 인덱싱 하는 방법.

### $\mathbf{3\;EXPERIMENTAL\;RESULTS}$

#### $\mathbf{3.1\;METHODOLOGY}$

> Unless noted otherwise, all style transfer networks were trained using the hyperparameters outlined in the Appendix’s Table 1. We used the same network architecture as in Johnson et al. (2016), except for two key details: zero-padding is replaced with mirror-padding, and transposed convolutions (also sometimes called deconvolutions) are replaced with nearest-neighbor upsampling followed by a convolution. The use of mirror-padding avoids border patterns sometimes caused by zero-padding in SAME-padded convolutions, while the replacement for transposed convolutions avoids checkerboard patterning, as discussed in in Odena et al. (2016). We find that with these two improvements training the network no longer requires a total variation loss that was previously employed to remove high frequency noise as proposed in Johnson et al. (2016).
>> 달리 언급되지 않는 한, 모든 스타일 전송 네트워크는 부록의 표 1에 설명된 하이퍼 파라미터를 사용하여 교육되었습니다. 우리는 존슨 외 연구진(2016)과 동일한 네트워크 아키텍처를 사용했는데, 제로 패딩은 미러 패딩으로 대체되고, 전치된 컨볼루션(때로는 디콘볼루션이라고도 함)은 가장 가까운 이웃 업샘플링으로 대체된 다음 컨볼루션으로 대체됩니다. 미러 패딩을 사용하면 SAME 패드 컨볼루션의 제로 패딩으로 인해 때때로 발생하는 경계 패턴을 피할 수 있으며, 전치 컨볼루션의 대체는 Odena et al. (2016)에서 논의된 바와 같이 체커보드 패턴을 피할 수 있습니다. 이러한 두 가지 개선 사항으로 우리는 네트워크가 존슨 외 연구진(2016)에서 제안한 대로 고주파 잡음을 제거하기 위해 이전에 채택되었던 총 변동 손실을 더 이상 요구하지 않는다는 것을 발견했습니다.

> Our training procedure follows Johnson et al. (2016). Briefly, we employ the ImageNet dataset (Deng et al., 2009) as a corpus of training content images. We train the N-style network with stochastic gradient descent using the Adam optimizer (Kingma & Ba, 2014). Details of the model architecture are in the Appendix. A complete implementation of the model in TensorFlow (Abadi et al., 2016) as well as a pretrained model are available for download 1 . The evaluation images used for this work were resized such that their smaller side has size 512. Their stylized versions were then center-cropped to 512x512 pixels for display.
>> 우리의 교육 절차는 존슨 외(2016)를 따릅니다. 간단히, 우리는 ImageNet 데이터 세트(Deng 등, 2009)를 교육 콘텐츠 이미지의 코퍼스로 사용합니다. 우리는 Adam 최적화기를 사용하여 확률적 그레이디언트 강하로 N 스타일 네트워크를 훈련합니다(Kingma & Ba, 2014). 모델 아키텍처에 대한 자세한 내용은 부록을 참조하십시오. TensorFlow 모델(Abadi et al., 2016)의 완전한 구현과 사전 훈련된 모델을 다운로드 1에서 사용할 수 있습니다. 이 작업에 사용된 평가 이미지는 크기가 작아서 크기가 512가 되도록 크기를 조정했습니다. 스타일링된 버전은 디스플레이를 위해 512x512 픽셀로 중앙에 잘렸습니다.

#### $\mathbf{3.2\;TRAINING\;A\;SINGLE\;NETWORK\;ON\;N\;STYLES\;PRODUCES\;STYLIZATIONS\;COMPARABLE\;TO\;INDEPENDENTLY\;TRAINED\;MODELS}$

> As a first test, we trained a 10-styles model on stylistically similar images, namely 10 impressionist paintings from Claude Monet. Figure 4 shows the result of applying the trained network on evaluation images for a subset of the styles, with the full results being displayed in the Appendix. The model captures different color palettes and textures. We emphasize that 99.8% of the parameters are shared across all styles in contrast to 0.2% of the parameters which are unique to each painting style.
>> 첫 번째 테스트로 우리는 클로드 모네의 인상파 그림 10점 등 양식적으로 유사한 이미지에 대한 10가지 스타일 모델을 교육했습니다. 그림 4는 일부 스타일의 평가 이미지에 대해 훈련된 네트워크를 적용한 결과를 나타내며 전체 결과는 부록에 표시됩니다. 이 모델은 다양한 색상 팔레트와 질감을 포착합니다. 우리는 99.8%의 매개 변수가 모든 스타일에서 공유된다는 점을 강조합니다. 이는 각 그림 스타일에 고유한 매개 변수의 0.2%와 대조됩니다.

> To get a sense of what is being traded off by folding 10 styles into a single network, we trained a separate, single-style network on each style and compared them to the 10-styles network in terms of style transfer quality and training speed (Figure 5).
>> 10가지 스타일을 단일 네트워크로 접어서 트레이드오프되는 것을 파악하기 위해 각 스타일에서 별도의 단일 스타일 네트워크를 교육하고 스타일 전송 품질 및 교육 속도 측면에서 10가지 스타일 네트워크와 비교했습니다(그림 5).

> The left column compares the learning curves for style and content losses between the single-style networks and the 10-styles network. The losses were averaged over 32 random batches of content images. By visual inspection, we observe that the 10-styles network converges as quickly as the single-style networks in terms of style loss, but lags slightly behind in terms of content loss.
>> 왼쪽 열은 단일 스타일 네트워크와 10 스타일 네트워크 간의 스타일 및 콘텐츠 손실에 대한 학습 곡선을 비교합니다. 손실은 32개의 랜덤 콘텐츠 이미지 배치에서 평균화되었습니다. 육안 검사를 통해 10 스타일 네트워크는 스타일 손실 측면에서 단일 스타일 네트워크만큼 빠르게 수렴하지만 콘텐츠 손실 측면에서 약간 뒤처지는 것을 관찰합니다.

> In order to quantify this observation, we compare the final losses for 10-styles and single-style models (center column). The 10-styles network’s content loss is around $8.7±3.9%$ higher than its single-style counterparts, while the difference in style losses ($8.9±16.5%$ lower) is insignificant. While the N-styles network suffers from a slight decrease in content loss convergence speed, this may not be a fair comparison, given that it takes $N$ times more parameter updates to train $N$ singlestyle networks separately than to train them with an N-styles network.
>> 이 관측치를 정량화하기 위해 10가지 스타일과 단일 스타일 모델(중앙 열)에 대한 최종 손실을 비교합니다. 10 스타일 네트워크의 콘텐츠 손실은 단일 스타일 네트워크보다 약 $8.7±3.9%$ 높은 반면 스타일 손실의 차이($8.9±16.5%$ 낮음)는 미미합니다. N-스타일 네트워크는 콘텐츠 손실 수렴 속도가 약간 감소하지만, N-스타일 네트워크를 사용하여 훈련하는 것보다 $N$개의 단일 스타일 네트워크를 별도로 훈련하는 데 $N$배 더 많은 매개 변수 업데이트가 필요하기 때문에 이는 공정한 비교가 아닐 수 있습니다.

> The right column shows a comparison between the pastiches produced by the 10-styles network and the ones produced by the single-style networks. We see that both results are qualitatively similar.
>> 오른쪽 열은 10 스타일 네트워크에서 생성된 패치와 단일 스타일 네트워크에서 생성된 패시를 비교한 것입니다. 두 결과 모두 질적으로 유사하다는 것을 알 수 있습니다.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-26-(GAN)A-LEARNED-REPRESENTATION-FOR-ARTISTIC-STYLE/Figure-4.JPG)

> Figure 4: A single style transfer network was trained to capture the style of 10 Monet paintings, five of which are shown here. All 10 styles in this single model are in the Appendix. Golden Gate Bridge photograph by Rich Niewiroski Jr.
>> 그림 4: 단일 스타일 전송 네트워크는 10개의 모네 그림의 스타일을 캡처하도록 훈련되었으며, 그 중 5개는 여기에 나와 있습니다. 이 단일 모델의 10가지 스타일은 모두 부록에 있습니다. 금문교 사진: 리치 니에이로스키 주니어입니다.

#### $\mathbf{3.3\;THE\;lN-STYLES\;MODEL\;IS\;FLEXIBLE\;ENOUGH\;TO\;CAPTURE\;VERY\;DIFFERENT\;STYLES}$

> We evaluated the flexibility of the N-styles model by training a style transfer network on 32 works of art chosen for their diversity. Figure 1a shows the result of applying the trained network on evaluation images for a subset of the styles. Once again, the full results are displayed in the Appendix. The model appears to be capable of modeling all 32 styles in spite of the tremendous variation in color palette and the spatial scale of the painting styles.
>> 우리는 N-스타일 모델의 다양성을 위해 선택된 32개의 예술 작품에 대한 스타일 전송 네트워크를 훈련시켜 N-스타일 모델의 유연성을 평가했습니다. 그림 1a는 스타일의 하위 집합에 대한 평가 이미지에 훈련된 네트워크를 적용한 결과를 보여줍니다. 다시 한 번 전체 결과가 부록에 표시됩니다. 이 모델은 색상 팔레트의 엄청난 변동과 그림 스타일의 공간적 규모에도 불구하고 32가지 스타일을 모두 모델링할 수 있는 것으로 보입니다.

#### $\mathbf{3.4\;THE\;TRAINED\;NETWORK\;GENERALIZES\;ACROSS\;PAINTING\;STYLES}$

> Since all weights in the transformer network are shared between styles, one way to incorporate a new style to a trained network is to keep the trained weights fixed and learn a new set of $\gamma{}$ and $\beta{}$ parameters. To test the efficiency of this approach, we used it to incrementally incorporate Monet’s Plum Trees in Blossom painting to the network trained on 32 varied styles. Figure 6 shows that doing so is much faster than training a new network from scratch (left) while yielding comparable pastiches: even after eight times fewer parameter updates than its single-style counterpart, the finetuned model produces comparable pastiches (right).
>> 트랜스포머 네트워크의 모든 가중치는 스타일 간에 공유되므로, 훈련된 네트워크에 새로운 스타일을 통합하는 한 가지 방법은 훈련된 가중치를 고정하고 새로운 $α$ 및 $\beta{}$ 매개 변수 세트를 학습하는 것입니다. 이 접근법의 효율성을 테스트하기 위해, 우리는 32가지 다양한 스타일로 훈련된 네트워크에 Blossom Painting의 Monet's Fum Trees를 점진적으로 통합하는 데 사용했습니다. 그림 6을 보면 새 네트워크를 처음부터 교육하는 것보다 훨씬 빠르며(왼쪽) 비교 가능한 패시치가 생성된다는 것을 알 수 있습니다. 단일 스타일 업데이트보다 8배 적은 매개 변수 업데이트 후에도 미세 조정된 모델은 비교 가능한 패시치(오른쪽)를 생성합니다.

#### $\mathbf{3.5\;THE\;TRAINED\;NETWORK\;CAN\;ARBITRARILY\;COMBINE\;PAINTING\;STYLES}$

> The conditional instance normalization approach raises some interesting questions about style representation. In learning a different set of $\gamma{}$ and $\beta{}$ parameters for every style, we are in some sense learning an embedding of styles.
>> 조건부 인스턴스 정규화 접근법은 스타일 표현에 대한 몇 가지 흥미로운 질문을 제기합니다. 모든 스타일에 대해 서로 다른 $\gamma{}$ 및 $\beta{}$ 매개 변수 세트를 학습할 때, 우리는 어떤 의미에서 스타일의 임베딩을 학습하고 있습니다.

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-26-(GAN)A-LEARNED-REPRESENTATION-FOR-ARTISTIC-STYLE/Figure-5.JPG)

> Figure 5: The N-styles model exhibits learning dynamics comparable to individual models. (Left column) The N-styles model converges slightly slower in terms of content loss (top) and as fast in terms of style loss (bottom) than individual models. Training on a single Monet painting is represented by two curves with the same color. The dashed curve represents the N-styles model, and the full curves represent individual models. Emphasis has been added on the styles for Vetheuil (1902) (teal) and Water Lilies (purple) for visualization purposes; remaining colors correspond to other Monet paintings (see Appendix). (Center column) The N-styles model reaches a slightly higher final content loss than (top, $8.7±3.9%$ increase) and a final style loss comparable to (bottom, $8.9±16.5%$ decrease) individual models. (Right column) Pastiches produced by the N-styles network are qualitatively comparable to those produced by individual networks.
>> 그림 5: N-스타일 모델은 개별 모델과 유사한 학습 역학을 보여줍니다.(왼쪽 열) N-스타일 모델은 내용 손실(위) 측면에서 약간 느리고 스타일 손실(아래) 측면에서 개별 모델보다 빠릅니다. 하나의 모네 그림에 대한 훈련은 같은 색을 가진 두 개의 곡선으로 표현됩니다. 점선 곡선은 N-스타일 모형을 나타내고 전체 곡선은 개별 모형을 나타냅니다. 시각화를 위해 Vetheuil (1902) (teal)과 수련 (보라색)의 스타일을 강조했습니다. 나머지 색상은 다른 모네 그림과 일치합니다(부록 참조). (중앙 열) N-스타일 모델은 (상단, $8.7±3.9%$ 증가)보다 약간 높은 최종 내용 손실과 (하단, $8.9±16.5%$ 감소) 개별 모델에 필적하는 최종 스타일 손실(오른쪽 열) N-스타일 네트워크에서 생산된 패스티치는 개별 네트워크에서 생산된 패스티치와 질적으로 비교할 수 있습니다.

![Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-26-(GAN)A-LEARNED-REPRESENTATION-FOR-ARTISTIC-STYLE/Figure-6.JPG)

> Figure 6: The trained network is efficient at learning new styles. (Left column) Learning $\gamma{}$ and $\beta{}$ from a trained style transfer network converges much faster than training a model from scratch. (Right) Learning $\gamma{}$ and $\beta{}$ for 5,000 steps from a trained style transfer network produces pastiches comparable to that of a single network trained from scratch for 40,000 steps. Conversely, 5,000 step of training from scratch produces leads to a poor pastiche.
>> 그림 6: 훈련된 네트워크는 새로운 스타일을 학습하는 데 효율적입니다. (왼쪽 열) 훈련된 스타일 전송 네트워크에서 $\gamma{}$ 및 $\beta{}$를 학습하면 모델을 처음부터 학습하는 것보다 훨씬 빠르게 수렴됩니다. (오른쪽) 훈련된 스타일 전송 네트워크에서 5,000단계에 대해 $\gamma{}$ 및 $\beta{}$를 학습하면 40,000단계에 대해 처음부터 학습한 단일 네트워크와 비슷한 페이시치가 생성된다. 반대로, 처음부터 5,000 단계의 훈련은 형편없는 페이스티쉬로 이어집니다.

![Figure 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-26-(GAN)A-LEARNED-REPRESENTATION-FOR-ARTISTIC-STYLE/Figure-7.JPG)

> Figure 7: The N-styles network can arbitrarily combine artistic styles. (Left) Combining four styles, shown in the corners. Each pastiche corresponds to a different convex combination of the four styles’ $\gamma{}$ and $\beta{}$ values. (Right) As we transition from one style to another (Bicentennial Print and Head of a Clown in this case), the style losses vary monotonically.
>> 그림 7: N-styles 네트워크는 임의로 예술적 스타일을 결합할 수 있습니다. (왼쪽) 모서리에 표시된 네 가지 스타일을 결합합니다. 각 페이시체는 네 가지 스타일의 $\gamma{}$ 및 $\beta{}$ 값의 서로 다른 볼록 조합에 해당합니다. (오른쪽) 한 스타일에서 다른 스타일(이 경우 바이센테니얼 프린트 및 광대 머리)로 전환함에 따라 스타일 손실은 단조롭게 변화합니다.

> Previous work suggested that cleverly balancing optimization strategies offers an opportunity to blend painting styles 2 . To probe the utility of this embedding, we tried convex combinations of the $\gamma{}$ and $\beta{}$ values to blend very distinct painting styles (Figure 1b; Figure 7, left column). Employing a single convex combination produces a smooth transition from one style to the other. Suppose ($λ_{1}, β_{1}$) and ($λ_{2},β_{2}$) are the parameters corresponding to two different styles. We use $γ=α\times{}γ_{1}+(1−α)\times{}γ_{2}$ and $β=α\times{}β_{1}+(1−α)\times{}β_{2}$ to stylize an image. Employing convex combinations may be extended to an arbitrary number of styles 3 . Figure 7 (right column) shows the style loss from the transformer network for a given source image, with respect to the Bicentennial Print and Head of a Clown paintings, as we vary $α$ from 0 to 1. As $α$ increases, the style loss with respect to Bicentennial Print increases, which explains the smooth fading out of that style’s artifact in the transformed image.
>> 이전 연구는 교묘하게 균형을 맞추는 최적화 전략이 페인트 스타일을 혼합할 기회를 제공한다고 제안했습니다 2. 이 임베딩의 유용성을 조사하기 위해 $\gamma{}$와 $\beta{}$ 값의 볼록한 조합을 시도하여 매우 뚜렷한 페인트 스타일을 혼합했습니다(그림 1b; 그림 7, 왼쪽 열). 하나의 볼록한 조합을 사용하면 한 스타일에서 다른 스타일로 부드럽게 전환할 수 있습니다. ($λ_{1}, α_{1}$) 및 ($λ_{2},α_{2}$)가 서로 다른 두 가지 스타일에 해당하는 매개 변수라고 가정합니다. 우리는 이미지를 스타일링하기 위해 $γ=α\times{}γ_{1}+(1−α)\times{}γ_{2}$와 $β=α\times{}β_{1}+(1−α)\times{}β_{2}$를 사용합니다. 볼록 조합을 사용하는 것은 임의의 수의 스타일 3으로 확장될 수 있습니다. 그림 7(오른쪽 열)은 $α$가 0에서 1까지 다양하기 때문에, 주어진 소스 이미지에 대한 트랜스포머 네트워크의 스타일 손실을 보여줍니다. $$$가 증가할수록 Bicentennial Print와 관련된 스타일 손실이 증가하는데, 이는 변환된 이미지에서 해당 스타일의 아티팩트가 부드럽게 사라지는 것을 설명합니다.
### $\mathbf{4\;DISCUSSION}$

> It seems surprising that such a small proportion of the network’s parameters can have such an impact on the overall process of style transfer. A similar intuition has been observed in auto-regressive models of images (van den Oord et al., 2016b) and audio (van den Oord et al., 2016a) where the conditioning process is mediated by adjusting the biases for subsequent samples from the model. That said, in the case of art stylization when posed as a feedforward network, it could be that the specific network architecture is unable to take full advantage of its capacity. We see evidence for this behavior in that pruning the architecture leads to qualitatively similar results. Another interpretation could be that the convolutional weights of the style transfer network encode transformations that represent “elements of style”. The scaling and shifting factors would then provide a way for each style to inhibit or enhance the expression of various elements of style to form a global identity of style. While this work does not attempt to verify this hypothesis, we think that this would constitute a very promising direction of research in understanding the computation behind style transfer networks as well as the representation of images in general.
>> 네트워크 매개 변수의 작은 부분이 스타일 전송의 전체 프로세스에 이러한 영향을 미칠 수 있다는 것은 놀라운 일입니다. 이미지(van den Oord 등, 2016b)와 오디오(van den Oord 등, 2016a)의 자동 회귀 모델에서도 유사한 직관이 관찰되었으며, 여기서 모델로부터 후속 샘플에 대한 바이어스를 조정하여 조건화 과정이 중재됩니다. 즉, 피드포워드 네트워크로 포즈될 때 예술 양식화의 경우, 특정 네트워크 아키텍처가 그 용량을 충분히 활용할 수 없을 수 있습니다. 우리는 아키텍처를 가지치기하는 것이 질적으로 유사한 결과로 이어진다는 점에서 이러한 행동에 대한 증거를 봅니다. 또 다른 해석은 스타일 전송 네트워크의 컨볼루션 가중치가 "스타일 요소"를 나타내는 변환을 인코딩한다는 것입니다. 그런 다음 스케일링과 이동 요소는 각 스타일이 다양한 스타일의 요소의 표현을 억제하거나 강화하여 스타일의 전체적인 정체성을 형성할 수 있는 방법을 제공합니다. 이 연구는 이 가설을 검증하려고 시도하지는 않지만, 스타일 전송 네트워크 뒤의 계산과 일반적인 이미지 표현을 이해하는 데 있어 매우 유망한 연구 방향을 구성할 것으로 생각합니다.

> Concurrent to this work, Gatys et al. (2016b) demonstrated exciting new methods for revising the loss to selectively adjust the spatial scale, color information and spatial localization of the artistic style information. These methods are complementary to the results in this paper and present an interesting direction for exploring how spatial and color information uniquely factor into artistic style representation.
>> 이 작업과 동시에, Gatys 등(2016b)은 예술 스타일 정보의 공간 규모, 색상 정보 및 공간 현지화를 선택적으로 조정하기 위해 손실을 수정하는 흥미로운 새로운 방법을 시연했습니다. 이러한 방법은 이 논문의 결과를 보완하며 공간 및 색 정보가 어떻게 예술적 스타일 표현에 고유하게 영향을 미치는지 탐구하기 위한 흥미로운 방향을 제시합니다.

> The question of how predictive each style image is of its corresponding style representation is also of great interest. If it is the case that the style representation can easily be predicted from a style image, one could imagine building a transformer network which skips learning an individual conditional embedding and instead learn to produce a pastiche directly from a style and a content image, much like in the original neural algorithm of artistic style, but without any optimization loop at test time.
>> 각 스타일 이미지가 해당 스타일 표현에 대해 얼마나 예측 가능한지에 대한 질문도 큰 관심사입니다. 스타일 이미지에서 스타일 표현을 쉽게 예측할 수 있는 경우라면, 개별 조건부 임베딩 학습을 생략하고 대신 스타일과 컨텐츠 이미지에서 직접 페이시체를 생산하는 것을 배우는 트랜스포머 네트워크를 구축하는 것을 상상할 수 있습니다. 예술 스타일의 원래 신경 알고리즘에서와 매우 유사하지만, 테스트 시간에 어떠한 최적화 루프도 없습니다.

> Finally, the learned style representation opens the door to generative models of style: by modeling enough paintings of a given artistic movement (e.g. impressionism), one could build a collection of style embeddings upon which a generative model could be trained. At test time, a style representation would be sampled from the generative model and used in conjunction with the style transfer network to produce a random pastiche of that artistic movement.
>> 마지막으로, 학습된 스타일 표현은 스타일의 생성 모델에 대한 문을 엽니다. 주어진 예술적 움직임(예: 인상주의)의 충분한 그림을 모델링함으로써, 생성 모델이 훈련될 수 있는 스타일 임베딩의 컬렉션을 구축할 수 있습니다. 테스트 시, 스타일 표현은 생성 모델에서 샘플링되고 스타일 전송 네트워크와 함께 사용되어 해당 예술적 움직임의 무작위 페이시체를 생산합니다.

> In summary, we demonstrated that conditional instance normalization constitutes a simple, efficient and scalable modification of style transfer networks that allows them to model multiple styles at the same time. A practical consequence of this approach is that a new painting style may be transmitted to and stored on a mobile device with a small number of parameters. We showed that despite its simplicity, the method is flexible enough to capture very different styles while having very little impact on training time and final performance of the trained network. Finally, we showed that the learned representation of style is useful in arbitrarily combining artistic styles. This work suggests the existence of a learned representation for artistic styles whose vocabulary is flexible enough to capture a diversity of the painted world.
>> 요약하면 조건부 인스턴스 정규화가 동시에 여러 스타일을 모델링할 수 있는 스타일 전송 네트워크의 간단하고 효율적이며 확장 가능한 수정을 구성한다는 것을 입증했습니다. 이 접근법의 실질적인 결과는 새로운 도장 스타일이 소수의 매개 변수를 가진 모바일 장치로 전송되고 저장될 수 있다는 것입니다. 이 방법은 단순함에도 불구하고 매우 다른 스타일을 포착할 수 있을 만큼 유연하면서도 훈련된 네트워크의 훈련 시간과 최종 성능에 거의 영향을 미치지 않는다는 것을 보여주었습니다. 마지막으로, 우리는 학습된 스타일의 표현이 예술적 스타일을 임의로 결합하는 데 유용하다는 것을 보여주었습니다. 이 작품은 회화 세계의 다양성을 포착할 수 있을 만큼 어휘가 유연한 예술적 스타일에 대한 학습된 표현의 존재를 암시합니다.

### $\mathbf{ACKNOWLEDGMENTS}$

> We would like to thank Fred Bertsch, Douglas Eck, Cinjon Resnick and the rest of the Google Magenta team for their feedback; Peyman Milanfar, Michael Elad, Feng Yang, Jon Barron, Bhavik Singh, Jennifer Daniel as well as the the Google Brain team for their crucial suggestions and advice; an anonymous reviewer for helpful suggestions about applying this model in a mobile domain. Finally, we would like to thank the Google Cultural Institute, whose curated collection of art photographs was very helpful in finding exciting style images to train on.
>> Google Brain 팀은 물론 Fred Bertsch, Douglas Eck, Cinjon Resnick 및 Google Magenta 팀의 피드백에 감사드립니다. Peyman Milanfar, Michael Elad, Feng Yang, Jon Barron, Bavik Singh, Jennifer Daniel, Google Brain 팀의 중요한 제안과 조언에 대한 익명의 검토자에게 감사드립니다. 이 모델은 모바일 도메인에 있습니다. 마지막으로, 우리는 구글 문화 연구소에 감사드리고 싶습니다. 구글 문화 연구소의 큐레이션된 예술 사진 모음은 훈련할 흥미로운 스타일의 이미지를 찾는데 매우 도움이 되었습니다.

---

### $\mathbf{References}$

<a href="#footnote_1_2" name="footnote_1_1">[1]</a> Martın Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467, 2016.

<a href="#footnote_2_2" name="footnote_2_1">[2]</a> Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. 

<a href="#footnote_3_2" name="footnote_3_1">[3]</a> IEEE Conference on, pp. 248–255. IEEE, 2009.

<a href="#footnote_4_2" name="footnote_4_1">[4]</a> Alexei A Efros and William $T$ Freeman. Image quilting for texture synthesis and transfer. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pp. 341–346. ACM, 2001.

<a href="#footnote_5_2" name="footnote_5_1">[5]</a> Alexei A Efros and Thomas K Leung. Texture synthesis by non-parametric sampling. In Computer Vision, 1999. The Proceedings of the Seventh IEEE International Conference on, volume 2, pp. 1033–1038. IEEE, 1999.

<a href="#footnote_6_2" name="footnote_6_1">[6]</a> Michael Elad and Peyman Milanfar. Style-transfer via texture-synthesis. arXiv preprint arXiv:1609.03057, 2016. 

<a href="#footnote_7_2" name="footnote_7_1">[7]</a> Oriel Frigo, Neus Sabater, Julie Delon, and Pierre Hellier. Split and match: Example-based adaptive patch sampling for unsupervised style transfer. 2016.

<a href="#footnote_8_2" name="footnote_8_1">[8]</a> Leon Gatys, Alexander S Ecker, and Matthias Bethge. Texture synthesis using convolutional neural networks. In Advances in Neural Information Processing Systems, pp. 262–270, 2015a.

<a href="#footnote_9_2" name="footnote_9_1">[9]</a> Leon A Gatys, Alexander S Ecker, and Matthias Bethge. A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576, 2015b.

<a href="#footnote_10_2" name="footnote_10_1">[10]</a> Leon A Gatys, Matthias Bethge, Aaron Hertzmann, and Eli Shechtman. Preserving color in neural artistic style transfer. arXiv preprint arXiv:1606.05897, 2016a.

<a href="#footnote_11_2" name="footnote_11_1">[11]</a> Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, Aaron Hertzmann, and Eli Shechtman. Controlling perceptual factors in neural style transfer. CoRR, abs/1611.07865, 2016b. URL http://arxiv.org/abs/1611.07865.

<a href="#footnote_12_2" name="footnote_12_1">[12]</a> Aaron Hertzmann, Charles E Jacobs, Nuria Oliver, Brian Curless, and David H Salesin. Image analogies. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pp. 327–340. ACM, 2001.

<a href="#footnote_13_2" name="footnote_13_1">[13]</a> Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. arXiv preprint arXiv:1603.08155, 2016.

<a href="#footnote_14_2" name="footnote_14_1">[14]</a> Bela Julesz. Visual pattern discrimination. IRE Trans. Info Theory, 8:84–92, 1962.

<a href="#footnote_15_2" name="footnote_15_1">[15]</a> Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

<a href="#footnote_16_2" name="footnote_16_1">[16]</a> Vivek Kwatra, Irfan Essa, Aaron Bobick, and Nipun Kwatra. Texture optimization for examplebased synthesis. ACM Transactions on Graphics (ToG), 24(3):795–802, 2005.

<a href="#footnote_17_2" name="footnote_17_1">[17]</a> Chuan Li and Michael Wand. Precomputed real-time texture synthesis with markovian generative adversarial networks. ECCV, 2016. URL http://arxiv.org/abs/1604.04382.

<a href="#footnote_18_2" name="footnote_18_1">[18]</a> Lin Liang, Ce Liu, Ying-Qing Xu, Baining Guo, and Heung-Yeung Shum. Real-time texture synthesis by patch-based sampling. ACM Transactions on Graphics (ToG), 20(3):127–150, 2001.

<a href="#footnote_19_2" name="footnote_19_1">[19]</a> Augustus Odena, Christopher Olah, and Vincent Dumoulin. Avoiding checkerboard artifacts in neural networks. Distill, 2016.

<a href="#footnote_20_2" name="footnote_20_1">[20]</a> Javier Portilla and Eero Simoncelli. A parametric texture model based on joint statistics of complex wavelet coefficients. International Journal of Computer Vision, 40:49–71, 1999.

<a href="#footnote_21_2" name="footnote_21_1">[21]</a> Eero Simoncelli and Bruno Olshausen. Natural image statistics and neural representation. Annual Review of Neuroscience, 24:1193–1216, 2001.

<a href="#footnote_22_2" name="footnote_22_1">[22]</a> Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

<a href="#footnote_23_2" name="footnote_23_1">[23]</a> Dmitry Ulyanov, Vadim Lebedev, Andrea Vedaldi, and Victor Lempitsky. Texture networks: Feedforward synthesis of textures and stylized images. arXiv preprint arXiv:1603.03417, 2016a.

<a href="#footnote_24_2" name="footnote_24_1">[24]</a> Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016b.

<a href="#footnote_25_2" name="footnote_25_1">[25]</a> Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew W. Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. CoRR, abs/1609.03499, 2016a. URL http://arxiv.org/abs/1609.03499.

<a href="#footnote_26_2" name="footnote_26_1">[26]</a> Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with pixelcnn decoders. CoRR, abs/1606.05328, 2016b. URL http://arxiv.org/abs/1606.05328.

<a href="#footnote_27_2" name="footnote_27_1">[27]</a> Li-Yi Wei and Marc Levoy. Fast texture synthesis using tree-structured vector quantization. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pp.479–488. ACM Press/Addison-Wesley Publishing Co., 2000.

<a href="#footnote_28_2" name="footnote_28_1">[28]</a> Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European Conference on Computer Vision, pp. 818–833. Springer, 2014.