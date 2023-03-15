---
layout: post 
title: "(GAN)DCGAN Translation"
categories: [1. Computer Engineering]
tags: [1.7. Paper Review, 1.2.2.5. GAN, 1.2.2.2. CNN]
---

### [GAN Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

$UNSUPERVISED\;REPRESENTATION\;LEARNING$  
$WITH\;DEEP\;CONVOLUTIONAL$  
$GENERATIVE\;ADVERSARIAL\;NETWORKS$  

$ABSTRACT$
> In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications.
>> 최근 몇년간, 합성곱망(CNNs)을 사용한 지도학습은 computer vision applications에서 크게 채택되었다.

> Comparatively, unsupervised learning with CNNs has received less attention.
>> 비교적이게도, CNNs를 사용한 비지도학습은 더 적은 관심을 받았다.

> In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning
>> 이번 일로 우리는 지도 학습과 비지도 학습에 대한 합성곱의 성공의 차이를 해소하는데 도움이 되기를 바란다.

> We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning.
>> 우리는 깊은 합성곱 생성 적대적 네트워크(DCGANs)라고 불리는 CNNs의 계층을 소개한다. 이는 특정 아키텍처의 제약 조건을 가지며, 비지도 학습의 강력한 후보자임을 입증한다.

> Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator.
>> 다양한 이미지 데이터세트에 대한 학습을 통해, 우리의 심층 합성곱 적대적인 두쌍은 generator와 discriminator가 장면의 객체 요소에서 표현 계층 구조를  학습한다는 것에 설득력있는 증거를 보여준다.

> Additionally, we use the learned features for novel tasks - demonstrating their applicability as general images representations.
>> 또한, 우리는 학습된 특징을 새로운 작업에 사용하여 일반적인 이미지에 적용가능성을 입증한다.

$1.\;INTRODUCTION$

> Learning reusable feature representations from large unlabeled datasets has been an area of active research.
>> 라벨링되지 않은 큰 데이터 셋으로부터 재사용 가능한 특징 표현을 학습하는 것은 활발한 연구 분야가 되었다.

> In the context of computer vision, one can leverage the practically unlimited amount of unlabeled images and videos to learn good intermediate representations, which can then be used on a variety of supervised learning tasks such as image classification.
>> 컴퓨터 비전의 맥락에서, 라벨링이 되지않은 이미지와 비디오를 실질적으로 무제한 활용하여 좋은 중간 표현을 학습할 수 있으며, 이는 이미지 분류와 같은 다양한 지도 학습에 사용될 수 있다.

> We propose that one way to build good image representations is by training Generative Adversarial Networks(GANs) (Goodfellowet al., 2014), and later reusing parts of the generator and discriminator networks as feature extractors for supervised tasks.
>> 우리는 좋은 이미지표현을 만들어내는 한가지 방법이 GAN을 통해 훈련시키는 것이라고 제안한다, 그 이후 generator와 discriminator 네트워크의 일부를 supervised task을 위한 특징 추출기로 재사용한다.

> GANs provide an attractive alternative to maximum likelihood techniques.
>> GAN은 MLE(Maximum Likelihood) 기술에 대한 매력적인 대안을 제공한다.

> One can additionally argue that their learning process and the lack of a heuristic cost function (such as pixel-wise independent mean-square error) are attractive to representation learning.
>> 학습 프로세스와 heuristic 비용함수(pixel-wise independent MSE 등)의 부재가 표현 학습에 매력적이라고 일반적으로 주장할 수 있다.

> GANs have been known to be unstable to train, often resulting in generators that produce nonsensical outpus.
>> GAN은 훈련이 불안정하다고 알려져 있고, generator는 터무니없는 출력을 자주 보여준다.

> There has been very limited published research in trying to understand and visualize what GANs learn, and the intermediate representations of multi-layer GANs.
>> GAN이 학습하는 내용과 다중 계층 GAN의 중간 표현을 이해하고 시각화하기 위해 발표된 연구는 매우 제한적이다.

> In this paper, we make the following contributions
>> 이 논문에서 우리는 다음과 같은 기여를 한다.

* > We propose and evaluate a set of constraints on the architectural topology of Convolutional GANs that make them stable to train in most settings. We name this class of architectures Deep Convolutional GANs(DCGAN)
    >>  우리는 합성곱 GAN의 architectural topology에 대한 제약을 제안하고 평가하여 대부분의 환경에서 안정적으로 훈련한다. 우리는 이 class의 아키텍처를 DCGAN이라고 한다.

* > We used the trained discriminators for image classification tasks, showing competitive performance with other unsupervised algorithms.
    >> 우리는 훈련된 discriminator를 이미지 분류에 사용하여 다른 비지도 학습 알고리즘과 경쟁적인 성능을 보여주었다.

* > We visualize the filters learnt by GANs and empirically show that specific filters have learned to draw specific objects.
    >> 우리는 GAN에 의해 학습된 필터를 시각화하고 특정 필터가 특정 객체를 그리는 방법을 학습했음을 경험적으로 보여준다.

* > We show that the generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples.
    >> 우리는 생성자가 생성된 샘플의 많은 의미적 특성을 쉽게 조작할 수 있는 흥미로운 벡터 산술 특성을 가지고 있음을 보여준다.

$2.\;RELATED\;WORK$

$2.1\;REPRESENTATION\;LEARNING\;FROM\;UNLABELED\;DATA$

> Unsupervised representation learning is a fairly well studied problem in general computer vision research, as well as in the context of images.
>> 비지도 묘사 학습은 일반적인 컴퓨터 비전 연구 문제뿐만 아니라 이미지의 맥락에서 꽤 잘 학습된다.

> A classic approach to unsupervised representation learning is to do clustring on the data (for example using K-means), and leverage the clusters for improved classification scores.
>> 보편적인 비지도 표현 학습의 접근 방법은 데이터를 군집화(예를 들어 k-means를 사용하는 등)하고 영향력 있는 항상된 분류 점수를 개선하기 위해 군집을 활용하는 것이다.

> In the context of images, one can do hierarchical clustering of image patches to learn powerful image representations.
>> 이미지의 문맥에서, 이는 강력한 이미지 표현을 학습하기 위해 이미지 패치의 계층적 군집화를 할 수 있다.

> Another popular method is to train auto-encoders separating the what and where components of the code, ladder structures that encode an image into a compact code, and decode the code to reconstruct the image as accurately as possible.
>> 다른 인기있는 방법은 코드의 what과 where 구성 요소를 분리하는 auto-encoder, 이미지를 컴팩트 코드로 인코딩하는 래더 구조, 코드를 디코딩하여 가능한 정확하게 이미지를 재구성하는 것이다.

> These methods have also been shown to learn good feature representations from image pixels. Deep belief networks (Lee et al., 2009) have also been shown to work well in learning hierarchical representations.
>> 이러한 방법은 또한 이미지 픽셀에서 좋은 특징 표현을 학습하는 것으로 나타났다. 심층 신념 네트워크는 계층적 표현을 학습하는 데도 잘 작동하는 것으로 나타났다.

$2.2\;GENERATING\;NATURAL\;IMAGES$

> Generative image models are well studied and fall into two categories: parametric and nonparametric. The non-parametric models often do matching from a database of existing images, often matching patches of images, and have been used in texture synthesis (Efros et al., 1999), super-resolution (Freeman et al., 2002) and in-painting (Hays & Efros, 2007). Parametric models for generating images has been explored extensively (for example on MNIST digits or for texture synthesis (Portilla & Simoncelli, 2000)). However, generating natural images of the real world have had not much success until recently. A variational sampling approach to generating images (Kingma & Welling, 2013) has had some success, but the samples often suffer from being blurry. Another approach generates images using an iterative forward diffusion process (Sohl-Dickstein et al., 2015). Generative Adversarial Networks (Goodfellow et al., 2014) generated images suffering from being noisy and incomprehensible. A laplacian pyramid extension to this approach (Denton et al., 2015) showed higher quality images, but they still suffered from the objects looking wobbly because of noise introduced in chaining multiple models. A recurrent network approach (Gregor et al., 2015) and a deconvolution network approach (Dosovitskiy et al., 2014) have also recently had some success with generating natural images. However, they have not leveraged the generators for supervised tasks.
>> 생성 이미지 모델은 잘 연구되고 있으며 파라메트릭과 비모수라는 두 가지 범주로 나뉜다. 비모수 모델은 종종 기존 이미지의 데이터베이스에서 매칭 작업을 수행하며, 종종 이미지의 패치를 매칭하며 텍스처 합성(Efros et al., 1999), 초해상도(Freeman et al., 2002) 및 인페인팅(Hays & Efros, 2007)에 사용되었다. 이미지 생성을 위한 파라메트릭 모델은 광범위하게 연구되었다(예: MNIST 숫자 또는 텍스처 합성을 위해). (Portilla & Simoncelli, 2000 하지만, 현실 세계의 자연스러운 이미지를 만들어내는 것은 최근까지 큰 성공을 거두지 못했다. 이미지 생성에 대한 변형 샘플링 접근 방식(Kingma & Welling, 2013)은 어느 정도 성공을 거두었지만 샘플이 흐릿한 경우가 많다. 또 다른 접근 방식은 반복적인 전방 확산 과정을 사용하여 이미지를 생성한다(Sohl-Dickstein et al., 2015). 생성적 적대 네트워크(Generative Adversarial Networks, Goodfellow et al., 2014)는 소음이 심하고 이해할 수 없는 것으로 고통 받는 이미지를 생성했다. 이 접근법에 대한 라플라시안 피라미드 확장(Denton et al., 2015)은 더 높은 품질의 이미지를 보여주었지만, 여러 모델을 연결하는 데 유입된 노이즈 때문에 물체가 흔들리는 것으로 인해 여전히 어려움을 겪었다. 순환 네트워크 접근 방식(Gregor et al., 2015)과 디콘볼루션 네트워크 접근 방식(Dosovitskyy et al., 2014)도 최근 자연 이미지를 생성하는 데 어느 정도 성공했다. 그러나 그들은 supervised tasks에 Generator를 활용하지 않았다.

$2.3\;VISUALIZING\;THE\;INTERNALS\;OF\;CNNs$

> One constant criticism of using neural networks has been that they are black-box methods, with little understanding of what the networks do in the form of a simple human-consumable algorithm. In the context of CNNs, Zeiler et. al. (Zeiler & Fergus, 2014) showed that by using deconvolutions and filtering the maximal activations, one can find the approximate purpose of each convolution filter in the network. Similarly, using a gradient descent on the inputs lets us inspect the ideal image that activates certain subsets of filters (Mordvintsev et al.).
>> 신경망을 사용하는 것에 대한 지속적인 비판 중 하나는 네트워크가 사람이 소비하는 간단한 알고리듬의 형태로 수행하는 것에 대한 이해가 거의 없는 블랙박스 방식이라는 것이다. CNN의 맥락에서, Zeiler 등(Zeiler & Fergus, 2014)은 디콘볼루션과 최대 활성화를 필터링함으로써 네트워크에서 각 convolution 필터의 대략적인 목적을 찾을 수 있음을 보여주었다. 마찬가지로, 입력에 그레이디언트 강하를 사용하면 특정 필터 하위 세트를 활성화하는 이상적인 이미지를 검사할 수 있다(Mordvintseve 등).

$3\;APPROACH\;AND\;MODEL\;ARCHITECTURE$

> Historical attempts to scale up GANs using CNNs to model images have been unsuccessful. This motivated the authors of LAPGAN (Denton et al., 2015) to develop an alternative approach to iteratively upscale low resolution generated images which can be modeled more reliably. We also encountered difficulties attempting to scale GANs using CNN architectures commonly used in the supervised literature. However, after extensive model exploration we identified a family of architectures that resulted in stable training across a range of datasets and allowed for training higher resolution and deeper generative models. Core to our approach is adopting and modifying three recently demonstrated changes to CNN architectures. The first is the all convolutional net (Springenberg et al., 2014) which replaces deterministic spatial pooling functions (such as maxpooling) with strided convolutions, allowing the network to learn its own spatial downsampling. We use this approach in our generator, allowing it to learn its own spatial upsampling, and discriminator. Second is the trend towards eliminating fully connected layers on top of convolutional features. The strongest example of this is global average pooling which has been utilized in state of the art image classification models (Mordvintsev et al.). We found global average pooling increased model stability but hurt convergence speed. A middle ground of directly connecting the highest convolutional features to the input and output respectively of the generator and discriminator worked well. The first layer of the GAN, which takes a uniform noise distribution Z as input, could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack. For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output. See Fig. 1 for a visualization of an example model architecture. Third is Batch Normalization (Ioffe & Szegedy, 2015) which stabilizes learning by normalizing the input to each unit to have zero mean and unit variance. This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models. This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples to a single point which is a common failure mode observed in GANs. Directly applying batchnorm to all layers however, resulted in sample oscillation and model instability. This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer. The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output layer which uses the Tanh function. We observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution. Within the discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work well, especially for higher resolution modeling. This is in contrast to the original GAN paper, which used the maxout activation (Goodfellow et al., 2013).
>> CNN을 사용하여 이미지를 모델링하는 GAN을 확장하려는 과거의 시도는 성공하지 못했다. 이는 LAPGAN(Denton et al., 2015)의 저자들이 보다 안정적으로 모델링할 수 있는 저해상도 생성 이미지를 반복적으로 상향 조정하기 위한 대안적 접근 방식을 개발하도록 동기를 부여했다. 우리는 또한 감독 문헌에서 일반적으로 사용되는 CNN 아키텍처를 사용하여 GAN을 확장하려고 시도하는 데 어려움을 겪었다. 그러나 광범위한 모델 탐색 후 다양한 데이터 세트에 걸쳐 안정적인 학습을 제공하고 고해상도 및 심층 생성 모델을 훈련할 수 있는 아키텍처 제품군을 식별했다. 우리의 접근 방식의 핵심은 CNN 아키텍처에 대해 최근에 입증된 세 가지 변경 사항을 채택하고 수정하는 것이다. 첫 번째는 결정론적 공간 pooling 기능(maxpooling 등)을 스트라이드 convolution으로 대체하여 네트워크가 자체 공간 다운샘플링을 학습할 수 있는 all convolutional net(Springenberg et al., 2014)이다. 우리는 Generator에서 이 접근 방식을 사용하여 자체 공간 업샘플링 및 discriminator를 학습할 수 있다. 두 번째는 convolution 특징 위에 fully connected된 레이어를 제거하는 추세이다. 이것의 가장 강력한 예는 최신 이미지 분류 모델(Mordvintsev 등)에서 활용된 글로벌 평균 pooling이다. 우리는 글로벌 평균 pooling이 모델 안정성을 증가시켰지만 수렴 속도를 손상시켰다는 것을 발견했다. Generator와 discriminator의 입력과 출력 각각에 가장 높은 convolution 특징을 직접 연결하는 중간 지대가 잘 작동했다. 균일한 노이즈 분포 Z를 입력으로 취하는 GAN의 첫 번째 레이어는 행렬 곱셈일 뿐이므로 fully connected되었다고 할 수 있지만, 결과는 4차원 텐서로 재구성되어 convolution 스택의 시작으로 사용된다. discriminator의 경우, 마지막 convolution 레이어는 평평해진 다음 단일 시그모이드 출력으로 공급된다. 예시적인 모델 아키텍처의 시각화는 그림 1을 참조한다. 세 번째는 배치 정규화(Ioffe & Szegdy, 2015)로, 각 단위에 대한 입력을 평균 및 단위 분산이 0이 되도록 정규화하여 학습을 안정화한다. 이는 열악한 초기화로 인해 발생하는 학습 문제를 해결하는 데 도움이 되며 더 깊은 모델에서 그레이디언트 흐름을 돕는다. 이는 딥 제너레이터가 학습을 시작하여 GAN에서 관찰되는 일반적인 고장 모드인 단일 점으로 모든 샘플을 접는 것을 방지하는 데 중요한 것으로 입증되었다. 그러나 배치 규범을 모든 레이어에 직접 적용하면 샘플 진동과 모델이 불안정해진다. 이는 generator 출력층과 discriminator 입력층에 배치 노름을 적용하지 않음으로써 회피되었다. ReLU 활성화(Nair & Hinton, 2010)는 탄 함수를 사용하는 출력 계층을 제외하고 Generator에 사용된다. 우리는 제한된 활성화를 사용하면 모델이 훈련 분포의 색 공간을 포화시키고 커버하는 것을 더 빨리 배울 수 있다는 것을 관찰했다. discriminator 내에서 특히 고해상도 모델링을 위해 누출 정류 활성화(Maas et al., 2013)(Xu et al., 2015)가 잘 작동한다는 것을 발견했다. 이는 최대치 활성화를 사용한 원래의 GAN 논문과는 대조적이다(Goodfellow et al., 2013).

> Architecture guidelines for stable Deep Convolutional GANs
>> 안정적인 Deep Convolutional GAN을 위한 아키텍처 지침

* > Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
    >> pooling 레이어를 스트라이드 convolution(discriminator) 및 분수 스트라이드 convolution(generator)으로 대체합니다.
* > Use batchnorm in both the generator and the discriminator.
    >> generator와 discriminator 모두에서 batch norm을 사용합니다.
* > Remove fully connected hidden layers for deeper architectures.
    >> 심층적인 아키텍처를 위해 fully connected된 숨겨진 레이어를 제거합니다.
* > Use ReLU activation in generator for all layers except for the output, which uses Tanh.
    >> Tanh를 사용하는 출력을 제외한 모든 레이어에 대해 제너레이터에서 ReLU 활성화를 사용합니다.
* > Use LeakyReLU activation in the discriminator for all layers.
    >> LeakyRe 사용모든 레이어에 대한 discriminator에서 LU 활성화.

$4\;DETAILS\;OF\;ADVERSARIAL\;TRAINING$

> We trained DCGANs on three datasets, Large-scale Scene Understanding (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset. Details on the usage of each of these datasets are given below. No pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1]. All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128. All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02. In the LeakyReLU, the slope of the leak was set to 0.2 in all models. While previous GAN work has used momentum to accelerate training, we used the Adam optimizer (Kingma & Ba, 2014) with tuned hyperparameters. We found the suggested learning rate of 0.001,
to be too high, using 0.0002 instead. Additionally, we found leaving the momentum term β1 at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.
>> 우리는 Large-scale Scene Understanding(LSUN) (Yu et al., 2015), Imagenet-1k 및 새로 조립된 얼굴 데이터 세트의 세 가지 데이터 세트에 대해 DCGAN을 학습했다. 이러한 각 데이터 세트의 사용에 대한 자세한 내용은 아래에 나와 있습니다. tanh activation function의 범위[-1, 1]로 스케일링하는 것 외에는 학습 영상에 전처리가 적용되지 않았습니다. 모든 모델은 128의 미니 배치 크기를 가진 mini-batch stochastic gradient descent(SGD)로 훈련되었다. 모든 가중치는 표준 편차가 0.02인 0 중심 정규 분포에서 초기화되었습니다. LeakyReLU는 모든 모델에서 slope of the leak
가 0.2로 설정되었습니다. 이전 GAN 작업은 모멘텀을 사용하여 학습을 가속화했지만, 우리는 튜닝된 하이퍼 매개 변수를 가진 Adam optimizer(Kingma & Ba, 2014)를 사용했다. 우리는 0.001의 권장 학습률을 발견했다.
0.0002를 대신 사용하여 너무 높습니다. 또한 운동량 항 α1을 제안된 값 0.9로 남겨두면 훈련 진동과 불안정성을 초래하고 0.5로 줄이면 훈련 안정화에 도움이 된다는 것을 발견했다.

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-1.JPG)

> Figure 1: DCGAN generator used for LSUN scene modeling. A 100 dimensional uniform distribution Z is projected to a small spatial extent convolutional representation with many feature maps. A series of four fractionally-strided convolutions (in some recent papers, these are wrongly called deconvolutions) then convert this high level representation into a 64 × 64 pixel image. Notably, no fully connected or pooling layers are used.
>> 그림 1: LSUN 장면 모델링에 사용되는 DCGAN 제너레이터 100차원 균일 분포 Z는 많은 특징 맵을 가진 작은 공간 범위 convolution 표현에 투영된다. 4개의 부분 스트라이드 convolution 시리즈(최근 논문에서, 이것들은 디콘볼루션이라고 잘못 불린다)는 이 높은 수준의 표현을 64 × 64 픽셀 이미지로 변환한다. 특히 fully connected된 계층이나 pooling 계층은 사용되지 않습니다.

$4.1\;LSUN$

> As visual quality of samples from generative image models has improved, concerns of over-fitting and memorization of training samples have risen. To demonstrate how our model scales with more data and higher resolution generation, we train a model on the LSUN bedrooms dataset containing a little over 3 million training examples. Recent analysis has shown that there is a direct link between how fast models learn and their eneralization performance (Hardt et al., 2015). We show samples from one epoch of training (Fig.2), mimicking online learning, in addition to samples after convergence (Fig.3), as an opportunity to demonstrate that our model is not producing high quality samples via simply overfitting/memorizing training examples. No data augmentation was applied to the images.
>> 생성 이미지 모델에서 샘플의 시각적 품질이 향상됨에 따라 훈련 샘플의 과적합 및 memorization 문제가 대두되었다. 우리의 모델이 더 많은 데이터와 더 높은 해상도 생성으로 어떻게 확장되는지 보여주기 위해, 우리는 300만 개가 조금 넘는 훈련 예를 포함하는 LSUN 침실 데이터 세트에 대한 모델을 훈련시킨다. 최근 분석에 따르면 모델의 학습 속도와 일반화 성능 사이에 직접적인 연관성이 있다(Hardt et al., 2015). 우리는 융합 후 샘플(그림 3) 외에 온라인 학습을 모방한 한 학습 시대(그림 2)의 샘플을 보여주는데, 이는 우리 모델이 단순히 과적합/기억 훈련 예를 통해 고품질 샘플을 생산하지 않는다는 것을 보여주는 기회이다. 이미지에 데이터 확장이 적용되지 않았습니다.

$4.1.1\;DEDUPLICATION$

> To further decrease the likelihood of the generator memorizing input examples (Fig.2) we perform a simple image de-duplication process. We fit a 3072-128-3072 de-noising dropout regularized RELU autoencoder on 32x32 downsampled center-crops of training examples. The resulting code layer activations are then binarized via thresholding the ReLU activation which has been shown to be an effective information preserving technique (Srivastava et al., 2014) and provides a convenient form of semantic-hashing, allowing for linear time de-duplication . Visual inspection of hash collisions showed high precision with an estimated false positive rate of less than 1 in 100. Additionally, the technique detected and removed approximately 275,000 near duplicates, suggesting a high recall.
>> 입력 예제를 기억하는 제너레이터의 가능성을 더욱 낮추기 위해(그림 2) 간단한 이미지 중복 제거 프로세스를 수행한다. 우리는 32x32 다운샘플링된 훈련 예제에 3072-128-3072 디노이즈 드롭아웃 정규화 RELU 자동 인코더를 장착한다. 그 결과 코드 계층 활성화는 효과적인 정보 보존 기법(Srivastava et al., 2014)으로 입증되고 선형 시간 중복 제거를 허용하는 편리한 형태의 의미 해싱을 제공하는 ReLU 활성화 임계값을 통해 이진화된다. 해시 충돌의 시각적 검사는 높은 정밀도로 나타났다. 100분의 1 미만의 추정 거짓 양성률 또한, 이 기술은 약 275,000개의 중복을 탐지하고 제거했으며, 이는 높은 회수율을 시사한다.

$4.2\;FACES$

> We scraped images containing human faces from random web image queries of peoples names. The people names were acquired from dbpedia, with a criterion that they were born in the modern era. This dataset has 3M images from 10K people. We run an OpenCV face detector on these images, keeping the detections that are sufficiently high resolution, which gives us approximately 350,000 face boxes. We use these face boxes for training. No data augmentation was applied to the images.
>> 우리는 사람 이름의 무작위 웹 이미지 쿼리에서 사람의 얼굴이 포함된 이미지를 긁어냈다. 사람들의 이름은 근대에 태어났다는 기준과 함께 dbpedia에서 따왔다. 이 데이터 세트는 1만 명의 사람으로부터 3백만 개의 이미지를 가지고 있다. 우리는 이 이미지들에 OpenCV 얼굴 탐지기를 실행하여 충분히 높은 해상도의 탐지기를 유지하며 약 350,000개의 얼굴 상자를 제공합니다. 우리는 이 페이스 박스를 훈련에 사용합니다. 이미지에 데이터 확장이 적용되지 않았습니다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-2.JPG)

> Figure 2: Generated bedrooms after one training pass through the dataset. Theoretically, the model could learn to memorize training examples, but this is experimentally unlikely as we train with a small learning rate and minibatch SGD. We are aware of no prior empirical evidence demonstrating memorization with SGD and a small learning rate.
>> 그림 2: 데이터 세트를 한번 통과한 후 생성된 침실들. 이론적으로, 모델은 훈련 예시를 기억하는 법을 배울 수 있지만, 우리가 적은 학습률과 미니 배치 SGD로 훈련하기 때문에 이것은 실험적으로 가능성이 낮다. 우리는 SGD와 적은 학습률로 memorization를 입증하는 이전의 경험적 증거가 없다는 것을 알고 있다.

![Figure 3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-3.JPG)

> Figure 3: Generated bedrooms after five epochs of training. There appears to be evidence of visual under-fitting via repeated noise textures across multiple samples such as the base boards of some of the beds.
>> 그림 3: 5epoch 동안 학습을 받은 후 생성된 침실. 일부 침대의 베이스보드와 같은 여러 샘플에 걸쳐 반복적인 소음 질감을 통해 시각적 언더피팅의 증거가 있는 것으로 보인다.

$4.3\;IMAGENET-1K$

> We use Imagenet-1k (Deng et al., 2009) as a source of natural images for unsupervised training. We train on 32 × 32 min-resized center crops. No data augmentation was applied to the images.
>> 우리는 감독되지 않은 훈련을 위한 자연 이미지의 소스로 Imagenet-1k(Deng et al., 2009)를 사용한다. 우리는 32×32분 크기의 중앙 작물에 대해 훈련한다. 이미지에 데이터 확장이 적용되지 않았습니다.

$5\;EMPIRICAL\;VALIDATION\;OF\;DCGANs\;CAPABILITIES$

$5.1\;CLASSIFYING\;CIFAR-10\;USING\;GANs\;AS\;A\;FEATURE\;EXTRACTOR$

> One common technique for evaluating the quality of unsupervised representation learning algorithms is to apply them as a feature extractor on supervised datasets and evaluate the performance of linear models fitted on top of these features.
>> 비지도 표현 학습 알고리듬의 품질을 평가하기 위한 한 가지 일반적인 기술은 이들을 지도 데이터 세트에 특징 추출기로 적용하고 이러한 특징 위에 적합한 선형 모델의 성능을 평가하는 것이다.

> On the CIFAR-10 dataset, a very strong baseline performance has been demonstrated from a well tuned single layer feature extraction pipeline utilizing K-means as a feature learning algorithm. When using a very large amount of feature maps (4800) this technique achieves 80.6% accuracy. An unsupervised multi-layered extension of the base algorithm reaches 82.0% accuracy (Coates & Ng, 2011). To evaluate the quality of the representations learned by DCGANs for supervised tasks, we train on Imagenet-1k and then use the discriminator’s convolutional features from all layers, maxpooling each layers representation to produce a 4 × 4 spatial grid. These features are then flattened and concatenated to form a 28672 dimensional vector and a regularized linear L2-SVM classifier is trained on top of them. This achieves 82.8% accuracy, out performing all K-means based approaches. Notably, the discriminator has many less feature maps (512 in the highest layer) compared to K-means based techniques, but does result in a larger total feature vector size due to the many layers of 4 × 4 spatial locations. The performance of DCGANs is still less than that of Exemplar CNNs (Dosovitskiy et al., 2015), a technique which trains normal discriminative CNNs in an unsupervised fashion to differentiate between specifically chosen, aggressively augmented, exemplar samples from the source dataset. Further improvements could be made by finetuning the discriminator’s representations, but we leave this for future work. Additionally, since our DCGAN was never trained on CIFAR-10 this experiment also demonstrates the domain robustness of the learned features.
>> CIFAR-10 데이터 세트에서, K-평균을 특징 학습 알고리듬으로 활용하는 잘 조정된 단일 계층 특징 추출 파이프라인에서 매우 강력한 기준 성능이 입증되었다. 매우 많은 양의 피처 맵(4800)을 사용할 때 이 기술은 80.6%의 정확도를 달성한다. 기본 알고리듬의 비지도 다층 확장은 82.0% 정확도에 도달한다(Coates & Ng, 2011). supervised tasks에 대해 DCGAN이 학습한 표현의 품질을 평가하기 위해 Imagenet-1k에서 학습한 다음 모든 레이어에서 discriminator의 convolution 기능을 사용하여 각 레이어 표현을 최대 pooling하여 4×4 공간 그리드를 생성한다. 그런 다음 이러한 기능은 평탄화되고 연결되어 28672차원 벡터를 형성하고 정규화된 선형 L2-SVM 분류기가 그 위에 훈련된다. 이는 82.8%의 정확도를 달성하여 모든 K-평균 기반 접근 방식을 능가한다. 특히 discriminator는 K-평균 기반 기법에 비해 피처 맵이 훨씬 적지만(가장 높은 레이어에서 512개) 4×4 공간 위치의 많은 레이어로 인해 총 피처 벡터 크기가 더 커진다. DCGAN의 성능은 소스 데이터 세트에서 구체적으로 선택되고 공격적으로 증강된 예시 샘플을 구별하기 위해 정상적인 차별적 CNN을 비지도 방식으로 훈련하는 기술인 Examplear CNNs(Dosovitskyy et al., 2015)의 성능보다 여전히 낮다. 판별자의 표현을 미세하게 조정하면 더 많은 개선이 이루어질 수 있지만, 우리는 이것을 향후 작업을 위해 남겨둔다. 또한 DCGAN은 CIFAR-10에 대해 훈련되지 않았기 때문에 이 실험은 학습된 기능의 도메인 견고성도 보여준다.

$5.2\;CLASSIFYING\;SVHN\;DIGITS\;USING\;GANS\;AS\;A\;FEATURE\;EXTRACTOR$

> On the StreetView House Numbers dataset (SVHN)(Netzer et al., 2011), we use the features of the discriminator of a DCGAN for supervised purposes when labeled data is scarce. Following similar dataset preparation rules as in the CIFAR-10 experiments, we split off a validation set of 10,000 examples from the non-extra set and use it for all hyperparameter and model selection. 1000 uniformly class distributed training examples are randomly selected and used to train a regularized linear L2-SVM classifier on top of the same feature extraction pipeline used for CIFAR-10. This achieves state of the art (for classification using 1000 labels) at 22.48% test error, improving upon another modifcation of CNNs designed to leverage unlabled data (Zhao et al., 2015). Additionally, we validate that the CNN architecture used in DCGAN is not the key contributing factor of the model’s performance by training a purely supervised CNN with the same architecture on the same data and optimizing this model via random search over 64 hyperparameter trials (Bergstra & Bengio, 2012). It achieves a signficantly higher 28.87% validation error.
>> StreetView House Numbers 데이터 세트(SVHN)(Netzer et al., 2011)에서 레이블링된 데이터가 부족할 때 감독 목적으로 DCGAN의 discriminator 기능을 사용한다. CIFAR-10 실험에서와 유사한 데이터 세트 준비 규칙을 따라 추가되지 않은 세트에서 10,000개의 예시로 구성된 검증 세트를 분리하여 모든 하이퍼 파라미터 및 모델 선택에 사용한다. CIFAR-10에 사용된 동일한 기능 추출 파이프라인 위에 정규화된 선형 L2-SVM 분류기를 훈련하기 위해 1000개의 균일 클래스 분산 훈련 예를 무작위로 선택하여 사용한다. 이는 22.48%의 테스트 오류에서 최첨단(1000개의 레이블을 사용한 분류의 경우)을 달성하여 비표시 데이터를 활용하도록 설계된 CNN의 또 다른 수정에 따라 개선된다(Zhao et al., 2015). 또한, 우리는 DCGAN에 사용된 CNN 아키텍처가 동일한 데이터에 대해 동일한 아키텍처를 가진 순수하게 감독된 CNN을 훈련시키고 64번의 초 매개 변수 시험에 대한 무작위 검색을 통해 이 모델을 최적화함으로써 모델 성능에 핵심적인 기여 요소가 아님을 검증한다(Bergstra & Bengio, 2012). 이는 28.87%의 유효화 오류를 크게 높인다.

$6\;INVESTIGATING\;AND\;VISUALIZING\;THE\;INTERNALS\;OF\;THE\;NETWORKS$

> We investigate the trained generators and discriminators in a variety of ways. We do not do any kind of nearest neighbor search on the training set. Nearest neighbors in pixel or feature space are trivially fooled (Theis et al., 2015) by small image transforms. We also do not use log-likelihood metrics to quantitatively assess the model, as it is a poor (Theis et al., 2015) metric.
>> 우리는 훈련된 generator와 discriminator를 다양한 방법으로 조사한다. 우리는 훈련 세트에서 어떤 종류의 가장 가까운 이웃 탐색도 하지 않는다. 픽셀 또는 피처 공간에서 가장 가까운 이웃은 작은 이미지 변환에 의해 사소한 것으로 속는다(This et al., 2015). 또한 로그 우도 메트릭을 사용하여 모델을 정량적으로 평가하지 않는다(This et al., 2015).

$6.1\;WALKING\;IN\;THE\;LATENT\;SPACE$

> The first experiment we did was to understand the landscape of the latent space. Walking on the manifold that is learnt can usually tell us about signs of memorization (if there are sharp transitions) and about the way in which the space is hierarchically collapsed. If walking in this latent space results in semantic changes to the image generations (such as objects being added and removed), we can reason that the model has learned relevant and interesting representations. The results are shown in Fig.4.
>> 우리가 한 첫 번째 실험은 잠재된 공간의 풍경을 이해하는 것이었습니다. 학습된 매니폴드를 걷는 것은 일반적으로 memorization의 징후(예리한 전환이 있는 경우)와 공간이 계층적으로 붕괴되는 방식에 대해 알려줄 수 있다. 이 잠재 공간을 걷는 것이 이미지 생성(예: 추가 및 제거되는 객체)에 의미론적 변화를 가져온다면, 우리는 모델이 관련적이고 흥미로운 표현을 학습했다고 추론할 수 있다. 결과는 그림 4에 나와 있습니다.

$6.2\;VISUALIZING\;THE\;DISCRIMINATOR\;FEATURES$

> In addition to the representations learnt by a discriminator, there is the question of what representations the generator learns. The quality of samples suggest that the generator learns specific object representations for major scene components such as beds, windows, lamps, doors, and miscellaneous furniture. In order to explore the form that these representations take, we conducted an experiment to attempt to remove windows from the generator completely.
>> discriminator에 의해 학습된 표현 외에, 생성자가 학습하는 표현에 대한 질문이 있다. 샘플의 품질은 제너레이터가 침대, 창문, 램프, 문 및 기타 가구와 같은 주요 장면 구성 요소에 대한 특정 객체 표현을 학습한다는 것을 시사합니다. 이러한 표현이 취하는 형태를 탐구하기 위해, 우리는 Generator에서 창을 완전히 제거하는 실험을 수행했다.

> On 150 samples, 52 window bounding boxes were drawn manually. On the second highest convolution layer features, logistic regression was fit to predict whether a feature activation was on a window (or not), by using the criterion that activations inside the drawn bounding boxes are positives and random samples from the same images are negatives. Using this simple model, all feature maps with weights greater than zero ( 200 in total) were dropped from all spatial locations. Then, random new samples were generated with and without the feature map removal.
>> 150개의 표본에서 52개의 창 경계 상자가 수동으로 그려졌습니다. 두 번째로 높은 convolution 레이어 특징에서, 로지스틱 회귀는 그려진 경계 상자 내부의 활성화가 긍정적이고 동일한 이미지의 무작위 샘플이 부정적이라는 기준을 사용하여 특징 활성화가 창에 있는지 여부를 예측하기 위해 적합했다. 이 간단한 모델을 사용하여 가중치가 0보다 큰 모든 형상 맵(총 200개)을 모든 공간 위치에서 삭제했다. 그런 다음 피쳐 맵 제거와 함께 또는 제거 없이 랜덤 새 샘플이 생성되었습니다.

> The generated images with and without the window dropout are shown in Fig.6, and interestingly, the network mostly forgets to draw windows in the bedrooms, replacing them with other objects.
>> 윈도우 드롭아웃이 있거나 없는 생성된 이미지는 그림 6에 나와 있으며, 흥미롭게도 네트워크는 대부분 침실에 창문을 그리는 것을 잊어버리고 다른 물체로 대체한다.

![Figure 4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-4.JPG)

> Figure 4: Top rows: Interpolation between a series of 9 random points in Z show that the space learned has smooth transitions, with every image in the space plausibly looking like a bedroom. In the 6th row, you see a room without a window slowly transforming into a room with a giant window. In the 10th row, you see what appears to be a TV slowly being transformed into a window.
>> 그림 4: 위쪽 행: Z에서 9개의 무작위 점들 사이의 보간은 학습된 공간이 매끄러운 전환을 가지고 있으며, 공간의 모든 이미지가 침실처럼 보인다. 여섯 번째 줄에서는 창문이 없는 방이 서서히 거대한 창문이 있는 방으로 변신하는 것을 볼 수 있다. 10번째 줄에서, 여러분은 TV로 보이는 것이 천천히 창으로 바뀌는 것을 볼 수 있습니다.

$6.3.2\;VECTOR\;ARITHMETIC\;ON\;FACE\;SAMPLES$

> In the context of evaluating learned representations of words (Mikolov et al., 2013) demonstrated that simple arithmetic operations revealed rich linear structure in representation space. One canonical example demonstrated that the vector(”King”) - vector(”Man”) + vector(”Woman”) resulted in a vector whose nearest neighbor was the vector for Queen. We investigated whether similar structure emerges in the Z representation of our generators. We performed similar arithmetic on the Z vectors of sets of exemplar samples for visual concepts. Experiments working on only single samples per concept were unstable, but averaging the Z vector for three examplars showed consistent and stable generations that semantically obeyed the arithmetic. In addition to the object manipulation shown in (Fig. 7), we demonstrate that face pose is also modeled linearly in Z space (Fig. 8).
>> 학습된 단어 표현을 평가하는 맥락에서(Mikolov et al., 2013)는 간단한 산술 연산이 표현 공간에서 풍부한 선형 구조를 드러낸다는 것을 보여주었다. 하나의 표준적인 예는 벡터("King") - 벡터("Man") + 벡터("Woman")가 퀸에 대한 벡터인 벡터를 생성한다는 것을 증명하였다. 우리는 Generator의 Z 표현에 유사한 구조가 나타나는지 조사했다. 시각적 개념을 위한 샘플 샘플 세트의 Z 벡터에 대해 유사한 연산을 수행했다. 개념당 단일 샘플로만 작업하는 실험은 불안정했지만, 세 개의 예시에 대한 Z 벡터의 평균화는 의미론적으로 산술에 복종하는 일관되고 안정적인 세대를 보여주었다. (그림 7)에 표시된 객체 조작 외에도 얼굴 포즈도 Z 공간에서 선형으로 모델링된다는 것을 보여준다(그림 8).

> These demonstrations suggest interesting applications can be developed using Z representations learned by our models. It has been previously demonstrated that conditional generative models can learn to convincingly model object attributes like scale, rotation, and position (Dosovitskiy et al., 2014). This is to our knowledge the first demonstration of this occurring in purely unsupervised models. Further exploring and developing the above mentioned vector arithmetic could dramatically reduce the amount of data needed for conditional generative modeling of complex image distributions.
>> 이러한 시연은 모델이 학습한 Z 표현을 사용하여 흥미로운 응용 프로그램을 개발할 수 있음을 시사한다. 조건부 생성 모델이 스케일, 회전 및 위치와 같은 객체 속성을 설득력 있게 모델링하는 방법을 배울 수 있다는 것이 이전에 입증되었다(Dosovitskyy et al., 2014). 이것은 우리가 아는 한 순수하게 감독되지 않은 모델에서 이것이 발생하는 첫 번째 시연이다. 위에서 언급한 벡터 산술을 추가로 탐색하고 개발하면 복잡한 이미지 분포의 조건부 생성 모델링에 필요한 데이터 양을 크게 줄일 수 있다.

![Figure 5](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-5.JPG)

> Figure 5: On the right, guided backpropagation visualizations of maximal axis-aligned responses for the first 6 learned convolutional features from the last convolution layer in the discriminator. Notice a significant minority of features respond to beds - the central object in the LSUN bedrooms dataset. On the left is a random filter baseline. Comparing to the previous responses there is little to no discrimination and random structure.
>> 그림 5: 오른쪽에는 discriminator의 마지막 convolution 레이어에서 처음 6개의 학습된 convolution 기능에 대한 최대 축 정렬 응답의 안내된 역 전파 시각화. LSUN 침실 데이터 세트의 중심 객체인 침대에 반응하는 기능이 상당히 적다. 왼쪽에는 임의 필터 기준선이 있습니다. 이전 응답과 비교했을 때 차별과 무작위 구조가 거의 또는 전혀 없습니다.

![Figure 6](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-6.JPG)

> Figure 6: Top row: un-modified samples from model. Bottom row: the same samples generated with dropping out ”window” filters. Some windows are removed, others are transformed into objects with similar visual appearance such as doors and mirrors. Although visual quality decreased, overall scene composition stayed similar, suggesting the generator has done a good job disentangling scene representation from object representation. Extended experiments could be done to remove other objects from the image and modify the objects the generator draws.
>> 그림 6: 상단 행: 모델의 수정되지 않은 샘플 맨 아래 행: "창" 필터를 삭제하면서 생성된 동일한 샘플입니다. 일부 창문은 제거되고, 다른 창문은 문과 거울과 같은 유사한 시각적 외관을 가진 물체로 변환됩니다. 시각적 품질은 감소했지만, 전체적인 장면 구성은 유사하게 유지되었으며, 이는 generator가 객체 표현과 장면 표현을 분리하는 작업을 잘 수행했음을 시사한다. 이미지에서 다른 개체를 제거하고 생성자가 그리는 개체를 수정하기 위해 확장 실험을 수행할 수 있습니다.

$7\;CONCLUSION\;AND\;FUTURE\;WORK$

> We propose a more stable set of architectures for training generative adversarial networks and we give evidence that adversarial networks learn good representations of images for supervised learning and generative modeling. There are still some forms of model instability remaining - we noticed as models are trained longer they sometimes collapse a subset of filters to a single oscillating mode. Further work is needed to tackle this from of instability. We think that extending this framework to other domains such as video (for frame prediction) and audio (pre-trained features for speech synthesis) should be very interesting. Further investigations into the properties of the learnt latent space would be interesting as well.
>> 우리는 생성적 적대 네트워크를 훈련시키기 위한 보다 안정적인 아키텍처 세트를 제안하고 적대적 네트워크가 지도 학습 및 생성 모델링을 위한 이미지의 좋은 표현을 학습한다는 증거를 제공한다. 모델 불안정성의 일부 형태는 여전히 남아 있다. 모델이 더 오래 훈련될수록 필터의 하위 집합을 단일 진동 모드로 접는 경우가 있다는 것을 알았다. 불안정성으로부터 이것을 다루기 위해서는 추가적인 작업이 필요하다. 우리는 이 프레임워크를 비디오(프레임 예측을 위한)와 오디오(음성 합성을 위한 사전 훈련된 기능)와 같은 다른 영역으로 확장하는 것이 매우 흥미로울 것이라고 생각한다. 학습된 잠재 공간의 특성에 대한 추가 조사도 흥미로울 것이다.

![Figure 7](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-7.JPG)

> Figure 7: Vector arithmetic for visual concepts. For each column, the Z vectors of samples are averaged. Arithmetic was then performed on the mean vectors creating a new vector Y . The center sample on the right hand side is produce by feeding Y as input to the generator. To demonstrate the interpolation capabilities of the generator, uniform noise sampled with scale +-0.25 was added to Y to produce the 8 other samples. Applying arithmetic in the input space (bottom two examples) results in noisy overlap due to misalignment.
>> 그림 7: 시각적 개념을 위한 벡터 산술 각 열에 대해 표본의 Z 벡터가 평균화됩니다. 그런 다음 평균 벡터에 대해 연산을 수행하여 새로운 벡터 Y를 생성한다. 오른쪽의 중앙 샘플은 Y를 generator에 입력으로 공급하여 생성된다. Generator의 보간 기능을 입증하기 위해 척도 +-0.25의 균일한 노이즈를 Y에 추가하여 다른 8개의 샘플을 생성했다. 입력 공간(아래 두 예)에 산술을 적용하면 정렬 불량으로 인해 노이즈가 겹칩니다.

![Figure 8](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-8.JPG)

> Figure 8: A ”turn” vector was created from four averaged samples of faces looking left vs looking right. By adding interpolations along this axis to random samples we were able to reliably transform their pose.
>> 그림 8:"turn" 벡터 얼굴의 4평균 표본 왼쪽 vs바로 관찰로 창조되었다.이 축을 따라 무작위로 샘플에 interpolations을 추가함으로써 우리는을 신뢰할 수 있게 포즈를 바꿀 수 있었다.

$8\;SUPPLEMENTARY\;MATERIAL$

$8.1\;EVALUATING\;DCGANs\;CAPABILITY\;TO\;CAPTURE\;DATA\;DISTRIBUTIONS$

> We propose to apply standard classification metrics to a conditional version of our model, evaluating the conditional distributions learned. We trained a DCGAN on MNIST (splitting off a 10K validation set) as well as a permutation invariant GAN baseline and evaluated the models using a nearest neighbor classifier comparing real data to a set of generated conditional samples. We found that removing the scale and bias parameters from batchnorm produced better results for both models. We speculate that the noise introduced by batchnorm helps the generative models to better explore and generate from the underlying data distribution. The results are shown in Table 3 which compares our models with other techniques. The DCGAN model achieves the same test error as a nearest neighbor classifier fitted on the training dataset - suggesting the DCGAN model has done a superb job at modeling the conditional distributions of this dataset. At one million samples per class, the DCGAN model outperforms InfiMNIST (Loosli et al., 2007), a hand developed data augmentation pipeline which uses translations and elastic deformations of training examples. The DCGAN is competitive with a probabilistic generative data augmentation technique utilizing learned per class transformations (Hauberg et al., 2015) while being more general as it directly models the data instead of transformations of the data.
>> 학습된 조건부 분포를 평가하여 모델의 조건부 버전에 표준 분류 메트릭을 적용할 것을 제안한다. 우리는 순열 불변 GAN 기준뿐만 아니라 MNIST(10K 검증 세트 분할)에 대한 DCGAN을 훈련하고 생성된 조건부 샘플 세트와 실제 데이터를 비교하는 가장 가까운 이웃 분류기를 사용하여 모델을 평가했다. 배치 규범에서 스케일 및 바이어스 매개 변수를 제거하면 두 모델 모두에서 더 나은 결과가 나온다는 것을 발견했다. 배치 노름에 의해 유입된 노이즈가 생성 모델이 기본 데이터 분포를 더 잘 탐색하고 생성하는 데 도움이 된다고 추측한다. 결과는 우리의 모델을 다른 기술과 비교하는 표 3에 나와 있다. DCGAN 모델은 훈련 데이터 세트에 적합한 가장 가까운 이웃 분류기와 동일한 테스트 오류를 달성한다. 이는 DCGAN 모델이 이 데이터 세트의 조건부 분포를 모델링하는 데 탁월한 성과를 거뒀음을 시사한다. 클래스당 100만 개의 샘플에서 DCGAN 모델은 훈련 예제의 번역과 탄성 변형을 사용하는 수작업으로 개발된 데이터 증강 파이프라인 InfiMNIST(Loosli et al., 2007)를 능가한다. DCGAN은 학습된 클래스당 변환을 활용하는 확률론적 생성 데이터 확대 기법(Hauberg et al., 2015)과 경쟁하는 동시에 데이터의 변환 대신 데이터를 직접 모델링하기 때문에 더 일반적이다.


![Figure 9](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-9.JPG)

> Figure 9: Side-by-side illustration of (from left-to-right) the MNIST dataset, generations from a baseline GAN, and generations from our DCGAN.
>> 그림 9: MNIST 데이터 세트(왼쪽에서 오른쪽으로)와 기본 GAN의 세대 및 DCGAN의 세대를 나란히 나타낸 그림.

![Figure 10](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-10.JPG)

> Figure 10: More face generations from our Face DCGAN.
>> 그림 10: Face DCGAN의 더 많은 얼굴 세대

![Figure 11](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-05-18-(GAN)DCGAN-translation/Figure-11.JPG)

> Figure 11: Generations of a DCGAN that was trained on the Imagenet-1k dataset
>> 그림 11: Imagenet-1k 데이터 세트에 대해 학습을 받은 DCGAN 세대