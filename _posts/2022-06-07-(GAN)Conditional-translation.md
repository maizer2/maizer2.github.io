---
layout: post 
title: "(GAN)Conditional Generative Adversarial Nets Translation"
categories: [1. Computer Engineering]
tags: [1.7. Literature Review, 1.2.2.5. GAN]
---

### [GAN Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/Literature-of-GAN.html)

### $$\mathbf{Conditional\;Generative\;Adversarial\;Nets}$$

#### $$\mathbf{Abstract}$$

> Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.
>> 생성적 적대 신경망[8]은 생성 모델을 훈련시키는 새로운 방법으로 최근에 소개되었다. 이 연구에서 우리는 data와 y를 단순히 공급하여 구성할 수 있는 조건부 버전의 생성 적대적 네트워크를 소개하며, 우리는 생성자와 판별자 모두에 조건을 붙이기를 원한다. 우리는 이 모델이 클래스 레이블에서 조건화된 MNIST 숫자를 생성할 수 있음을 보여준다. 또한 이 모델이 다중 모드 모델을 학습하는 데 어떻게 사용될 수 있는지 설명하고, 이 접근 방식이 교육 레이블의 일부가 아닌 설명적 태그를 어떻게 생성할 수 있는지 보여주는 이미지 태그에 대한 응용 프로그램의 예시를 제공한다.

### $1\;\mathbf{Introduction}$

> Generative adversarial nets were recently introduced as an alternative framework for training generative models in order to sidestep the difficulty of approximating many intractable probabilistic computations.
>> 많은 다루기 어려운 확률론적 계산의 근사화를 피하기 위해 생성 모델을 훈련시키기 위한 대안적 프레임워크로 생성적 적대 네트워크가 최근에 도입되었다.

> Adversarial nets have the advantages that Markov chains are never needed, only backpropagation is used to obtain gradients, no inference is required during learning, and a wide variety of factors and interactions can easily be incorporated into the model.
>> 적대적 네트는 마르코프 체인이 절대 필요하지 않고, 그레이디언트를 얻기 위해 역 전파만 사용되며, 학습 중에 추론이 필요하지 않으며, 다양한 요인과 상호 작용을 모델에 쉽게 통합할 수 있다는 장점이 있다.

> Furthermore, as demonstrated in [8], it can produce state of the art log-likelihood estimates and realistic samples.
>> 또한, [8]에 설명된 바와 같이, 최첨단 로그 우도 추정치와 현실적인 샘플을 생성할 수 있다.

> In an unconditioned generative model, there is no control on modes of the data being generated. However, by conditioning the model on additional information it is possible to direct the data generation process. Such conditioning could be based on class labels, on some part of data for inpainting like [5], or even on data from different modality.
>> 무조건 생성 모델에서는 생성되는 데이터의 모드를 제어할 수 없습니다. 그러나 모델을 추가 정보로 조정함으로써 데이터 생성 프로세스를 지시할 수 있다. 그러한 조건은 클래스 라벨, [5]와 같은 인페인팅 데이터의 일부 또는 다른 양식의 데이터에 기초할 수 있다.

> In this work we show how can we construct the conditional adversarial net. And for empirical results we demonstrate two set of experiment. One on MNIST digit data set conditioned on class labels and one on MIR Flickr 25,000 dataset [10] for multi-modal learning.
>> 본 연구에서는 조건부 적대적 네트워크를 구성하는 방법을 보여준다. 그리고 경험적 결과를 위해 우리는 두 가지 실험을 시연한다. 하나는 MNIST 디지털 데이터 세트이며 하나는 다중 모드 학습을 위한 MIR Flickr 25,000 데이터 세트[10]에 있다.

### $2\;\mathbf{Related\;Work}$

#### $\mathbf{2.1 Multi-modal\;Learning\;For\;Image\;Labelling}$

> Despite the many recent successes of supervised neural networks (and convolutional networks in particular) [13, 17], it remains challenging to scale such models to accommodate an extremely large number of predicted output categories. A second issue is that much of the work to date has focused on learning one-to-one mappings from input to output. However, many interesting problems are more naturally thought of as a probabilistic one-to-many mapping. For instance in the case of image labeling there may be many different tags that could appropriately applied to a given image, and different (human) annotators may use different (but typically synonymous or related) terms to describe the same image.
>> 감독된 신경 네트워크(특히 컨볼루션 네트워크)의 최근 많은 성공에도 불구하고, 매우 많은 수의 예측 출력 범주를 수용하기 위해 이러한 모델을 확장하는 것은 여전히 어려운 일이다. 두 번째 문제는 지금까지의 많은 작업이 입력에서 출력까지 일대일 매핑을 학습하는 데 집중되었다는 것이다. 그러나 많은 흥미로운 문제는 확률적 일대다 매핑으로 더 자연스럽게 생각된다. 예를 들어 이미지 라벨링의 경우 주어진 이미지에 적절하게 적용할 수 있는 다양한 태그가 있을 수 있으며, 다른 (인간) 주석자들은 동일한 이미지를 설명하기 위해 다른 (그러나 일반적으로 동의어 또는 관련) 용어를 사용할 수 있다.

> One way to help address the first issue is to leverage additional information from other modalities: for instance, by using natural language corpora to learn a vector representation for labels in which geometric relations are semantically meaningful. When making predictions in such spaces, we benefit from the fact that when prediction errors we are still often ‘close’ to the truth (e.g. predicting ’table’ instead of ’chair’), and also from the fact that we can naturally make predictive generalizations to labels that were not seen during training time. Works such as [3] have shown that even a simple linear mapping from image feature-space to word-representation-space can yield improved classification performance.
>> 첫 번째 문제를 해결하는 데 도움이 되는 한 가지 방법은 다른 양식에서 추가 정보를 활용하는 것이다. 예를 들어, 기하학적 관계가 의미적으로 의미가 있는 레이블에 대한 벡터 표현을 학습하기 위해 자연어 말뭉치를 사용하는 것이다. 그러한 공간에서 예측을 할 때, 우리는 예측 오류가 여전히 진실에 '가까이'하다는 사실(예: '의자' 대신 '표'를 예측하는 것)과 훈련 시간 동안 보이지 않았던 레이블에 대한 예측 일반화를 자연스럽게 할 수 있다는 사실로부터 이익을 얻는다. [3]과 같은 연구는 이미지 특징 공간에서 단어 표현 공간으로 가는 간단한 선형 매핑도 향상된 분류 성능을 산출할 수 있다는 것을 보여주었다.

> One way to address the second problem is to use a conditional probabilistic generative model, the input is taken to be the conditioning variable and the one-to-many mapping is instantiated as a conditional predictive distribution.
>> 두 번째 문제를 해결하는 한 가지 방법은 조건부 확률론적 생성 모델을 사용하는 것이며, 입력은 조건 변수로 간주되고 일대다 매핑은 조건부 예측 분포로 인스턴스화된다.

> [16] take a similar approach to this problem, and train a multi-modal Deep Boltzmann Machine on the MIR Flickr 25,000 dataset as we do in this work.
>> [16] 이 문제에 유사한 접근 방식을 취하고, 이 작업에서와 같이 MIR Flickr 25,000 데이터 세트에서 다중 모달 딥 볼츠만 기계를 훈련시킨다.

> Additionally, in [12] the authors show how to train a supervised multi-modal neural language model, and they are able to generate descriptive sentence for images.
>> 또한, [12]에서 저자는 감독된 다중 모드 신경 언어 모델을 훈련하는 방법을 보여주며, 이미지에 대한 설명 문장을 생성할 수 있다.

### $\mathbf{3\;Conditional\;Adversarial\;Nets}$

#### $\mathbf{3.1\;Generative\;Adversarial\;Nets}$

> Generative adversarial nets were recently introduced as a novel way to train a generative model. They consists of two ‘adversarial’ models: a generative model $G$ that captures the data distribution, and a discriminative model $D$ that estimates the probability that a sample came from the training data rather than $G$. Both $G$ and $D$ could be a non-linear mapping function, such as a multi-layer perceptron.
>> 생성적 적대망은 생성 모델을 훈련시키는 새로운 방법으로 최근에 소개되었다. 이들은 두 가지 '적대적' 모델, 즉 데이터 분포를 포착하는 생성 모델 $G$와 $G$가 아닌 훈련 데이터에서 샘플이 나왔을 확률을 추정하는 차별 모델 $D$로 구성된다. $G$와 $D$는 모두 다층 퍼셉트론과 같은 비선형 매핑 함수일 수 있다.

> To learn a generator distribution $p_{g}$ over data data $x$, the generator builds a mapping function from a prior noise distribution $p_{z}(z)$ to data space as $G(z; \theta_{g})$. And the discriminator, $D(x; \theta_{d})$, outputs a single scalar representing the probability that $x$ came form training data rather than $p_{g}$. 
>> 데이터 $x$에 대한 생성기 분포 $p_{g}$를 학습하기 위해, 생성기는 이전의 노이즈 분포 $p_{z}(z)$에서 $G(z; \theta_{g})$로 데이터 공간에 대한 매핑 함수를 구축한다. 그리고 판별기 $D(x; \theta_{d})$는 $x$가 $p_{g}$가 아닌 훈련 데이터에서 왔을 확률을 나타내는 단일 스칼라를 출력한다.

> $G$ and $D$ are both trained simultaneously: we adjust parameters for $G$ to minimize $\log{(1 − D(G(z))}$ and adjust parameters for $D$ to minimize $\log{D(X)}$, as if they are following the two-player min-max game with value function $V (G, D)$:
>> $G$와 $D$는 모두 동시에 훈련된다. $G$에 대한 매개 변수를 조정하여 $\log{(1 - D(z)}$를 최소화하고 $D$에 대한 매개 변수를 조정하여 가치 함수 $V(G, D)$를 가진 2인용 미니맥스 게임을 따르는 것처럼 $\log{D(X)}$를 최소화한다.

$$\underset{G}{\min}\underset{D}{\max}V(D,G)=E_{x\sim p_{data}}[\log{D(x)}]+E_{z\sim{noise}}[\log{(1-D(G(z)))}].$$

#### $\mathbf{3.2\;Conditional\;Adversarial\;Nets}$

> Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are conditioned on some extra information $y$. $y$ could be any kind of auxiliary information, such as class labels or data from other modalities. We can perform the conditioning by feeding $y$ into the both the discriminator and generator as additional input layer.
>> 생성자 및 판별자 모두가 일부 추가 정보 $y$에 따라 조건화된 경우 생성 적대적 네트는 조건부 모델로 확장될 수 있다. $y$는 클래스 레이블 또는 다른 양식의 데이터와 같은 모든 종류의 보조 정보일 수 있다. $y$를 판별기와 생성기 모두에 추가 입력 레이어로 공급하여 조건화를 수행할 수 있다.

> In the generator the prior input noise $p_{z}(z)$, and $y$ are combined in joint hidden representation, and the adversarial training framework allows for considerable flexibility in how this hidden representation is composed.
>> 생성기에서 이전 입력 노이즈 $p_{z}(z)$와 $y$는 공동 숨겨진 표현으로 결합되며, 적대적 훈련 프레임워크는 이 숨겨진 표현이 구성되는 방법에 상당한 유연성을 허용한다.

> In the discriminator $x$ and $y$ are presented as inputs and to a discriminative function (embodied again by a MLP in this case).
>> 판별기에서 $x$와 $y$는 입력 및 판별 함수로 표시된다(이 경우 MLP에 의해 다시 구현된다).

> The objective function of a two-player minimax game would be as Eq 2
>> 2인용 미니맥스 게임의 목적 함수는 Eq 2와 같다.

$$\underset{G}{\min}\underset{D}{\max}V(D,G)=E_{x\sim{p_{data}(x)}}[\log{D(x\mid{y})}]+E_{z\sim{p_{z}(z)}}[\log{(1-D(G(z\mid{y})))}].$$

> Fig 1 illustrates the structure of a simple conditional adversarial net.
>> 그림 1은 간단한 조건부 적대적 네트워크의 구조를 보여준다.

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Conditional-translation/Figure-1.JPG)

### $\mathbf{4\;Experimental\;Results}$

#### $\mathbf{4.1\;Unimodal}$

> We trained a conditional adversarial net on MNIST images conditioned on their class labels, encoded as one-hot vectors.
>> 우리는 클래스 레이블에 따라 조건화된 MNIST 이미지에 대한 조건부 적대적 네트워크를 훈련시켜 원핫 벡터로 인코딩했다.

> In the generator net, a noise prior $z$ with dimensionality 100 was drawn from a uniform distribution within the unit hypercube. Both $z$ and $y$ are mapped to hidden layers with Rectified Linear Unit (ReLu) activation [4, 11], with layer sizes 200 and 1000 respectively, before both being mapped to second, combined hidden ReLu layer of dimensionality 1200. We then have a final sigmoid unit layer as our output for generating the 784-dimensional MNIST samples.
>> 발생기 네트워크에서, 차원 100을 가진 $z$ 이전의 노이즈는 단위 하이퍼큐브 내의 균일한 분포로부터 도출되었다. $z$와 $y$는 모두 정류 선형 단위(ReLu) 활성화 [4, 11]로 숨겨진 레이어에 매핑되며, 레이어 크기는 각각 200과 1000이며, 두 번째 결합된 차원 1200의 숨겨진 ReLu 레이어에 매핑된다. 그런 다음 784차원 MNIST 샘플을 생성하기 위한 출력으로 최종 시그모이드 단위 레이어를 갖는다.

> Table 1: Parzen window-based log-likelihood estimates for MNIST. We followed the same procedure as [8] for computing these values.
>> 표 1: MNIST에 대한 Parzen 창 기반 로그 우도 추정치. 이러한 값을 계산하기 위해 [8]과 동일한 절차를 따랐다.

> The discriminator maps $x$ to a maxout [6] layer with 240 units and 5 pieces, and $y$ to a maxout layer with 50 units and 5 pieces. Both of the hidden layers mapped to a joint maxout layer with 240 units and 4 pieces before being fed to the sigmoid layer. (The precise architecture of the discriminator is not critical as long as it has sufficient power; we have found that maxout units are typically well suited to the task.)
>> 판별기는 $x$를 240개의 유닛과 5개의 피스로 구성된 최대 [6] 레이어에 매핑하고 $y$를 50개의 유닛과 5개의 피스로 구성된 최대 레이어에 매핑한다. 시그모이드 층에 공급되기 전에 240개의 유닛과 4개의 조각이 있는 공동 최대 출력 층에 매핑된 숨겨진 레이어 둘 다. (판별기의 정확한 아키텍처는 충분한 전력을 가지고 있는 한 중요하지 않다; 우리는 최대 출력 단위가 일반적으로 작업에 잘 적합하다는 것을 발견했다.

> The model was trained using stochastic gradient decent with mini-batches of size 100 and initial learning rate of 0.1 which was exponentially decreased down to .000001 with decay factor of 1.00004. Also momentum was used with initial value of .5 which was increased up to 0.7. Dropout [9] with probability of 0.5 was applied to both the generator and discriminator. And best estimate of log-likelihood on the validation set was used as stopping point.
>> 모델은 크기가 100인 미니 배치로 괜찮은 확률적 그레이디언트를 사용하여 훈련되었으며, 0.1의 초기 학습률은 붕괴 계수가 1.00004인 .000001로 기하급수적으로 감소했다. 또한 모멘텀은 0.7까지 상승한 0.5의 초기값을 사용하였다. 발생기와 판별기 모두에 0.5의 확률로 탈락[9]을 적용했다. 그리고 유효성 검사 세트에 대한 로그 우도의 최선의 추정치가 정지점으로 사용되었습니다.

![Table 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Conditional-translation/Table-1.JPG)

> Table 1 shows Gaussian Parzen window log-likelihood estimate for the MNIST dataset test data. 1000 samples were drawn from each 10 class and a Gaussian Parzen window was fitted to these samples. We then estimate the log-likelihood of the test set using the Parzen window distribution. (See [8] for more details of how this estimate is constructed.)
>> 표 1은 MNIST 데이터 세트 테스트 데이터에 대한 가우스 파르젠 창 로그 우도 추정치를 보여준다. 각 10개 클래스에서 1000개의 검체가 추출되었으며 이러한 검체에 가우스 파르젠 창이 장착되었습니다. 그런 다음 Parzen 창 분포를 사용하여 테스트 세트의 로그 우도를 추정한다. (이 추정치의 구성 방법에 대한 자세한 내용은 [8]을 참조하십시오.)

> The conditional adversarial net results that we present are comparable with some other network based, but are outperformed by several other approaches – including non-conditional adversarial nets. We present these results more as a proof-of-concept than as demonstration of efficacy, and believe that with further exploration of hyper-parameter space and architecture that the conditional model should match or exceed the non-conditional results. 
>> 우리가 제시하는 조건부 적대적 네트 결과는 다른 네트워크 기반과 비교할 수 있지만, 비조건적 적대적 네트 등 몇 가지 다른 접근 방식에 의해 능가한다. 우리는 이러한 결과를 효능의 입증보다 개념 증명으로 제시하고, 하이퍼 매개 변수 공간 및 아키텍처에 대한 추가 탐색을 통해 조건부 모델이 조건 없는 결과와 일치하거나 초과해야 한다고 믿는다.

> Fig 2 shows some of the generated samples. Each row is conditioned on one label and each column is a different generated sample.
>> 그림 2는 생성된 샘플 중 일부를 보여준다. 각 행은 하나의 레이블에서 조건화되어 있으며 각 열은 서로 다른 생성된 표본입니다.

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Conditional-translation/Figure-2.JPG)

#### $\mathbf{4.2\;Multimodal}$

> Photo sites such as Flickr are a rich source of labeled data in the form of images and their associated user-generated metadata (UGM) — in particular user-tags.
>> Flickr와 같은 사진 사이트는 이미지 및 관련 사용자 생성 메타데이터(UGM)(특히 사용자 태그) 형태의 레이블링된 데이터의 풍부한 소스입니다.

> User-generated metadata differ from more ‘canonical’ image labelling schems in that they are typically more descriptive, and are semantically much closer to how humans describe images with natural language rather than just identifying the objects present in an image. Another aspect of UGM is that synoymy is prevalent and different users may use different vocabulary to describe the same concepts — consequently, having an efficient way to normalize these labels becomes important. Conceptual word embeddings [14] can be very useful here since related concepts end up being represented by similar vectors.
>> 사용자 생성 메타데이터는 일반적으로 더 설명적이라는 점에서 더 '규범적인' 이미지 레이블링 체계와 다르며, 의미론적으로 인간이 이미지에 존재하는 객체를 식별하는 것보다 자연 언어로 이미지를 설명하는 방법에 훨씬 가깝다. UGM의 또 다른 측면은 동의어가 널리 보급되어 있고 다른 사용자들이 동일한 개념을 설명하기 위해 다른 어휘를 사용할 수 있다는 것이다. 개념적 단어 임베딩[14]은 관련 개념이 유사한 벡터로 표현되기 때문에 여기서 매우 유용할 수 있다.

> In this section we demonstrate automated tagging of images, with multi-label predictions, using conditional adversarial nets to generate a (possibly multi-modal) distribution of tag-vectors conditional on image features.
>> 이 섹션에서는 이미지 특징에 따라 태그 벡터의 (아마도 다중 모달) 분포를 생성하기 위해 조건부 적대적 네트워크를 사용하여 다중 레이블 예측과 함께 이미지의 자동 태그 지정을 보여준다.

> For image features we pre-train a convolutional model similar to the one from [13] on the full ImageNet dataset with 21,000 labels [15]. We use the output of the last fully connected layer with 4096 units as image representations.
>> 이미지 기능을 위해 21,000개의 레이블이 있는 전체 ImageNet 데이터 세트에서 [13]의 것과 유사한 컨볼루션 모델을 사전 교육한다[15]. 우리는 4096개의 유닛이 있는 마지막 완전히 연결된 레이어의 출력을 이미지 표현으로 사용한다.

> For the world representation we first gather a corpus of text from concatenation of user-tags, titles and descriptions from YFCC100M 2 dataset metadata. After pre-processing and cleaning of the text we trained a skip-gram model [14] with word vector size of 200. And we omitted any word appearing less than 200 times from the vocabulary, thereby ending up with a dictionary of size 247465.
>> 세계 표현을 위해 먼저 YFCC100M2 데이터 세트 메타데이터의 사용자 태그, 제목 및 설명 연결에서 텍스트 코퍼스를 수집한다. 텍스트의 사전 처리 및 정리 후 단어 벡터 크기가 200인 스킵그램 모델 [14]을 교육했다. 그리고 우리는 단어에서 200번도 안 되는 단어를 생략했고, 결과적으로 247465 크기의 사전이 되었습니다.

> We keep the convolutional model and the language model fixed during training of the adversarial net. And leave the experiments when we even backpropagate through these models as future work.
>> 우리는 적대적 네트워크를 훈련하는 동안 컨볼루션 모델과 언어 모델을 고정시킨다. 그리고 우리가 이 모델들을 통해 역확산할 때 실험을 미래의 작업으로 남겨두세요.

> For our experiments we use MIR Flickr 25,000 dataset [10], and extract the image and tags features using the convolutional model and language model we described above. Images without any tag were omitted from our experiments and annotations were treated as extra tags. The first 150,000 examples were used as training set. Images with multiple tags were repeated inside the training set once for each associated tag.
>> 실험을 위해 MIR Flickr 25,000 데이터 세트[10]를 사용하고 위에서 설명한 컨볼루션 모델과 언어 모델을 사용하여 이미지와 태그 기능을 추출한다. 태그가 없는 이미지는 실험에서 제외되었으며 주석이 추가 태그로 처리되었다. 처음 150,000개의 예제가 훈련 세트로 사용되었다. 연관된 각 태그에 대해 교육 세트 내에서 여러 개의 태그가 있는 이미지를 한 번 반복했습니다.

> For evaluation, we generate 100 samples for each image and find top 20 closest words using cosine similarity of vector representation of the words in the vocabulary to each sample. Then we select the top 10 most common words among all 100 samples. Table 4.2 shows some samples of the user assigned tags and annotations along with the generated tags.
>> 평가를 위해 각 이미지에 대해 100개의 샘플을 생성하고 각 샘플에 대한 어휘의 벡터 표현의 코사인 유사성을 사용하여 상위 20개의 가장 가까운 단어를 찾는다. 그런 다음 100개의 샘플 중 가장 일반적인 단어 10개를 선택합니다. 표 4.2는 생성된 태그와 함께 사용자가 할당한 태그 및 주석의 일부 샘플을 보여준다.

> The best working model’s generator receives Gaussian noise of size 100 as noise prior and maps it to 500 dimension ReLu layer. And maps 4096 dimension image feature vector to 2000 dimension ReLu hidden layer. Both of these layers are mapped to a joint representation of 200 dimension linear layer which would output the generated word vectors.
>> 최상의 작업 모델의 생성기는 크기 100의 가우스 노이즈를 노이즈 사전으로 수신하고 이를 500차원 ReLu 레이어에 매핑한다. 그리고 4096차원 이미지 특징 벡터를 2000차원 ReLu 은닉 레이어에 매핑한다. 이 두 계층은 생성된 단어 벡터를 출력하는 200차원 선형 계층의 공동 표현에 매핑된다.

> The discriminator is consisted of 500 and 1200 dimension ReLu hidden layers for word vectors and image features respectively and maxout layer with 1000 units and 3 pieces as the join layer which is finally fed to the one single sigmoid unit.
>> 판별기는 각각 단어 벡터와 이미지 특징을 위한 500 및 1200차원 ReLu 은닉 레이어와 하나의 단일 시그모이드 유닛에 최종적으로 공급되는 조인 레이어로 1000 유닛과 3개의 조각을 갖는 maxout 레이어로 구성된다.

> The model was trained using stochastic gradient decent with mini-batches of size 100 and initial learning rate of 0.1 which was exponentially decreased down to .000001 with decay factor of 1.00004. Also momentum was used with initial value of .5 which was increased up to 0.7. Dropout with probability of 0.5 was applied to both the generator and discriminator. 
>> 모델은 크기가 100인 미니 배치로 괜찮은 확률적 그레이디언트를 사용하여 훈련되었으며, 0.1의 초기 학습률은 붕괴 계수가 1.00004인 .000001로 기하급수적으로 감소했다. 또한 모멘텀은 0.7까지 상승한 0.5의 초기값을 사용하였다. 발생기와 판별기 모두에 0.5의 확률로 탈락이 적용되었다.

> The hyper-parameters and architectural choices were obtained by cross-validation and a mix of random grid search and manual selection (albeit over a somewhat limited search space.)
>> 초 매개 변수와 아키텍처 선택은 교차 검증과 무작위 그리드 검색 및 수동 선택을 혼합하여 얻었다(어느 정도 제한된 검색 공간에도 불구하고).

### $\mathbf{5\;Future\;Work}$

> The results shown in this paper are extremely preliminary, but they demonstrate the potential of conditional adversarial nets and show promise for interesting and useful applications.
>> 본 논문에서 보여지는 결과는 극히 예비적이지만 조건부 적대적 네트의 잠재력을 입증하고 흥미롭고 유용한 애플리케이션에 대한 가능성을 보여준다.

> In future explorations between now and the workshop we expect to present more sophisticated models, as well as a more detailed and thorough analysis of their performance and characteristics.
>> 지금부터 워크숍 사이의 향후 탐색에서는 보다 정교한 모델과 더불어 성능 및 특성에 대한 보다 상세하고 철저한 분석을 제시할 것으로 예상된다.

![Table 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-07-(GAN)Conditional-translation/Table-2.JPG)

> Also, in the current experiments we only use each tag individually. But by using multiple tags at the same time (effectively posing generative problem as one of ‘set generation’) we hope to achieve better results.
>> 또한, 현재 실험에서는 각 태그를 개별적으로만 사용합니다. 그러나 동시에 여러 태그를 사용함으로써('세트 세대' 중 하나로 효과적으로 생성 문제를 제기함) 더 나은 결과를 얻기를 바란다.

> Another obvious direction left for future work is to construct a joint training scheme to learn the language model. Works such as [12] has shown that we can learn a language model for suited for the specific task.
>> 향후 작업을 위해 남겨진 또 다른 분명한 방향은 언어 모델을 학습하기 위한 공동 훈련 계획을 구성하는 것이다. [12]와 같은 연구는 우리가 특정 작업에 적합한 언어 모델을 배울 수 있다는 것을 보여주었다.

#### $\mathbf{Acknowledgments}$

> This project was developed in Pylearn2 [7] framework, and we would like to thank Pylearn2 developers. We also like to thank Ian Goodfellow for helpful discussion during his affiliation at University of Montreal. The authors gratefully acknowledge the support from the Vision & Machine Learning, and Production Engineering teams at Flickr (in alphabetical order: Andrew Stadlen, Arel Cordero, Clayton Mellina, Cyprien Noel, Frank Liu, Gerry Pesavento, Huy Nguyen, Jack Culpepper, John Ko, Pierre Garrigues, Rob Hess, Stacey Svetlichnaya, Tobi Baumgartner, and Ye Lu)
>> 이 프로젝트는 Pylearn2 [7] 프레임워크에서 개발되었으며, Pylearn2 개발자들에게 감사를 드립니다. 우리는 또한 몬트리올 대학에서의 이안 굿펠로우가 소속되어 있는 동안 도움이 되는 논의를 해주신 것에 대해 감사드리고 싶습니다. 저자는 Flickr의 Vision & Machine Learning 및 Production Engineering 팀(가나다순: 앤드루 슈타들렌, 아렐 코데로, 클레이튼 멜리나, 사이프리앵 노엘, 프랭크 류, 게리 페사벤토, 휴이 응우옌, 잭 쿨페퍼, 존 코, 피에르 개리그스, 롭 헤스, 스테이시 스베틀리흐나야, 토비 바움가르트너)