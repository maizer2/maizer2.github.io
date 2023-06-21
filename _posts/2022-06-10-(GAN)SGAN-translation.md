---
layout: post 
title: "(GAN)Semi-Supervised Learning with Generative Adversarial Networks Translation"
categories: [1. Computer Engineering]
tags: [1.7. Paper Review, 1.2.2.5. GAN]
---

### [GAN Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/05/23/paper-of-GAN.html)

## <center>$$\mathbf{Semi-Supervised\;Learning\;with\;Generative\;Adversarial\;Networks}$$</center>

### $$\mathbf{Abstract}$$

> We extend Generative Adversarial Networks (GANs) to the semi-supervised context by forcing the discriminator network to output class labels. We train a generative model G and a discriminator D on a dataset with inputs belonging to one of N classes. At training time, D is made  to predict which of N+1 classes the input belongs to, where an extra class is added to correspond to the outputs of G. We show that this method can be used to create a more data-efficient classifier and that it allows for generating higher quality samples than a regular GAN.
>> 우리는 판별기 네트워크가 클래스 레이블을 출력하도록 하여 GAN(Generative Adversarial Network)을 준지도 컨텍스트까지 확장한다. 우리는 N 클래스 중 하나에 속하는 입력으로 데이터 세트에서 생성 모델 G와 판별기 D를 훈련한다. 훈련 시간에 D는 입력이 N+1 클래스 중 어느 것에 속하는지 예측하기 위해 만들어지며, 여기서 G의 출력에 대응하는 추가 클래스가 추가된다. 우리는 이 방법을 사용하여 더 데이터 효율적인 분류기를 만들 수 있으며 일반 GAN보다 더 높은 품질의 샘플을 생성할 수 있음을 보여준다.

### $\mathbf{1\;Introduction}$

> Work on generating images with Generative Adversarial Networks (GANs) has shown promising results (Goodfellow et al., 2014). A generative net G and a discriminator D are trained simultaneously with conflicting objectives. G takes in a noise vector and outputs an image, while D takes in an image and outputs a prediction about whether the image is a sample from G. G is trained to maximize the  probability that D makes a mistake, and D is trained to minimize that probability. Building on these ideas, one can generate good output samples using a cascade (Denton et al., 2015) of convolutional neural networks. More recently (Radford et al., 2015), even better samples were created from a single generator network. Here, we consider the situation where we try to solve a semi-supervised classification task and learn a generative model simultaneously. For instance, we may learn a generative model for MNIST images while we train an image classifier, which we’ll call C. Using generative models on semi-supervised learning tasks is not a new idea - Kingma et al. (2014) expand work on variational generative techniques (Kingma & Welling, 2013; Rezende et al., 2014) to do just that. Here, we attempt to do something similar with GANs. We are not the first to use GANs for semi-supervised learning. The CatGAN (Springenberg, 2015) modifies the objective function to take into account mutual information between observed examples and their predicted class distribution. In Radford et al. (2015), the features learned by D are reused in classifiers.
>> GAN(Generative Adversarial Networks)을 사용하여 이미지를 생성하는 작업은 유망한 결과를 보여주었다(Goodfellow et al., 2014). 생성 네트워크 G와 판별기 D는 상충되는 목표를 가지고 동시에 훈련된다. G는 노이즈 벡터를 받아들여 이미지를 출력하는 반면, D는 이미지를 가져와 이미지가 G의 샘플인지 여부에 대한 예측을 출력한다. G는 D가 실수할 확률을 최대화하도록 훈련되고 D는 그 확률을 최소화하도록 훈련된다. 이러한 아이디어를 바탕으로 컨볼루션 신경망의 캐스케이드(Denton et al., 2015)를 사용하여 좋은 출력 샘플을 생성할 수 있다. 더 최근에는(Radford et al., 2015), 단일 발전기 네트워크에서 훨씬 더 나은 샘플이 생성되었다. 여기서는 준지도 분류 과제를 해결하고 생성 모델을 동시에 학습하려고 하는 상황을 고려한다. 예를 들어, 우리는 이미지 분류기를 훈련하는 동안 MNIST 이미지에 대한 생성 모델을 배울 수 있으며, 이를 C라고 부른다. 준지도 학습 과제에 생성 모델을 사용하는 것은 새로운 아이디어가 아니다. Kingma 외(2014)는 바로 그렇게 하기 위해 다양한 생성 기술(Kingma & Welling, 2013; Rezende 외, 2014)에 대한 작업을 확장한다. 여기서 우리는 GAN과 유사한 것을 시도한다. 준지도 학습에 GAN을 사용한 것은 우리가 처음이 아니다. CatGAN(Springenberg, 2015)은 관찰된 예와 예측된 클래스 분포 사이의 상호 정보를 고려하기 위해 목적 함수를 수정한다. Radford 외 연구진(2015)에서 D가 학습한 기능은 분류기에서 재사용된다.

> The latter demonstrates the utility of the learned  representations, but it has several undesirable properties. First, the fact that representations learned by D help improve C is not surprising - it seems reasonable that this should work. However, it also seems reasonable that learning a good C would help to improve the performance of D. For instance, maybe images where the output of C has high entropy are more likely to come from G. If we simply use the learned representations of D after the fact to augment C, we don’t take advantage of this. Second, using the learned representations of D after the fact doesn’t allow for training C and G simultaneously. We’d like to be able to do this for efficiency reasons, but there is a more important motivation. If improving D improves C, and improving C improves D (which we know improves G) then we may be able to take advantage of a sort of feedback loop, in which all 3 components (G,C and D) iteratively make each other better. 
>> 후자는 학습된 표현의 유용성을 보여주지만, 몇 가지 바람직하지 않은 특성을 가지고 있다. 첫째, D가 학습한 표현이 C를 개선하는 데 도움이 된다는 사실은 놀라운 일이 아니다. - 이것이 효과가 있어야 한다는 것은 합리적인 것처럼 보인다. 그러나 좋은 C를 배우는 것이 D의 성능을 향상시키는 데 도움이 될 것이라는 점 또한 타당해 보인다. 예를 들어, C의 출력이 높은 엔트로피를 갖는 이미지는 G에서 나올 가능성이 더 높다. 만약 우리가 단순히 C를 증가시키기 위해 그 사실 이후에 학습된 D의 표현을 사용한다면, 우리는 이것을 이용하지 않는다. 둘째, 사실 이후에 학습된 D의 표현을 사용하는 것은 C와 G를 동시에 훈련시키는 것을 허용하지 않는다. 효율성의 이유로 이 작업을 수행할 수 있기를 바라지만, 더 중요한 동기가 있습니다. D를 개선하면 C가 개선되고, C를 개선하면 D가 개선된다면, 우리는 세 가지 요소(G, C, D)가 반복적으로 서로를 더 좋게 만드는 일종의 피드백 루프를 이용할 수 있을 것이다.

> In this paper, inspired by the above reasoning, we make the following contributions:
>> 위의 추론에서 영감을 받아 본 논문에서 우리는 다음과 같은 기여를 한다.

* > First, we describe a novel extension to GANs that allows them to learn a generative model and a classifier simultaneously. We call this extension the SemiSupervised GAN, or SGAN.
     >> 먼저, 우리는 생성 모델과 분류기를 동시에 학습할 수 있는 GAN에 대한 새로운 확장을 설명한다. 우리는 이 확장을 Semi Supervised GAN 또는 SGAN이라고 부른다.

* > Second, we show that SGAN improves classification performance on restricted data sets over a baseline classifier with no generative component.
    >> 둘째, 우리는 SGAN이 생성 구성 요소가 없는 기준 분류기를 통해 제한된 데이터 세트에 대한 분류 성능을 향상시킨다는 것을 보여준다.
    
* > Finally, we demonstrate that SGAN can significantly improve the quality of the generated samples and reduce training times for the generator.
    >> 마지막으로, 우리는 SGAN이 생성된 샘플의 품질을 크게 개선하고 발전기의 훈련 시간을 줄일 수 있음을 보여준다.

### $\mathbf{2.\;The\;SGAN\;Model}$

> The discriminator network D in a normal GAN outputs an estimated probability that the input image is drawn from the data generating distribution. Traditionally this is implemented with a feed-forward network ending in a single sigmoid unit, but it can also be implemented with a softmax output layer with one unit for each of the classes [REAL, FAKE]. Once this modification is made, it’s simple to see that D could have N+1 output units corresponding to [CLASS-1, CLASS-2, . . . CLASS-N, FAKE]. In this case, D can also act as C. We call this network D/C.
>> 정규 GAN의 판별기 네트워크 D는 입력 이미지가 데이터 생성 분포로부터 그려질 추정 확률을 출력한다. 전통적으로 이것은 단일 시그모이드 유닛으로 끝나는 피드포워드 네트워크로 구현되지만, 또한 각 클래스 [REAL, FAKE]에 대해 하나의 유닛을 갖는 소프트맥스 출력 레이어로 구현될 수 있다. 일단 이 수정이 이루어지면, D는 [CLASS-1, CLASS-2, .CLASS-N, FAKE]에 해당하는 N+1의 출력 단위를 가질 수 있다. 이 경우, D는 C 역할도 할 수 있다. 우리는 이 네트워크를 D/C라고 부릅니다.

![Algorithm](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-10-(GAN)SGAN-translation/Algorithm-1.JPG)

> Training an SGAN is similar to training a GAN. We simply use higher granularity labels for the half of the minibatch that has been drawn from the data generating distribution. D/C is trained to minimize the negative log likelihood with respect to the given labels and G is trained to maximize it, as shown in Algorithm 1. We did not use the modified objective trick described in Section 3 of Goodfellow et al. (2014).
>> SGAN 훈련은 GAN 훈련과 유사하다. 우리는 단순히 데이터 생성 분포에서 도출된 미니 배치의 절반에 대해 더 높은 세분화 레이블을 사용한다. D/C는 주어진 레이블에 대한 음의 로그 가능성을 최소화하도록 훈련되고 G는 알고리즘 1에 표시된 것처럼 이를 최대화하도록 훈련된다. Goodfellow et al. (2014)의 섹션 3에 설명된 수정된 목표 속임수를 사용하지 않았습니다.

> Note: in concurrent work, (Salimans et al., 2016) propose the same method for augmenting the discriminator and perform a much more thorough experimental evaluation of the technique.
>> 참고: 동시 작업에서 (Salimans et al., 2016)는 판별기를 보강하기 위한 동일한 방법을 제안하고 기술에 대한 훨씬 철저한 실험 평가를 수행한다.

### $\mathbf{3.\;Results}$

> The experiments in this paper were conducted with [https://github.com/DoctorTeeth/supergan](https://github.com/DoctorTeeth/supergan), which borrows heavily from [https://github.com/carpedm20/DCGANtensorflow](https://github.com/carpedm20/DCGANtensorflow) and which contains more details about the experimental setup.
>> 본 논문의 실험은 [https://github.com/carpedm20/DCGANtensorflow](https://github.com/carpedm20/DCGANtensorflow)을 통해 수행되었으며, [https://github.com/carpedm20/DCGANtensorflow](https://github.com/carpedm20/DCGANtensorflow)에서 많은 부분을 차용하고 실험 설정에 대한 자세한 내용을 담고 있다.

#### $\mathbf{3.1.\;Generative\;Results}$

> We ran experiments on the MNIST dataset (LeCun et al., 1998) to determine whether an SGAN would result in better generative samples than a regular GAN. Using an architecture similar to that in Radford et al. (2015), we trained an SGAN both using the actual MNIST labels and with only the labels REAL and FAKE. Note that the second configuration is semantically identical to a normal GAN. Figure 1 contains examples of generative outputs from both GAN and SGAN. The SGAN outputs are significantly more clear than the GAN outputs. This seemed to hold true across different initializations and network architectures, but it is hard to do a systematic evaluation of sample quality for varying hyperparameters.
>> 우리는 SGAN이 일반 GAN보다 더 나은 생성 샘플을 얻을 수 있는지 여부를 결정하기 위해 MNIST 데이터 세트(LeCun et al., 1998)에 대한 실험을 실행했다. Radford 등(2015)과 유사한 아키텍처를 사용하여 실제 MNIST 레이블을 사용하고 REAL과 FAKE 레이블만 사용하여 SGAN을 훈련시켰다. 두 번째 구성은 일반 GAN과 의미적으로 동일하다는 점에 유의하십시오. 그림 1에는 GAN과 SGAN의 생성 출력 예가 수록되어 있다. SGAN 출력은 GAN 출력보다 훨씬 더 명확하다. 이는 다른 초기화 및 네트워크 아키텍처에 걸쳐 적용되는 것처럼 보였지만, 다양한 하이퍼 파라미터에 대해 샘플 품질을 체계적으로 평가하기는 어렵다.

*Table 1*. Classifier Accuracy

|EXAMPLES|CNN|SGAN|
|--------|---|----|
|1000|0.965|0.964|
|100|0.895|0.928|
|50|0.859|0.883|
|25|0.750|0.802|

![Figure 1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-06-10-(GAN)SGAN-translation/Figure-1.JPG)

> Figure 1. Output samples from SGAN and GAN after 2 MNIST epochs. SGAN is on the left and GAN is on the right.
>> 그림 1. 2 MNIST epoch 후 SGAN 및 GAN에서 샘플을 출력합니다. SGAN은 왼쪽에 있고 GAN은 오른쪽에 있습니다.

#### $\mathbf{3.2.\;Classifier\;Results}$

> We also conducted experiments on MNIST to see whether the classifier component of the SGAN would perform better than an isolated classifier on restricted training sets. To train the baseline, we train SGAN without ever updating G. SGAN outperforms the baseline in proportion to how much we shrink the training set, suggesting that forcing D and C to share weights improves data-efficiency. Table 1 includes detailed performance numbers. To compute accuracy, we took the maximum of the outputs not corresponding to the FAKE label. For each model, we did a random search on the learning rate and reported the best result.
>> 우리는 또한 SGAN의 분류기 구성 요소가 제한된 훈련 세트에서 분리된 분류기보다 더 나은 성능을 발휘하는지 알아보기 위해 MNIST에 대한 실험을 수행했다. 기준선을 훈련하기 위해 G를 업데이트하지 않고 SGAN을 훈련한다. SGAN은 훈련 세트를 얼마나 축소하는지에 비례하여 기준선을 능가하며, D와 C가 가중치를 공유하도록 강제하면 데이터 효율성이 향상된다는 것을 시사한다. 표 1에는 자세한 성능 번호가 나와 있습니다. 정확도를 계산하기 위해 FAKE 레이블에 해당하지 않는 출력의 최대값을 취했다. 각 모델에 대해 학습률에 대한 무작위 검색을 실시하여 최상의 결과를 보고했습니다.

### $\mathbf{4.\;Conclusion\;and\;Future\;Work}$

> We are excited to explore the following related ideas:
>> 다음과 같은 관련 아이디어를 살펴보게 되어 기쁘게 생각합니다.

* > Share some (but not all) of the weights between D and C, as in the dual autoencoder (Sutskever et al., 2015). This could allow some weights to be specialized to discrimination and some to classification.
    >> 이중 자동 인코더에서와 같이 D와 C 사이에 무게의 일부(전부는 아님)를 공유한다(Sutskever et al., 2015). 이를 통해 일부 가중치를 차별에 특화하고 일부는 분류에 특화할 수 있다.

* > Make GAN generate examples with class labels (Mirza & Osindero, 2014). Then ask D/C to assign one of 2N labels [REAL-ZERO, FAKE-ZERO, . . . ,REAL-NINE, FAKE-NINE].
    >> GAN이 클래스 레이블을 사용하여 예를 생성하도록 합니다(Mirza & Osindero, 2014). 그런 다음 D/C에 2N 레이블 중 하나를 할당하도록 요청합니다(Real-ZERO, FAKE-ZERO, . , Real-NINE, FAKE-NINE).

* > Introduce a ladder network (Rasmus et al., 2015) L in place of D/C, then use samples from G as unlabeled data to train L with.
    >> D/C 대신 사다리 네트워크(Rasmus et al., 2015) L을 도입한 다음 G의 샘플을 레이블이 없는 데이터로 사용하여 L을 훈련시킵니다.

#### $\mathbf{Acknowledgments}$

> We thank the authors of Tensorflow (Abadi et al., 2016).
>> Tensorflow의 저자들에게 감사한다(Abadi et al., 2016).