---
layout: post
categories: [1. Computer Engineering, 1. Artificial Intelligence, 2. Deep Learning, 0. General]
title: "Instance Normalization: The Missing Ingredient for Fast Stylization"
tags: [Instance Normalization, Style Transfer, Paper review]
---

### [Normalization Literature List](https://maizer2.github.io/2022/10/04/paper-of-Normalization.html)

---

### Instance Normalization: The Missing Ingredient for Fast Stylization [https://arxiv.org/abs/1607.08022](https://arxiv.org/abs/1607.08022)

* 저자 : Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky (Skoltech, Oxford)
* 2016년 arXiv 테크니컬 리포트. 이후 CVPR 2017 "Improved Texture Networks"에 통합되었다.

## 한 줄 요약

Feed-forward 스타일 변환 네트워크(Ulyanov et al.의 Texture Networks, Johnson et al.의 Perceptual Loss 네트워크)의 generator에서 **batch normalization을 instance normalization으로 바꾸는 것만으로** 스타일 변환 품질이 Gatys의 느린 최적화 방식에 가까워진다는 것을 보인 짧은 논문이다.

## 1. 배경 : Fast stylization의 품질 문제

* Gatys et al.(2016)의 neural style transfer는 이미지마다 최적화를 돌려 느리지만 품질이 좋다.
* Ulyanov et al.(2016)과 Johnson et al.(2016)은 generator 네트워크를 학습해 한 번의 forward로 스타일 변환을 하는 **fast stylization**을 제안했으나, 품질이 Gatys 방식보다 눈에 띄게 떨어졌다.
* 저자들은 그 원인이 네트워크 구조나 손실이 아니라 **generator 안의 batch normalization**에 있다고 진단했다.

## 2. 핵심 관찰 : 스타일 변환 결과는 콘텐츠 이미지의 contrast와 무관해야 한다

* 스타일 변환의 목표는 출력의 질감·색 분포(스타일)가 스타일 이미지를 따르는 것이다. 따라서 **콘텐츠 이미지의 전체적인 밝기·대비 정보는 결과에 남아 있으면 안 된다.**
* 간단한 실험으로, 콘텐츠 이미지의 contrast를 바꾸어 넣으면 Gatys 방식은 거의 같은 결과를 내지만 batch norm generator는 결과가 크게 달라진다. 즉 generator가 "contrast를 버리는 법"을 배우지 못하고 있었다.
* Contrast normalization은 각 **이미지 한 장**의 통계로 정규화해야 한다. Batch normalization은 미니배치 전체의 통계를 쓰므로 이미지 하나의 contrast를 제거하지 못한다.

## 3. Instance Normalization

입력 텐서를 $x \in \mathbb{R}^{T\times C\times W\times H}$ (배치 $T$, 채널 $C$, 공간 $W\times H$)라 하자.

**Batch normalization**은 채널 $c$마다 배치와 공간 전체에 대해 통계를 낸다.

$$ \mu_c = \frac{1}{THW}\sum_{t=1}^{T}\sum_{w=1}^{W}\sum_{h=1}^{H} x_{tcwh}, \qquad \sigma_c^2 = \frac{1}{THW}\sum_{t,w,h}(x_{tcwh}-\mu_c)^2 $$

**Instance normalization**(contrast normalization)은 **샘플 $t$와 채널 $c$ 쌍마다** 공간 차원에 대해서만 통계를 낸다.

$$ \mu_{tc} = \frac{1}{HW}\sum_{w=1}^{W}\sum_{h=1}^{H} x_{tcwh}, \qquad \sigma_{tc}^2 = \frac{1}{HW}\sum_{w,h}(x_{tcwh}-\mu_{tc})^2 $$

$$ y_{tcwh} = \frac{x_{tcwh} - \mu_{tc}}{\sqrt{\sigma_{tc}^2 + \epsilon}} $$

* 배치 안의 다른 샘플에 의존하지 않으므로 **학습과 추론에서 동작이 동일**하다. Batch norm처럼 running 평균·분산을 따로 둘 필요가 없다.
* Generator의 모든 conv layer 뒤에서 batch norm을 instance norm으로 교체하기만 하면 된다. 추가 파라미터나 손실 변경은 없다.

## 4. 실험

* Ulyanov et al.의 Texture Networks generator와 Johnson et al.의 generator 두 가지에 각각 BN과 IN을 적용해 비교한다.
* **정성 결과** : IN을 쓴 generator의 출력은 질감이 선명하고 콘텐츠 이미지의 조명·대비에 따른 얼룩이 사라져 Gatys 방식과 시각적으로 비슷해진다. BN 결과에서 보이던 흐릿함과 색 번짐이 줄어든다.
* **학습 곡선** : IN을 쓰면 style loss가 더 낮은 값으로 더 빨리 수렴한다. 네트워크가 contrast 제거에 용량을 쓰지 않아도 되기 때문이라고 해석한다.
* 흥미롭게도 **배치 크기를 1로 둔 BN**(사실상 IN과 유사)도 큰 배치 BN보다 좋았다. 이는 품질 차이가 통계를 내는 단위에서 온다는 것을 뒷받침한다.

## 5. 의의와 이후 영향

* 4쪽짜리 리포트이지만 생성 모델 정규화의 방향을 바꾸었다. 이후 스타일 변환과 image-to-image translation(Pix2Pix, CycleGAN 등) generator는 거의 예외 없이 IN을 쓴다.
* IN의 평균·분산이 "스타일"을 담는다는 관찰은 **Conditional IN**(Dumoulin et al., 여러 스타일을 affine 파라미터로 전환)과 **[AdaIN](https://maizer2.github.io/1.%20computer%20engineering/2022/07/07/Arbitrary-Style-Transfer-in-Real-time-with-Adaptive-Instance-Normalization-Translation.html)**(Huang & Belongie, 스타일 이미지의 통계로 직접 정규화)으로 이어지며, StyleGAN의 스타일 주입 방식으로까지 발전한다.
* **한계** : 콘텐츠의 전역 대비 정보가 정말로 필요한 과제(예: 사진 보정)에서는 IN이 정보를 버려 해가 될 수 있다. 또 공간 통계만 쓰므로 공간 크기가 매우 작은 feature map에서는 통계가 불안정하다.
