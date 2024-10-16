---
layout: post 
title: "(Diffusion)3D Neural Field Generation using Triplane Diffusion Review"
categories: [1. Computer Engineering]
tags: [1.2.2.1. Computer Vision, 1.7. Paper Review]
---

### [CV Paper List](https://maizer2.github.io/1.%20computer%20engineering/2023/02/01/paper-of-diffusion.html)

# 3D Neural Field Generation using Triplane Diffusion

![Figure-2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2023-06-23-(diffusion)NFD/Figure-2.PNG)

## Abstract

본 논문은 3D-aware generation of neural fiels에 Diffusion 모델을 적용시킨다.

* 3D-aware generation은 생성 과정에서 3차원 정보를 고려하여 생성하는 방법입니다.

* Neural fiels는 신경망의 출력 값으로 이뤄진 공간적인 데이터 구조를 의미한다. 여기서 공간적인 데이터 구조는 출력값의 2D 혹은 3D 공간을 의미한다.

## 1. Introduction

3D-aware generative adversarial networks (GANs)의 생성 방법을 Diffusion model을 사용하여 성능을 개선시켰다.

3D-aware GANs는 2D generator를 사용하여 높은 성능의 3D shape 생성에 성공하였다.

본 논문은 3D-aware GANs의 아이디어를 발전시켜, 3D scenes과 radiance fields를 축에 정렬<sup>axis-aligned</sup>된 2D feature 평면의 집합으로 인코딩하는 "triplane representations"를 생성하는 것을 학습하는 것을 제안한다.

* Triplane은 3D 데이터를 표현하기 위해 세 개의 평면으로 구성된 표현 방법을 의미하며 그림에서 확인 할 수 있다.
본 논문에서 소개하는 Neural field-based diffusion framework for 3D representation learning은 두 단계로 진행된다.

1. 3D scenes 훈련 데이터셋은 장면별로 triplane features 세트와 단일 shared feature 디코더로 분해된다.

2. 분해된 triplane에 대해 2D Diffusion 모델을 훈련합니다. 

훈련된 Diffusion 모델은 추론 시에 새롭고 다양한 3D 장면을 생성하는 데 사용될 수 있다.

Triplane을 다중 채널 2D 이미지로 해석하여 생성과 렌더링을 분리함으로써, 현재의 (그리고 미래의) state-of-art 2D Diffusion 모델 백본을 거의 그대로 활용할 수 있다.