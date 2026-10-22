---
layout: post
categories: [1. Computer Engineering, 1. Artificial Intelligence, 2. Deep Learning, 1. Virtual Try-on Network]
title: "HumanDiffusion"
tags: [HumanDiffusion, Text-driven, Paper review]
---

### [CV Paper List](https://maizer2.github.io/2022/08/01/paper-of-VITON.html)

# HumanDiffusion: a Coarse-to-Fine Alignment Diffusion Framework for Controllable Text-Driven Person Image Generation

## Abstract

> Text-driven person image generation is an emerging and challenging task in cross-modality image generation. 
>> 텍스트 기반 개인 이미지 생성(Text-driven person image generation)은 교차 양식 이미지 생성(cross-modality image generatio)에서 새롭게 부상(emerging)하고 도전적인(challenging) 작업이다. 

> Controllable person image generation promotes a wide range of applications such as digital human interaction and virtual try-on. 
>> 제어 가능한 사람 이미지 생성(Controllable person image generation)은 디지털 인간 상호 작용(digital human interaction) 및 가상 트라이온(virtual try-on)과 같은 광범위한 응용 프로그램을 촉진한다(promotes). 

> However, previous methods mostly employ singlemodality information as the prior condition (e.g. poseguided person image generation) or utilize the preset words for text-driven human synthesis. 
>> 그러나 이전 방법은 대부분 단일 양식 정보(single-modality information)를 사전 조건(prior condition)(예: 포즈 안내 사람 이미지 생성(guided person image generation))으로 사용하거나 텍스트 기반 인간 합성(text-driven human synthesis)을 위해 사전 설정된 단어(preset words)를 활용한다. 

> Introducing a sentence composed of free words with an editable semantic pose map to describe person appearance is a more user-friendly way.
>> 사람의 외모를 설명하기(describe person appearance) 위해 편집 가능한 의미 포즈 맵(editable semantic pose map)과 함께 자유 단어(free words)로 구성된 문장을 도입하는 것이 더 사용자 친화적인 방법이다.

> In this paper, we propose HumanDiffusion, a coarse-to-fine alignment diffusion framework, for text-driven person image generation. 
>> 본 논문에서는 텍스트 중심 인물 이미지 생성(text-driven person image generation)을 위한 거칠고 미세한 정렬 확산 프레임워크(coarse-to-fine alignment diffusion framework)인 HumanDiffusion을 제안한다. 

> Specifically, two collaborative modules are proposed, the Stylized Memory Retrieval (SMR) module for fine-grained feature distillation in data processing and the Multi-scale Cross-modality Alignment (MCA) module for coarse-to-fine feature alignment in diffusion. 
>> 본 논문에서는 텍스트 중심 인물 이미지 생성(text-driven person image generation)을 위한 거칠고 미세한 정렬 확산 프레임워크(coarse-to-fine alignment diffusion framework)인 HumanDiffusion을 제안한다. 

> These two modules guarantee the alignment quality of the text and image, from image-level to feature-level, from low-resolution to high-resolution. 
>> 구체적으로, 데이터 처리에서 미세한 특징 증류(fine-grained feature distillation in data processing)를 위한 양식화된 메모리 검색(Stylized Memory Retrieval)(SMR) 모듈과 확산에서 거칠고 미세한 특징 정렬(coarse-to-fine feature alignment in diffusion)을 위한 다중 스케일 교차 양식 정렬(Multi-scale Cross-modality Alignment)(MCA) 모듈의 두 가지 협업 모듈이 제안된다. 

> As a result, HumanDiffusion realizes open-vocabulary person image generation with desired semantic poses. 
>> 결과적으로, 휴먼 확산(HumanDiffusion)은 원하는 의미론적 포즈(desired semantic poses)로 열린 어휘 사용자 이미지 생성(open-vocabulary person image generation)을 실현한다. 

> Extensive experiments conducted on  DeepFashion demonstrate the superiority of our method compared with previous approaches.
>> DeepFashion에 대해 수행된 광범위한 실험은 이전 접근 방식과 비교하여 우리 방법의 우수성을 입증한다.

> Moreover, better results could be obtained for complicated person images with various details and uncommon poses.
>> 또한, 다양한 세부 사항(various details)과 흔치 않은 포즈(uncommon poses)를 가진 복잡한 사람 이미지(complicated person images)에 대해 더 나은 결과를 얻을 수 있다.

## 1. Introduction

* 저자 : Kaiduo Zhang, Muyi Sun, Jianxin Sun, Binghao Zhao, Kunbo Zhang, Zhenan Sun, Tieniu Tan (CASIA)
* [arXiv:2211.06235](https://arxiv.org/abs/2211.06235), 2022년 11월

### 문제 설정

* **Text-driven person image generation** : 사람의 외형을 설명하는 자유로운 문장(예: "a woman wearing a red long-sleeve shirt and blue jeans")과 **semantic pose map**(신체 부위별 파싱 맵)을 입력받아, 그 자세를 가진 채 문장대로 입은 사람 이미지를 생성한다.
* 기존 접근의 한계
    * **Pose-guided generation**(PATN, ADGAN 등)은 조건이 포즈 하나뿐이라 옷의 종류·색을 자유롭게 지정할 수 없다.
    * **Text-driven 합성** 중 상당수는 미리 정해진 속성 단어(preset words)만 받아 open-vocabulary 문장을 처리하지 못한다.
    * GAN 기반 방법은 세부 질감과 흔치 않은 포즈에서 품질이 떨어진다.

### 핵심 아이디어 : coarse-to-fine alignment

* 텍스트와 이미지의 정렬(alignment)을 **두 수준**에서 수행한다.
    1. **이미지 수준 (SMR)** : 데이터 처리 단계에서 문장과 가장 잘 맞는 실제 옷 이미지를 검색해 "스타일 메모리"로 가져온다. 문장이 주지 못하는 미세한 질감 정보를 실제 이미지에서 증류한다.
    2. **특징 수준 (MCA)** : 확산 과정 안에서 텍스트·메모리 특징과 U-Net 특징을 여러 해상도에서 cross-attention으로 정렬한다. 저해상도에서는 전체 레이아웃, 고해상도에서는 세부 질감이 맞춰진다.

## 2. Method

### 2.1 Overview

* 백본은 조건부 latent diffusion(U-Net)이다. 입력 조건은 (a) 텍스트 문장, (b) semantic pose map, (c) SMR로 검색한 스타일 메모리 이미지의 특징 세 가지다.
* Semantic pose map은 U-Net 입력에 채널로 연결(concat)해 공간 구조를 강하게 제약하고, 텍스트와 스타일 메모리는 cross-attention으로 주입한다.

### 2.2 Stylized Memory Retrieval (SMR)

* 문장의 각 의복 구절(상의, 하의, 신발 등)에 대해, DeepFashion 데이터베이스에서 **CLIP 기반 text-image retrieval**로 가장 유사한 의복 영역 이미지를 찾는다.
* 검색된 이미지에서 의복 영역만 잘라 스타일 메모리로 만들고, 이미지 인코더로 fine-grained 특징을 추출해 확산 모델의 조건으로 넘긴다.
* 효과 : 텍스트만으로는 결정되지 않는 무늬·소재·세부 색을 실제 사진에서 가져오므로, 생성 결과가 "그럴듯한 평균"이 아니라 구체적인 질감을 가진다. Open-vocabulary 입력도 검색이 가능하기만 하면 처리된다.

### 2.3 Multi-scale Cross-modality Alignment (MCA)

* U-Net의 여러 해상도 블록마다 cross-attention 모듈을 두고, query는 이미지 특징, key/value는 텍스트 토큰 특징과 스타일 메모리 특징으로 구성한다.
* 저해상도 블록에서는 문장 전체와 포즈가 결정하는 **전역 배치**(어느 부위에 어떤 옷)가, 고해상도 블록에서는 스타일 메모리가 제공하는 **국소 질감**이 주로 반영되도록 scale별로 정렬이 이루어진다.
* 이 다중 scale 정렬 덕분에 텍스트와 이미지의 대응이 image-level에서 feature-level까지, low-resolution에서 high-resolution까지 일관되게 유지된다.

### 2.4 학습

* 표준 DDPM 노이즈 예측 손실을 사용하며, 조건은 모두 노이즈 예측 네트워크에 함께 입력된다.
* 추론 시에는 classifier-free guidance 방식으로 조건의 영향력을 조절한다.

## 3. Experiments

* **데이터셋** : DeepFashion(In-shop Clothes Retrieval). 이미지마다 사람 파싱 맵과 속성·캡션 텍스트가 있어 (문장, 포즈 맵, 이미지) 삼중쌍을 만들 수 있다.
* **비교 대상** : 포즈 조건 GAN 계열(ADGAN 등)과 텍스트 조건 생성 모델 계열.
* **평가** : FID, 그리고 텍스트-이미지 일치도(CLIP score 또는 속성 정확도), 사용자 평가.
* **결과**
    * 정량적으로 기존 방법보다 낮은 FID와 높은 텍스트 일치도를 보고한다.
    * 정성적으로 **복잡한 세부(무늬, 레이어드 의상)와 흔치 않은 포즈**에서 GAN 기반 방법보다 왜곡이 적다.
    * **Ablation** : SMR을 빼면 질감이 뭉개지고 open-vocabulary 입력에 약해지며, MCA를 단일 scale로 줄이면 텍스트와 생성 결과의 부위별 대응이 흐트러진다.
    * 같은 포즈 맵에 문장만 바꾸거나, 같은 문장에 포즈 맵을 편집하는 식으로 **외형과 자세를 독립적으로 제어**할 수 있음을 보여 준다.

## 4. 정리 및 Virtual Try-on 관점의 의의

* **기여** : 사람 이미지 생성에 확산 모델을 적용하면서, 텍스트의 모호함을 **검색된 실제 의복 이미지(SMR)**로 보완하고 **다중 scale cross-attention(MCA)**으로 정렬하는 설계를 제시했다.
* **VTON과의 연결** : SMR이 가져오는 의복 이미지를 사용자가 직접 지정하면 곧바로 try-on 문제가 된다. 즉 HumanDiffusion은 "텍스트 → 옷 검색 → 입히기"로 이어지는 text-driven virtual try-on의 초기 형태다. 같은 시기의 [PIDM](https://maizer2.github.io/1.%20computer%20engineering/2023/02/20/Person-Image-Synthesis-via-Denoising-Diffusion-Model.html)이 포즈 전이에 집중했다면, 이 논문은 텍스트 제어에 집중한다.
* **한계** : 검색 데이터베이스(DeepFashion)에 없는 스타일은 표현하기 어렵고, 검색 품질이 결과를 좌우한다. 확산 모델 특성상 추론이 느리며, 포즈 맵을 사람이 편집해야 하는 부담이 있다.
