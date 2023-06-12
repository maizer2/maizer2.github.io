---
layout: post
title: "(VITON)HumanDiffusion"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.7. Paper Review]
---

### [VITON Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/08/01/Paper-of-VITON.html)

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

