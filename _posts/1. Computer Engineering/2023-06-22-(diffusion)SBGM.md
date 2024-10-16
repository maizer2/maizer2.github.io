---
layout: post 
title: "(Diffusion)Improved Techniques for Training Score-Based Generative Models Review"
categories: [1. Computer Engineering]
tags: [1.2.2.1. Computer Vision, 1.7. Paper Review]
---

### [CV Paper List](https://maizer2.github.io/1.%20computer%20engineering/2023/02/01/paper-of-diffusion.html)

# Improved Techniques for Training Score-Based Generative Models

## Abstract

데이터 분포의 점수 매칭을 통해 추정된 기울기를 사용하여 랑주뱅 동역학<sup>Langevin dynamics</sup>을 통해 샘플을 생성하는 새로운 생성 모델을 소개한다.

데이터가 저차원 매니폴드에 위치할 때 기울기는 잘 정의되지 않고 추정하기 어려울 수 있다.

우리는 데이터를 다른 수준의 가우시안 노이즈로 동요시키고 해당하는 점수, 즉 동요된 데이터 분포의 기울기 벡터 필드를 모든 노이즈 수준에 대해 공동으로 추정한다.

샘플링을 위해, 우리는 점차적으로 감소하는 노이즈 수준에 해당하는 기울기를 사용하여 앤닐드 랑지반 동역학<sup>annealed Langevin dynamics</sup>을 제안한다.