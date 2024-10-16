---
layout: post 
title: "(Diffusion)Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis"
categories: [1. Computer Engineering]
tags: [1.2.2.1. Computer Vision, 1.7. Paper Review]
---

### [CV Paper List](https://maizer2.github.io/1.%20computer%20engineering/2023/02/01/paper-of-diffusion.html)

# Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis

## Abstract

기존의 Diffusion 모델들은 high-quality image synthesis에서 높은 성능을 보여주었다.

하지만 이미지에 크기가 다른 객체가 포함될 때, 쉽게말해 복잡한 이미지에서, 객체가 정확하게 생성하지 못한다.

본 논문에서는 이를 해결하기 위해 Unet 모델에 Feature Pyramid<sup>[FPN](https://maizer2.github.io/1.%20computer%20engineering/2023/06/12/(object-detection)FPN.html)</sup> 구조를 추가하여 course-to-fine feature를 학습한다.

