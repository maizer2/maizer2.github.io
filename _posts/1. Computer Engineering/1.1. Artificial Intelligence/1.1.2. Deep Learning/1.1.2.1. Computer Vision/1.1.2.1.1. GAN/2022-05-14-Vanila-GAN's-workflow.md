---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.1. Computer Vision, 1.1.2.1.1. GAN]
title: Vanila GAN's workflow
tags: [1.1.2.1. Computer Vision, 1.1.2. Deep Learning, 1.1. Artificial Intelligence]
---

### Vanila GAN's workflow

훈련하고자하는 훈련 데이터를 통해 학습된 생성 모델에 정규 분포를 따르는 잠재 공간의 잠재 벡터를 샘플링하여 샘플을 생성해 낸다.

$$Trained\; Ganerator(Latent Vector) = G(z)$$

생성기를 통해 생성된 데이터와 훈련 데이터