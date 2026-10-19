---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.3. Trustworthy AI, 1.1.2.3.2. Mechanistic Interpretability]
title: "Unsupervised Domain Clusters in Pretrained Language Models"
tags: [Unsupervised Clustering, Pretrained Language Models, Domain Adaptation, Domain Data Selection, Representation Analysis, BERTology, Positive-Unlabeled Learning, Latent Space Geometry, Neural Machine Translation, Paper review]
---

# 0. Summary

- 기본 정보 : ACL 2020, **[Unsupervised Domain Clusters in Pretrained Language Models](https://arxiv.org/pdf/2004.02105)**
- 연구 배경 : 도메인 경계가 모호하고 라벨링된 데이터가 부족하여 특정 도메인 모델 구축이 어려움
- 발견 사실 : 사전학습 언어 모델의 문장 임베딩 공간에서 텍스트가 비지도 방식으로 도메인별 클러스터링을 형성함
- 적용 및 성과 : 소량의 데이터만으로 효과적인 도메인 데이터 선별 기법을 제안하여 기계 번역 벤치마크에서 기존 기준선 대비 성능 개선