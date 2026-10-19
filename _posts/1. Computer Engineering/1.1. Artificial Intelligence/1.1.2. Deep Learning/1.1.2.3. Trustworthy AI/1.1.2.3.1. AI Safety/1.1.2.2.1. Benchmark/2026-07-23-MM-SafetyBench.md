---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.3. Trustworthy AI, 1.1.2.3.1. AI Safety, 1.1.2.2.1. Benchmark]
title: "MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models"
tags: [Benchmark, Dataset, VLM, AI Safety, Paper review]
---

# Review Summary

## Basic information

- **Published** : ECCV 2024
- **Code / Project page** : GitHub `https://github.com/isXinLiu/MM-SafetyBench`

## One-line summary

이 논문은 Multimodal Large Language Models(MLLMs)에서 질문(Text-modality)과 관련된 이미지(Image-modality)를 함께 모델에 입력하면 모델의 안전정렬이 쉽게 무력화되어 위험한 답변을 유출하게 된다는 사실을 발견하였고, 이를 평가할 수 있는 멀티모달 벤치마크 MM-SafetyBench를 공개하고, 논문에서 제시한 새로운 공격 방법(무해한 지시 텍스트와 유해 이미지를 사용하는 텍스트 필터 우회 공격)을 12개의 최신 모델에게 벤치마크를 사용하여 평가한 결과 안전정렬이 쉽게 무력화됨을 보여주고, 더 나아기 이를 대응하기 위해 효과적인 프롬프트 전략(Safety Prompt, 안전 프롬프트 기법, 사용자의 원래 질의 바로 앞에 매우 짧고 명확한 지시문 하나를 덧붙이는 방식)을 제시하였다.

## Benchmark dataset overview

* 13 scenarios
* total of 5,040 text-image pairs