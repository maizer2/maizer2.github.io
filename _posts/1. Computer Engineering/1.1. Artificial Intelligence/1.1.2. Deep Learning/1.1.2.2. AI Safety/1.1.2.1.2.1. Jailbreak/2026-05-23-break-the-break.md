---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.2. AI Safety, 1.1.2.1.2.1. Jailbreak]
title: "Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization"
tags: [Jailbreak, Jailbreak Attack, VLM, AI Safety, Safety Alignment, Transferability, Paper review]
---

# Review Summury

* Basic information : [11 May 2026][Arxiv] [Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization](https://arxiv.org/abs/2605.10764)
* One-line summary : VLM에서 Gradient-based 이미지 공격의 범용성을 높이기 위해, Decision tokens의 high-entropy를 극대화하여 거부 결과를 뒤집는 동시에, 나머지 low-entropy를 안정화하여 출력 품질을 유지하는 UJEM-KL(Untargeted Jailbreak via Entropy Maximization)을 제안한다.
* Key Contribution :
* Advantages & Learnings :
* Limitations & Questions :
    * 1. 범용성을 높이기 위한 실험이였다면, white-box 모델 외에 black-box 모델 실험 결과도 공개해야하지 않는가?
* Action Items :
    RQ 1. 

---

# Abstract

* 문제 제기 : 최근 연구되고 있는 <abbr title="VLM의 내부 수학적 구조를 역이용하여, 모델의 안전장치를 무력화하는 특정한 이미지 노이즈(적대적 섭동)를 생성하는 공격 방식">Gradient-based universal image jailbreak attack</abbr>은 모델 간 <abbr title="화이트박스 모델을 대상으로 생성한 adversarial image를 다른 모델에 사용했을 때 jailbreak attack이 성공한다면, 이를 전이성이 높다고 한다.">transferability(전이성)</abbr>이 거의 없거나 전혀 나타나지 않아, transferable(전이 가능한) multimodal jailbreak의 가능성에 의구심을 던진다.



