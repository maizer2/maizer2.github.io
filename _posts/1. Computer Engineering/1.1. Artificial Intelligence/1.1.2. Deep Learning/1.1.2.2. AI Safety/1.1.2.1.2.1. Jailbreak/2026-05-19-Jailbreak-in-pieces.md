---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.2. AI Safety, 1.1.2.1.2.1. Jailbreak]
title: "Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models"
tags: [Jailbreak, VLM, AI Safety, Safety Alignment, Paper review]
---

# Review Summury

* Basic information : [12 Mar 2024][ICLR 2024 spotlight] [Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models](https://openreview.net/pdf?id=plmBsXHxgR)
* One-line summary : 
* Key Contribution :
* Advantages & Learnings :
* Limitations & Questions :
* Action Items :
    RQ. 꼭 benign-appearing adversarial images 여야 할까? Harmful-appearing adversarial images 이면 더 잘 jailbreak되지 않을까?

---

# 핵심 Abstract 내용

* 새로운 VLM jailbreak attacks(cross-modality attacks)를 소개함
    * Cross-modality Attack 이란?
        * Compositional Strategy(Adversarial image와 benign textual prompt를 결합)을 사용하여 언어 모델의 정렬을 깸.
            * Benign textual prompt는 이미지(Adversarial Image)를 지칭하는 문장으로, benign함
            * Adversarial image는 benign한 외형의 harmful한 context가 포함되도록 adversarial attack noise가 추가된 이미지로, harmful하지만 약한 vision encoder의 정렬을 무너뜨림
    

