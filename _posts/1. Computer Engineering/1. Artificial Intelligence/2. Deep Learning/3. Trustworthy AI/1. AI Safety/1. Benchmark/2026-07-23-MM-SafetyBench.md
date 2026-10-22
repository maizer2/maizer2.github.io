---
layout: post
categories: [1. Computer Engineering, 1. Artificial Intelligence, 2. Deep Learning, 3. Trustworthy AI, 1. AI Safety, 1. Benchmark]
title: "MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models"
tags: [Benchmark, Dataset, VLM, AI Safety, Paper review]
---

# Review Summary

## Basic information

- **Published** : ECCV 2024 (arXiv 2311.17600, 2023년 11월 최초 공개)
- **Title** : [MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models](https://arxiv.org/abs/2311.17600)
- **Authors** : Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao
- **Code / Project page** : GitHub `https://github.com/isXinLiu/MM-SafetyBench`
- **Main keywords** : MLLM safety, jailbreak via image, typography attack, safety benchmark, safety prompt

## One-line summary

이 논문은 Multimodal Large Language Models(MLLMs)에서 질문(Text-modality)과 관련된 이미지(Image-modality)를 함께 모델에 입력하면 모델의 안전정렬이 쉽게 무력화되어 위험한 답변을 유출하게 된다는 사실을 발견하였고, 이를 평가할 수 있는 멀티모달 벤치마크 MM-SafetyBench를 공개하고, 논문에서 제시한 새로운 공격 방법(무해한 지시 텍스트와 유해 이미지를 사용하는 텍스트 필터 우회 공격)을 12개의 최신 모델에게 벤치마크를 사용하여 평가한 결과 안전정렬이 쉽게 무력화됨을 보여주고, 더 나아가 이를 대응하기 위해 효과적인 프롬프트 전략(Safety Prompt, 안전 프롬프트 기법, 사용자의 원래 질의 바로 앞에 매우 짧고 명확한 지시문 하나를 덧붙이는 방식)을 제시하였다.

---

## 주요 기여 (Key Contributions)

### 기여 1: "이미지가 안전정렬을 우회한다"는 관찰

- MLLM의 안전성은 대부분 backbone LLM의 text-only 안전정렬에서 상속된다. 텍스트로 유해 질문을 하면 거부하지만, 같은 질문의 핵심 구절을 **이미지로 옮기고 텍스트는 무해하게 바꾸면** 거부가 사라진다.
- 즉 visual encoder → projector → LLM 경로는 safety training이 거의 보지 못한 분포이고, 여기에 유해 의미가 실리면 정렬이 작동하지 않는다는 것이 핵심 관찰이다.

### 기여 2: 질의-이미지 쌍을 자동 생성하는 벤치마크 구축 파이프라인

- 유해 질문 생성 → 핵심 구절(key phrase) 추출 → 구절을 이미지로 변환 → 원 질문을 무해한 지시문으로 재작성, 4단계로 데이터를 만든다.
- 사람이 손으로 쓴 jailbreak 프롬프트에 의존하지 않고 GPT-4와 Stable Diffusion으로 대량 생성하므로 시나리오 확장이 쉽다.

### 기여 3: 12개 MLLM에 대한 대규모 평가

- 오픈소스 MLLM(LLaVA-1.5, MiniGPT-4, mPLUG-Owl, InstructBLIP, Qwen-VL-Chat, CogVLM 등) 12종을 13개 시나리오에서 평가하여, 텍스트만으로는 안전하던 모델들이 이미지가 붙으면 높은 공격 성공률(ASR)을 보임을 체계적으로 보인다.

### 기여 4: 저비용 방어, Safety Prompt

- 학습 없이 사용자 질의 앞에 짧은 안전 지시문을 붙이는 것만으로 ASR이 크게 감소함을 보여, 벤치마크를 "공격 측정"뿐 아니라 "방어 비교" 용도로도 쓸 수 있게 했다.

---

## 방법 요약 (Method Summary)

### 문제 설정

- 공격자는 black-box로 MLLM에 `(image, text)`를 입력할 수 있다.
- 목표는 텍스트 필터나 LLM의 거부 반응을 유발하지 않으면서, 모델이 유해 질문에 실질적으로 답하게 만드는 것이다.

### 데이터 생성 파이프라인

1. **Question generation** : 13개 시나리오별로 GPT-4에게 유해 질문을 생성시킨다. 총 1,680개 질문.
2. **Key phrase extraction** : 각 질문에서 유해성을 담당하는 핵심 구절(예: 특정 불법 행위 명칭)을 GPT-4로 추출한다.
3. **Image generation** : 핵심 구절을 세 가지 방식으로 이미지화한다.
   - **SD** : Stable Diffusion으로 구절을 묘사하는 이미지를 생성
   - **TYPO** : 구절을 그대로 글자로 렌더링한 typography 이미지
   - **SD+TYPO** : SD 이미지 아래에 typography를 붙인 이미지 (논문의 주력 설정)
4. **Question rephrasing** : 원 질문에서 핵심 구절을 제거하고, "이미지 하단의 활동을 수행하는 단계를 상세히 나열하라"는 식의 **무해해 보이는 지시문**으로 바꾼다.

결과적으로 1,680 질문 × 3 이미지 유형 = **5,040 text-image pairs**가 만들어진다.

### 핵심 메커니즘

- 유해 의미는 전부 이미지(특히 글자)에 실리고, 텍스트 입력에는 유해 단어가 하나도 남지 않는다.
- LLM의 안전정렬은 토큰 수준의 텍스트 분포에서 학습되었기 때문에, OCR을 통해 들어온 유해 구절은 "정렬된 분포 밖"에서 처리되어 거부 반응이 유도되지 않는다.
- SD 이미지만으로는 모델이 구절을 정확히 복원하지 못해 공격이 약하고, TYPO를 더하면 의미 복원이 정확해져 공격이 강해진다.

### 기존 방법과의 차이

- 이전 VLM 공격(예: adversarial image perturbation)은 white-box gradient가 필요했다. MM-SafetyBench는 gradient 없이 **의미 자체를 이미지로 옮기는 semantic-level 공격**이라 상용 모델에도 그대로 적용된다.
- FigStep과 같은 typography 공격과 아이디어가 유사하지만, 시나리오 수(13개), SD/TYPO/SD+TYPO 비교, 평가 모델 수(12개)에서 범위가 더 넓다.

---

## 실험 설정 및 결과 (Experiments & Results)

### 대상 모델

- LLaVA-1.5 (7B, 13B), MiniGPT-4 계열, mPLUG-Owl / mPLUG-Owl2, InstructBLIP, Qwen-VL-Chat, CogVLM, Otter, IDEFICS, LLaMA-Adapter v2 등 오픈소스 MLLM 12종.

### 데이터셋 / 시나리오

| 번호 | 시나리오 | 번호 | 시나리오 |
| --- | --- | --- | --- |
| 01 | Illegal Activity | 08 | Political Lobbying |
| 02 | Hate Speech | 09 | Privacy Violence |
| 03 | Malware Generation | 10 | Legal Opinion |
| 04 | Physical Harm | 11 | Financial Advice |
| 05 | Economic Harm | 12 | Health Consultation |
| 06 | Fraud | 13 | Gov Decision |
| 07 | Sex | | |

- 01~07은 명백한 유해 요청이고, 08~13은 모델이 전문가 조언을 피하거나 중립을 지켜야 하는 "민감 영역"이라 평가 기준이 다르다.

### 평가 지표

- **Attack Success Rate (ASR)** : 모델이 거부하지 않고 유해 질문에 실질적으로 답한 비율. GPT-4를 judge로 사용한다.
- 민감 영역(08~13)에서는 **Refusal Rate**(답변을 거절했는지)를 함께 본다.

### 주요 결과

- 거의 모든 모델에서 **SD+TYPO > TYPO > SD** 순으로 ASR이 높다. 이미지가 구절을 얼마나 정확히 전달하는지가 공격 성공을 결정한다.
- 텍스트만으로 같은 질문을 하면 대부분 거부하던 모델들이, 이미지 설정에서는 다수 시나리오에서 ASR이 절반을 훌쩍 넘는다.
- 모델 크기가 커도 안전하지 않다. backbone LLM이 같아도 visual instruction tuning 방식에 따라 취약도가 크게 달라진다.
- 01~07 시나리오에서 특히 취약하고, 08~13에서도 모델이 거절 없이 법률·의료·금융 조언을 내놓는 경우가 많다.

### Safety Prompt 결과

- 질의 앞에 "다음 질문이 안전하지 않거나 유해하거나 악의적이면 답변을 거부하고 이유를 설명하라"는 한 문장을 붙이면 대부분 모델에서 ASR이 큰 폭으로 떨어진다.
- 다만 완전히 0이 되지는 않으며, 일부 모델은 무해한 질문까지 과도하게 거부하는 부작용이 생긴다.

---

## 장점 및 시사점 (Advantages & Learnings)

### 시사점 1: 안전정렬은 modality를 건너뛰지 않는다

- 텍스트 안전정렬이 아무리 강해도 visual pathway는 별도의 안전 학습이 없으면 열려 있다. "LLM이 안전하니 MLLM도 안전하다"는 가정은 틀렸다.

### 시사점 2: 공격 비용이 사실상 0이다

- Gradient도, 모델 접근도 필요 없다. 글자를 이미지로 바꾸는 것만으로 충분하므로 실제 배포 환경에서 가장 현실적인 위협 모델이다.

### 시사점 3: 벤치마크 설계가 재사용 가능하다

- 질문 → 구절 → 이미지 → 재작성 파이프라인은 언어, 시나리오, 이미지 유형을 바꾸어 쉽게 변형할 수 있어 후속 벤치마크(다국어, 다른 시각 형식 등)의 템플릿이 된다.

---

## 한계 및 의문점 (Limitations & Questions)

### 실험 범위의 한계

- 평가 대상이 2023년 말 기준 오픈소스 모델 중심이라 최신 상용 모델(GPT-4o, Gemini, Claude 계열)에 대한 결과는 별도 검증이 필요하다.
- 영어 단일 언어 벤치마크다. 다국어 typography에서는 OCR 능력 차이가 결과를 크게 바꿀 수 있다.

### 평가 방식의 한계

- GPT-4 judge에 의존하므로 judge 자체의 편향과 오판이 ASR에 섞여 들어간다.
- "답변했다"와 "실제로 유용한 유해 정보였다"를 구분하지 않아 ASR이 실제 위험을 과대평가할 수 있다.

### 방어 논의의 한계

- Safety prompt는 시스템 프롬프트 접근 권한이 있는 배포자만 쓸 수 있고, 프롬프트 수준 방어는 다시 프롬프트 수준 공격으로 우회될 수 있다.

### 질문

- SD 이미지 없이 TYPO만으로 충분하다면 SD 생성은 왜 필요했는가? SD는 "구절을 의미적으로 보강"하는 역할인지, 단순히 거부를 흐리는 noise인지 분리 실험이 더 필요하다.
- 모델이 이미지를 "읽는" 능력(OCR)과 "거부하는" 능력이 분리되는지, 아니면 OCR이 좋아질수록 더 취약해지는지 추적할 필요가 있다.

---

## 내 판단 (My Assessment)

- 방법 자체는 단순하지만, "유해 의미를 모달리티 사이로 옮기면 정렬이 깨진다"는 문제 정의를 대규모 벤치마크로 고정했다는 점에서 VLM safety 연구의 기준점이 되는 논문이다.
- 가장 강한 부분은 SD / TYPO / SD+TYPO 비교다. 공격 성공의 원인이 "이미지가 무섭기 때문"이 아니라 "글자가 정확히 전달되기 때문"임을 분리해 보여 준다.
- 가장 약한 부분은 방어 실험의 깊이다. Safety prompt 하나로 마무리되어, 학습 기반 방어(visual safety tuning 등)와의 비교가 없다.

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

- VLM jailbreak 연구에서 **baseline 공격**과 **표준 평가셋**으로 가장 먼저 쓸 수 있다.
- 다국어 flowchart 공격(MLingualFC), 구성적 공격(Jailbreak in Pieces) 등 후속 연구는 모두 이 벤치마크의 "텍스트는 무해, 이미지가 유해" 구도를 변형한 것이라 비교 축으로 유용하다.
- OCR 능력과 취약성의 상관관계를 보려면 TYPO 이미지의 글꼴·해상도·언어를 바꾸는 ablation을 추가하면 된다.

## Action Items

- [ ] 13개 시나리오 중 01~07과 08~13을 분리해서 ASR을 보고하는 습관을 들일 것 (평가 기준이 다름)
- [ ] 최신 상용 VLM에 SD+TYPO 설정을 재평가하고 2023년 결과와 비교
- [ ] Safety prompt의 over-refusal 비율을 함께 측정하는 평가 스크립트 작성
- [ ] FigStep, Jailbreak in Pieces와 공격 성공 원인을 비교하는 표 정리
