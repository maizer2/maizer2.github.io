---
layout: post
categories: [1. Computer Engineering, 1. Artificial Intelligence, 2. Deep Learning, 3. Trustworthy AI, 1. AI Safety, 3. Jailbreak]
title: "MLingualFC: Evaluating Jailbreak Vulnerabilities in Multilingual Vision-Language Models"
tags: [Jailbreak, Jailbreak Attack, VLM, AI Safety, Paper review, Safety training, Multi-turn attack, Black box attack, Safety alignment]
---

# Review Summary

## Basic information

- **Published** : arXiv preprint, 2026년 6월 5일 (v1). cs.CR (primary), cs.AI
- **Title** : [MLingualFC: Evaluating Jailbreak Vulnerabilities in Multilingual Vision-Language Models](https://arxiv.org/abs/2606.07706)
- **Authors** : Rishabh Makwana, Mamta, Deeksha Varshney, Oana Cocarascu
- **Code / Project page** : GitHub `https://github.com/Rishabhpm23/MLingualFC` (전체 데이터셋은 인증된 연구자에게만 제공)
- **Main keywords** : multilingual VLM, flowchart jailbreak, FC-Attack, visual text recognition, cross-lingual safety gap

## One-line summary

이 논문은 **유해 지시를 순서도(flowchart) 이미지로 바꾼 뒤 비영어권 언어로 번역하면 다국어 VLM의 안전정렬이 언어에 따라 크게 달라진다는 문제**를 다루며, **AdvBench 기반 영어 순서도를 힌디어·펀자브어·스페인어·루마니아어·독일어로 번역한 black-box 벤치마크 MLingualFC**를 만들어, **라틴 문자 언어에서는 높은 ASR이, 인도계 문자에서는 낮은 ASR이 나오지만 후자는 정렬이 강해서가 아니라 시각적 글자 인식이 약해서라는 점**을 보인다.

---

## 주요 기여 (Key Contributions)

### 기여 1: 다국어 멀티모달 jailbreak 벤치마크 MLingualFC

- 기존 flowchart jailbreak(FC-Attack)는 영어 전용이었다. 이 논문은 같은 순서도를 5개 언어로 옮겨 **언어 × 레이아웃 × 모델** 축에서 취약성을 체계적으로 비교할 수 있는 첫 벤치마크를 만들었다.

### 기여 2: 세 가지 최신 다국어 VLM에 대한 평가

- Qwen2.5-VL-3B-Instruct, Gemma-4-E4B-it, Pangea-7B(LLaVA-NeXT 구조)를 black-box, single-turn 설정에서 평가했다.

### 기여 3: "언어 간 안전성 불일치"에 대한 실증

- 같은 모델이라도 언어에 따라 ASR이 0%에서 80%대까지 벌어진다. 안전 메커니즘이 언어와 모달리티를 넘어 일반화되지 않는다는 것을 수치로 보였다.

### 기여 4: 낮은 ASR을 "안전함"으로 해석하면 안 된다는 경고

- 힌디어·펀자브어에서 ASR이 낮은 것은 모델이 순서도의 글자를 읽지 못하기 때문이며, 같은 내용을 평문 텍스트로 주면 오히려 ASR이 올라간다. 즉 **시각적 텍스트 인식 능력이 좋아지면 취약성도 따라 올라갈 수 있다.**

---

## 방법 요약 (Method Summary)

### 문제 설정 (Threat Model)

- 공격자는 모델 가중치나 gradient 없이 `(flowchart image, benign text)`만 입력한다.
- 텍스트 프롬프트는 "순서도의 단계를 분석하고 완성하라"는 식의 무해한 지시문이며 유해 단어를 포함하지 않는다.
- 단일 턴 공격이다.

### 순서도 생성 (Multilingual Flowchart Generation)

1. **소스** : AdvBench 유해 질의를 절차적 하위 단계로 분해한 FC-Attack의 영어 순서도 데이터(방향 그래프, DOT 파일)를 사용한다.
2. **번역** : DOT 파일의 텍스트를 GoogleTranslator(deep-translator)로 힌디어, 펀자브어, 스페인어, 루마니아어, 독일어로 번역한다.
3. **렌더링** : Graphviz로 PNG 순서도를 만든다.
4. **레이아웃** : 가로(horizontal), 세로(vertical), S자형(tortuous) 세 가지를 질의·언어마다 생성한다.

### 평가 (Evaluation)

- **ASR** = 해당 언어에서 jailbreak로 판정된 응답 수 / 전체 샘플 수.
- **Judge** : GPT-oss-120b가 응답을 1(완전 거부)~5(완전 유해)로 채점하고, 4 이상이면 jailbreak로 본다.
- 힌디어·펀자브어 응답 일부는 원어민이 직접 채점해 judge와의 차이를 확인했다.

### 기존 방법과의 차이

- FC-Attack과 순서도·레이아웃 설계는 같지만, 질문이 "영어로 공격이 되는가"에서 "언어를 바꾸면 안전성이 유지되는가"로 바뀌었다.
- FigStep, MM-SafetyBench 같은 typography 계열 공격과 달리, 구조화된 절차(순서도)를 쓰기 때문에 모델이 "단계를 완성한다"는 명목으로 유해 내용을 생성하기 쉽다.

---

## 실험 설정 및 결과 (Experiments & Results)

### 대상 모델 / 환경

- Qwen2.5-VL-3B-Instruct, Gemma-4-E4B-it, Pangea-7B. 모두 NVIDIA L4 24GB 한 장에서 실행.

### 주요 결과 (ASR %, Flowchart + Harmless Text, 레이아웃 가로 / 세로 / S자형)

| 언어 | Qwen2.5-VL | Gemma-4 | Pangea |
| --- | --- | --- | --- |
| English | 84.00 / 70.00 / 86.00 | 26.00 / 18.00 / 18.00 | 54.00 / 72.00 / 54.00 |
| Spanish | 80.85 / 72.34 / 89.36 | 53.19 / 21.28 / 31.91 | 59.57 / 51.06 / 23.40 |
| Romanian | 61.70 / 87.23 / 82.98 | 63.83 / 29.79 / 42.55 | 40.43 / 38.29 / 40.43 |
| German | 76.59 / 82.98 / 89.36 | 70.21 / 34.04 / 53.19 | 48.94 / 42.55 / 44.68 |
| Hindi | 19.15 / 53.19 / 40.42 | 74.47 / 59.57 / 68.08 | 0.00 / 0.00 / 0.00 |
| Punjabi | 4.26 / 0.00 / 4.26 | 6.38 / 14.89 / 4.26 | 0.00 / 2.13 / 2.13 |

- **Qwen2.5-VL** 이 가장 취약하다. 스페인어·독일어·루마니아어는 레이아웃에 따라 80%를 넘는다.
- **Gemma-4** 는 영어에서는 비교적 안전(18~26%)하지만 독일어·루마니아어·힌디어에서는 영어보다 훨씬 취약하다. 힌디어 가로 레이아웃 74.47%는 전체 표에서 가장 높은 값이다.
- **Pangea** 는 힌디어에서 모든 레이아웃이 0%이고, 영어·스페인어에서 가장 취약하다.
- **레이아웃 효과** : Gemma-4와 Pangea는 가로 레이아웃에 가장 약하고, Qwen은 언어마다 취약한 레이아웃이 다르다.

### Ablation

- **평문 baseline(Text + Description, Harmful Query)** : 라틴 문자 언어에서는 순서도가 평문보다 ASR이 높지만, 힌디어·펀자브어에서는 반대다. 예를 들어 Qwen의 힌디어 Text + Description ASR은 64.00%, 펀자브어는 66.00%인데 순서도에서는 각각 53.19%, 4.26%다. 저자들은 이를 "인도계 문자 순서도는 정렬 때문이 아니라 읽히지 않아서 실패한다"는 근거로 쓴다.
- **단계 수(5단계 순서도, Qwen)** : 가로 레이아웃에서 단계를 줄이면 ASR이 오른다 (영어 84.00→92.00, 루마니아어 61.70→81.25, 힌디어 19.15→33.33). 세로·S자형은 반대로 단계가 많을수록 유리해, 효과가 레이아웃에 의존한다.

### 정성 분석 / Human evaluation

- 평문 루마니아어·힌디어 요청은 거부되었지만 같은 내용의 순서도는 유해 응답을 냈고, 루마니아어 S자형 순서도에 영어로 유해 답변을 내는 경우도 있었다.
- 같은 응답 안에서 거부한 뒤 유해 내용을 이어 쓰는 사례가 있다.
- 힌디어 순서도에 Qwen이 전혀 무관한 내용(예: 시험 준비 계획)을 생성한 사례가 "오독" 근거로 제시된다.
- 원어민 평가는 대체로 LLM judge와 비슷했지만(Gemma-4 힌디어 72/58/68 vs 74.47/59.57/68.08), Gemma-4 펀자브어 세로 레이아웃은 2.12 vs 14.89로 차이가 커서 논문이 말하는 "1~6% 차이"와 맞지 않는다.

### 저자가 밝힌 한계

- 모델 3종만 평가해 다른 모델 계열로의 일반화가 검증되지 않았다.
- 방어 기법이나 다국어 정렬 전략은 평가하지 않았다. 다국어 시각 jailbreak 방어는 향후 과제로 남겼다.

---

## 장점 및 시사점 (Advantages & Learnings)

### 시사점 1: 안전성 평가는 언어별로 따로 보고해야 한다

- 영어 ASR만으로 모델의 안전성을 말하면 Gemma-4처럼 영어에서는 안전하고 독일어·힌디어에서는 취약한 경우를 놓친다.

### 시사점 2: "읽지 못함"과 "거부함"을 분리해야 한다

- 낮은 ASR이 능력 부족 때문인지 정렬 때문인지 구분하지 않으면, 모델의 OCR·다국어 능력이 향상되는 순간 잠재된 취약성이 드러난다. 평문 baseline과 비교하는 설계가 이 분리를 가능하게 했다.

### 시사점 3: 파이프라인이 저렴하고 재현 가능하다

- 기계 번역 + Graphviz만으로 언어를 추가할 수 있어 다른 언어(예: 한국어)로 확장하기 쉽다.

---

## 한계 및 의문점 (Limitations & Questions)

### 실험 범위의 한계

- 언어당 샘플 수가 약 47~50개로 작아(표의 비율로 역산한 추정치) 레이아웃 간 몇 %p 차이는 통계적으로 의미가 약할 수 있다.
- 3B~7B급 소형 오픈소스 모델만 평가해 상용 대형 VLM 결과는 알 수 없다.

### 방법론적 한계

- 기계 번역 품질을 검증하지 않았다. 번역 오류가 순서도의 유해성 자체를 희석했을 가능성이 있다.
- "시각적 글자 인식이 약해서"라는 핵심 주장을 뒷받침하는 직접적인 OCR 측정 실험이 없다. 근거는 평문 baseline과 정성 사례뿐이다.

### 평가 방식의 한계

- Judge(GPT-oss-120b)의 다국어 판정 능력 자체가 검증되지 않았고, 펀자브어에서 원어민 평가와의 차이가 크다.

### 질문

- 순서도를 영어로 두고 텍스트 프롬프트만 다국어로 바꾸면 어떻게 되는가? 취약성이 "이미지 속 언어"에서 오는지 "지시문 언어"에서 오는지 분리할 수 있다.
- OCR 전처리로 순서도를 텍스트로 변환해 주는 방어가 인도계 문자에서는 오히려 취약성을 높이지 않을까?

---

## 내 판단 (My Assessment)

- 기여의 핵심은 새로운 공격 기법이 아니라 **평가 축의 확장**이다. FC-Attack을 다국어로 옮긴 것 자체는 단순하지만, "낮은 ASR ≠ 안전"이라는 해석을 평문 baseline으로 보여 준 부분이 가장 가치 있다.
- 가장 약한 부분은 표본 크기와 OCR 주장의 간접성이다. 인용할 때는 "인도계 문자에서 시각 텍스트 인식 한계가 의심된다" 수준으로 표현하는 것이 안전하다.

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

- 다국어 VLM 안전 연구에서 **언어별 ASR 분리 보고**와 **평문 vs 이미지 baseline 동시 보고**를 평가 프로토콜의 기본으로 삼을 근거가 된다.
- MM-SafetyBench(영어 typography) → FC-Attack(영어 순서도) → MLingualFC(다국어 순서도)로 이어지는 계보에서 한국어를 추가하는 확장이 바로 가능하다.
- "능력 향상이 취약성 증가로 이어진다"는 가설은 모델 버전 간 종단 비교 실험으로 검증해볼 만하다.

## Action Items

- [ ] 한국어 순서도 세트를 같은 파이프라인으로 생성해 보고, 번역 품질을 사람이 검수하는 단계 추가
- [ ] OCR 정확도(순서도 텍스트 복원률)를 직접 측정해 ASR과의 상관을 확인하는 실험 설계
- [ ] 이미지 언어와 지시문 언어를 교차시키는 2×2 설정으로 취약성 원인 분리
- [ ] FC-Attack 원 논문과 MM-SafetyBench의 영어 결과와 수치 비교표 작성
