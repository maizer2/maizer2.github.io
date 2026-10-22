---
layout: post
categories: [1. Computer Engineering, 1. Artificial Intelligence, 2. Deep Learning, 3. Trustworthy AI, 2. Mechanistic Interpretability]
title: "Unsupervised Domain Clusters in Pretrained Language Models"
tags: [Unsupervised Clustering, Pretrained Language Models, Domain Adaptation, Domain Data Selection, Representation Analysis, BERTology, Positive-Unlabeled Learning, Latent Space Geometry, Neural Machine Translation, Paper review]
---

# Review Summary

## Basic information

- **Published** : ACL 2020
- **Title** : [Unsupervised Domain Clusters in Pretrained Language Models](https://arxiv.org/abs/2004.02105)
- **Authors** : Roee Aharoni, Yoav Goldberg (Bar-Ilan University / Allen Institute for AI)
- **Code / Project page** : GitHub `https://github.com/roeeaharoni/unsupervised-domain-clusters`
- **Main keywords** : domain, sentence embedding, GMM clustering, data selection, NMT, positive-unlabeled learning

## One-line summary

- 연구 배경 : 도메인 경계가 모호하고 라벨링된 데이터가 부족하여 특정 도메인 모델 구축이 어려움
- 발견 사실 : 사전학습 언어 모델의 문장 임베딩 공간에서 텍스트가 **비지도 방식으로 도메인별 클러스터**를 형성함
- 적용 및 성과 : 소량의 in-domain 데이터만으로 효과적인 도메인 데이터 선별 기법을 제안하여 기계 번역 벤치마크에서 기존 기준선(Moore-Lewis) 대비 성능 개선

---

## 주요 기여 (Key Contributions)

### 기여 1: "도메인"을 데이터가 스스로 정의하게 한다

- NLP에서 도메인은 보통 출처(뉴스, 자막, 의학 문서 등)로 사람이 라벨링한다. 이 논문은 **도메인은 사전학습 LM의 표현 공간에서 자연스럽게 생기는 클러스터**라는 관점을 제안하고, 그 클러스터가 사람이 붙인 도메인 라벨과 높은 수준으로 일치함을 보인다.

### 기여 2: 다양한 사전학습 모델의 표현 공간 비교

- word2vec 평균, BERT, DistilBERT, RoBERTa, XLNet, GPT-2 등의 문장 임베딩에 GMM 클러스터링을 적용해 도메인 purity를 비교한다. Masked LM 계열(BERT, RoBERTa, DistilBERT)이 autoregressive 계열(GPT-2, XLNet)보다 도메인을 훨씬 잘 분리한다.

### 기여 3: 클러스터 관찰을 데이터 선별 알고리즘으로 연결

- 소량의 in-domain 문장만 주어졌을 때, 대규모 일반 코퍼스에서 같은 도메인의 문장을 골라내는 두 가지 방법(Domain-Cosine, Domain-Finetune)을 제안한다.

### 기여 4: NMT에서의 검증

- 독일어-영어 다도메인 번역 세팅에서, 제안 방법으로 고른 데이터로 학습한 모델이 고전적인 Moore-Lewis cross-entropy difference 선별보다 대부분 도메인에서 더 높은 BLEU를 달성한다.

---

## 방법 요약 (Method Summary)

### 문제 설정

- 다도메인 코퍼스(Koehn & Knowles 2017의 독-영 데이터: subtitles, medical, legal, Koran, IT 다섯 도메인)가 있을 때, 라벨 없이 문장만 보고 도메인을 복원할 수 있는가?
- 그리고 목표 도메인의 문장 몇 천 개만 있을 때 대규모 혼합 코퍼스에서 목표 도메인 데이터를 선별할 수 있는가?

### 핵심 메커니즘 1: 비지도 도메인 클러스터링

1. 각 문장을 사전학습 모델에 통과시키고 토큰 hidden state를 평균해 문장 벡터를 얻는다.
2. 도메인 수 k(여기서는 5)를 알고 있다고 가정하고 GMM으로 클러스터링한다.
3. 클러스터와 실제 도메인 라벨의 일치도를 **purity**로 측정한다.

- 결과적으로 MLM 기반 임베딩은 purity가 80% 후반대까지 올라가며, word2vec 평균이나 LDA 기반 baseline보다 훨씬 높다. 특히 BERT와 RoBERTa는 큰 모델이 작은 모델보다 조금 더 좋다.
- 시각화(PCA/t-SNE)에서 도메인별로 뚜렷한 덩어리가 보이고, 혼동은 의미적으로 가까운 도메인(예: legal↔medical의 형식적 문장) 사이에서 일어난다.

### 핵심 메커니즘 2: 도메인 데이터 선별

- **Domain-Cosine** : in-domain seed 문장들의 평균 벡터를 구하고, 혼합 코퍼스 문장과의 코사인 유사도가 높은 순으로 선택한다. 학습이 필요 없는 가장 단순한 방법이다.
- **Domain-Finetune** : BERT를 이진 분류기로 fine-tuning한다. positive는 in-domain seed, negative는 혼합 코퍼스에서 무작위로 뽑은 문장이다. 혼합 코퍼스 안에도 positive가 섞여 있으므로 사실상 **Positive-Unlabeled(PU) learning** 상황이며, 부정 샘플을 재추출하는 negative sampling 변형도 실험한다.
- 선별된 문장의 **precision / recall**(실제로 그 도메인에 속하는지)을 라벨로 평가하고, 선별 데이터로 NMT를 학습해 BLEU로 end-to-end 효과를 평가한다.

### 기존 방법과의 차이

- Moore-Lewis(2010)는 in-domain LM과 general LM의 cross-entropy 차이로 문장을 점수화한다. n-gram LM이라 어휘 중복에 민감하고 의미적 유사성은 보지 못한다.
- 제안 방법은 사전학습 LM의 문맥 표현을 쓰므로 어휘가 달라도 같은 도메인으로 묶을 수 있고, 특히 Domain-Finetune은 소량 seed로도 분류 경계를 잘 잡는다.

---

## 실험 설정 및 결과 (Experiments & Results)

### 데이터셋

- 독일어-영어 다도메인 병렬 코퍼스(subtitles, medical, legal, Koran, IT). 도메인별 크기 편차가 크다(subtitles가 압도적으로 큼).

### 평가 지표

- 클러스터링: purity
- 데이터 선별: precision, recall, F1 (라벨 기준)
- 번역: BLEU (도메인별 테스트셋)

### 주요 결과

- **클러스터링** : BERT/RoBERTa/DistilBERT 임베딩 + GMM이 가장 높은 purity. GPT-2, XLNet, word2vec은 눈에 띄게 낮다. 즉 "문장 임베딩이면 다 된다"가 아니라 MLM 목적함수로 학습된 표현이 도메인 정보를 더 잘 담는다.
- **선별 품질** : Domain-Finetune이 Moore-Lewis와 Domain-Cosine보다 precision·recall 모두 높다. Domain-Cosine은 간단한데도 Moore-Lewis와 비슷하거나 낫다.
- **NMT** : 선별 데이터로 학습한 Transformer가 대부분 도메인에서 Moore-Lewis 선별보다 높은 BLEU를 기록하고, 일부 도메인에서는 oracle(라벨로 고른 데이터)에 근접한다. 다른 도메인 데이터를 섞어 학습한 모델이 in-domain만 쓴 모델보다 나쁠 수 있다는 점도 함께 보여 "데이터는 많을수록 좋다"가 도메인 간에는 성립하지 않음을 확인한다.

### Ablation / 추가 분석

- seed 크기, negative sampling 유무, 선별 데이터 양에 따른 BLEU 변화를 보고한다. negative sampling을 쓴 Domain-Finetune이 가장 안정적이다.

---

## 장점 및 시사점 (Advantages & Learnings)

### 시사점 1: 표현 공간의 기하가 곧 데이터의 메타데이터다

- 사람이 붙인 "도메인" 라벨이 사실은 LM 표현 공간에서 거의 선형적으로 복원되는 구조라는 것은, 표현 분석(BERTology)과 데이터 관리(데이터 선별·필터링)를 연결하는 출발점이다.

### 시사점 2: PU learning 관점의 데이터 선별

- 라벨 없는 대규모 코퍼스에서 "positive만 몇 개 있다"는 상황은 매우 흔하다. BERT 이진 분류 + negative sampling이라는 단순한 레시피가 잘 작동한다는 것은 실무적으로 중요한 결과다.

### 시사점 3: 모델 계열에 따른 표현 차이

- MLM과 autoregressive 모델의 임베딩이 도메인 분리 능력에서 크게 갈린다는 관찰은, 이후 "어떤 모델의 표현을 분석에 쓸 것인가"를 정할 때 참고할 근거가 된다.

---

## 한계 및 의문점 (Limitations & Questions)

### 실험 범위의 한계

- 도메인이 5개이고 서로 꽤 다른 출처(자막 vs 쿠란 vs 법률)라 분리가 쉬운 세팅이다. 세부 도메인(예: 뉴스 안의 정치·스포츠)에서도 같은 purity가 나오는지는 별개 문제다.
- 언어쌍이 독-영 하나뿐이고 임베딩은 영어 쪽 문장에서만 뽑는다.

### 가정의 한계

- GMM에 클러스터 수 k를 정답으로 넣는다. 실제로는 k를 모르므로 "비지도"라고 하기엔 한 단계 정보가 들어가 있다.
- 토큰 평균으로 문장 벡터를 만드는 방식은 거칠다. 다른 pooling(CLS, SBERT 등)에서 결과가 달라질 수 있다.

### 질문

- 클러스터가 분리되는 이유가 "주제" 때문인지 "문체·길이·형식" 때문인지 구분하지 않는다. 자막(짧은 구어체)과 법률(긴 문어체)은 길이만으로도 갈릴 수 있다.
- 2020년 이후 LLM 임베딩에서도 같은 현상이 유지되는가? 모델이 커질수록 도메인 클러스터가 더 뚜렷해지는지, 반대로 희석되는지 추적할 가치가 있다.

---

## 내 판단 (My Assessment)

- 방법은 단순하지만 "도메인은 표현 공간의 클러스터다"라는 프레이밍이 명확해 자주 인용되는 논문이다. 가장 강한 부분은 관찰(클러스터링)을 바로 응용(데이터 선별 → BLEU)으로 연결해 실용성을 입증한 점이다.
- 가장 약한 부분은 분리 원인 분석의 부재다. 무엇이 클러스터를 만드는지(주제 vs 형식)를 통제 실험으로 보여 주지 않았다.

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

- 표현 공간 분석(mechanistic / representation-level 해석) 쪽에서, **라벨 없이 표현 공간의 구조만으로 데이터 속성을 복원하는** 가장 기본적인 실험 설계로 쓸 수 있다. 안전 연구에서는 "유해/무해", "jailbreak/정상 요청"이 표현 공간에서 비지도로 갈리는지 같은 방식으로 확인할 수 있다.
- Domain-Finetune의 PU learning 구성은 소량의 positive(예: 특정 공격 유형 샘플)만 가지고 대규모 로그에서 유사 사례를 찾는 데 그대로 재사용 가능하다.

## Action Items

- [ ] 최신 LLM(디코더 전용) hidden state로 같은 5-도메인 clustering purity를 재현해 2020년 결과와 비교
- [ ] 문장 길이·형식을 통제한 뒤에도 도메인 클러스터가 유지되는지 확인하는 통제 실험 설계
- [ ] 안전 데이터(유해/무해 프롬프트)에 Domain-Cosine / Domain-Finetune 선별 파이프라인 적용해 보기
