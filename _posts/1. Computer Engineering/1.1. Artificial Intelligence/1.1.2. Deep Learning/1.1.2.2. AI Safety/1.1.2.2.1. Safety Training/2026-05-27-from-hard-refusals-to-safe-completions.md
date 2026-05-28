---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.2. AI Safety, 1.1.2.2.1. Safety Training]
title: "From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training"
tags: [Safe-completions, Refusal training, Output-centric safety, Dual-use prompts, RLHF, Safety reward, Helpfulness reward, GPT-5, Biorisk, Human evaluation, Paper review]

---

# Review Summary

## Basic information

* Published : 2025-08-11
* Title : [From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training](https://arxiv.org/abs/2508.09224)

## One-line summary

이 논문은 **기존 refusal-based safety training이 dual-use 및 intent가 모호한 프롬프트에서 brittle하게 동작하는 문제**를 다루며, **모델 출력의 안전성과 유용성을 동시에 보상하는 safe-completion training**을 통해 **안전성을 유지하거나 높이면서도 helpfulness를 크게 개선하고, 실패 시 harm severity를 낮출 수 있음**을 보인다. 

---

## 주요 기여 (Key Contributions)

### 기여 1: refusal boundary의 brittle함 문제 제기

* 기존 안전 학습은 사용자의 intent를 기준으로 **comply vs refuse**를 이진 분류하는 방식에 가깝다.
* 논문은 이 방식이 명시적으로 악의적인 prompt에는 강하지만, **dual-use prompt**나 intent가 은폐된 prompt에서는 취약하다고 주장한다.
* 대표 예시로, pyrogen ignition 관련 두 prompt가 사실상 같은 정보를 요구하지만, 하나는 benign/dual-use처럼 보이고 다른 하나는 malicious하게 표현되어 o3가 전자에는 actionable detail을 제공하고 후자에는 hard refusal을 하는 사례를 제시한다.

### 기여 2: output-centric safety training인 safe-completions 제안

* 논문은 **safe-completion**이라는 training paradigm을 제안한다.

* 핵심 아이디어는 다음과 같다.

  `사용자 intent 분류 중심 학습 → 모델 output이 실제로 정책을 위반하는지 중심 학습`

* 즉, prompt 자체가 위험해 보이는지보다, **assistant response가 harmful action의 barrier를 낮추는지**를 중심으로 판단한다.

* 모델은 가능한 경우 직접 답변하고, 제한적 dual-use 상황에서는 high-level/non-operational guidance를 제공하며, 안전하게 답할 수 없는 경우에는 refusal과 safe redirection을 제공한다.

### 기여 3: safety와 helpfulness를 곱하는 RL reward 설계

* RL 단계에서 두 reward model을 사용한다.

  * Safety score `s_i ∈ [0,1]`
  * Helpfulness score `h_i ∈ [0,1]`

* 최종 reward는 다음과 같다.

  `r_i = h_i · s_i`

* 이 구조에서는 response가 unsafe이면 helpful해도 reward가 낮고, safe하지만 아무 도움도 안 되면 reward도 낮다.

* 따라서 모델은 안전 제약 내에서 **direct helpfulness** 또는 **indirect helpfulness**를 최대화하도록 학습된다.

### 기여 4: controlled experiment와 production model 비교

* 논문은 두 종류의 비교를 수행한다.

  * Controlled Experiments: CE-Refusal vs CE-SafeComplete
  * Production Models: o3 vs GPT-5 Thinking, 논문에서는 gpt5-r로 표기
* controlled setting에서는 architecture, pretraining corpus, post-training recipe를 고정하고 safety-training strategy만 바꾼다.
* production setting은 완전한 ablation은 아니지만 실제 배포 모델 간 비교라는 점에서 현실성이 높다.

### 기여 5: 정량적 결과와 human evaluation

* safe-completion은 dual-use prompt에서 safety를 개선하거나 유지하면서 helpfulness를 높인다.
* gpt5-r은 o3 대비 dual-use와 malicious prompt에서 safety가 각각 약 9, 10 percentage points 높다고 보고된다.
* human evaluation에서도 safe-completion 모델이 safety, helpfulness, balance에서 더 선호된다.
* biorisk case study에서는 gpt5-r의 unsafe response 중 high/moderate harm 비율이 14.7%로, o3의 42.7%보다 크게 낮다.

---

## 방법 요약 (Method Summary)

### 문제 설정

* 입력 : safety-relevant user prompt
* 출력 : assistant response
* 공격자 / 평가자 목표 : 모델이 safety policy를 위반하지 않으면서 가능한 한 유용한 답변을 생성하는지 평가
* 가정 : 각 prompt는 illicit, erotic, hate, sensitive information 등 safety category와 관련 spec을 가진다.
* 제약 조건 : response가 policy를 위반하거나 meaningful facilitation을 제공하면 unsafe로 간주된다.

### 핵심 메커니즘

논문의 방법은 크게 다음 단계로 구성된다.

1. **Policy spec 기반 SFT**

   * prompt에 safety policy spec과 “spec을 참고해 답하라”는 instruction을 붙인다.
   * base reasoning model로 spec-aware CoT와 answer를 생성한다.
   * unsafe answer는 judge model로 filtering한다.

2. **세 가지 response mode 학습**

   * Direct answer: 완전히 harmless한 요청에 직접 답변
   * Safe-completion: 제한적이거나 dual-use인 요청에 high-level, non-operational guidance 제공
   * Refuse with redirection: 안전하게 답할 수 없는 요청에는 거절과 safe alternative 제공

3. **RL 단계에서 safety/helpfulness reward 적용**

   * Safety RM은 output이 policy를 얼마나 준수하는지 평가한다.
   * Helpfulness RM은 direct helpfulness와 indirect helpfulness를 함께 평가한다.
   * 최종 reward는 `r_i = h_i · s_i`로 계산된다.

4. **출력 중심 정책 업데이트**

   * 기존 illicit policy의 중심을 “사용자가 지시를 요구했는가?”에서 “모델 output이 harmful action의 barrier를 의미 있게 낮추는가?”로 이동시킨다.

### 모델 / 시스템 구조상 중요한 지점

* 취약점이 발생하는 위치 : prompt intent classifier 또는 refusal boundary에 과도하게 의존하는 safety behavior
* 관찰 또는 조작하는 representation : 최종 assistant output의 safety/helpfulness
* safety mechanism과 충돌하는 지점 : dual-use request에서 full compliance와 hard refusal 사이의 binary trade-off
* 실패가 전파되는 경로 : ambiguous prompt → benign으로 오분류 → overly detailed operational output → harm facilitation

### 기존 방법과의 차이

* 기존 접근 : prompt intent를 기준으로 comply/refuse 결정
* 이 논문의 접근 : output이 policy를 위반하는지, 그리고 안전 제약 내에서 얼마나 도움이 되는지를 보상
* 실질적인 차이 : hard refusal 대신 high-level guidance, symbolic/template answer, safe alternatives, vendor checklist 등으로 **partial but safe assistance**를 제공한다.

---

## 실험 설정 및 결과 (Experiments & Results)

### 대상 모델 / 시스템

* CE-Refusal
* CE-SafeComplete
* o3
* GPT-5 Thinking / gpt5-r

### 데이터셋 / 벤치마크

* ChatGPT production data에서 추출한 safety-related prompt 약 9,000개
* harm category:

  * Illicit
  * Erotic
  * Hate
  * Sensitive Information
* biorisk-related prompt 620개
* human evaluation용 illicit behavior prompt 2,000개

### 평가 지표

* Safety: response가 content policy를 위반하는지 여부
* Helpfulness given safe output: safe response에 대해서만 1–4 helpfulness score 평가
* Intent class:

  * Benign
  * Dual-use
  * Malicious
* Harm severity:

  * Negligible
  * Low
  * Moderate
  * High
* Human evaluation:

  * absolute safety
  * relative helpfulness
  * balance, 즉 safety-helpfulness trade-off 선호도

### 주요 결과

* CE-SafeComplete는 CE-Refusal 대비 dual-use prompt에서 safety를 개선하고, benign/malicious에서는 대체로 유지한다.
* gpt5-r은 o3 대비 benign, dual-use, malicious 전반에서 safety가 개선되며, 특히 dual-use와 malicious에서 큰 차이를 보인다.
* helpfulness는 safe-completion 모델이 전반적으로 높고, malicious prompt에서는 hard refusal 대신 safe redirection을 제공하기 때문에 차이가 특히 크다.
* unsafe response만 놓고 보아도 safe-completion 모델은 High/Moderate harm 비중을 줄이고 Low/Negligible 쪽으로 분포를 이동시킨다.
* human evaluation에서도 CE-SafeComplete와 gpt5-r이 각각 refusal baseline보다 helpfulness와 balance에서 더 선호된다.

### 중요한 Figure / Table

* **Figure 1**

  * 보여주는 내용 : o3가 같은 정보를 요구하는 두 pyrogen prompt에 대해 dual-use 표현에는 상세 actionable answer를, malicious 표현에는 hard refusal을 제공한다.
  * 왜 중요한가 : refusal boundary가 user intent 표면 신호에 과도하게 의존한다는 핵심 문제를 직관적으로 보여준다.

* **Figure 2**

  * 보여주는 내용 : GPT-5는 같은 dual-use pyrogen prompt에 대해 actionable parameters를 제공하지 않고, standards, vendor datasheet, certified firing system 등 safe alternatives를 제시한다.
  * 왜 중요한가 : safe-completion이 단순 refusal이 아니라 “도움이 되는 안전한 대체 응답”을 목표로 함을 보여준다.

* **Figure 3**

  * 보여주는 내용 : safe-completion training stack과 `Safety Reward × Helpfulness Reward` 구조
  * 왜 중요한가 : 논문의 방법론적 핵심이다.

* **Figure 4**

  * 보여주는 내용 : intent별 safety와 helpfulness given safe output 비교
  * 왜 중요한가 : controlled experiment와 production model 모두에서 safe-completion이 safety-helpfulness trade-off를 개선한다는 핵심 결과다.

* **Figure 5**

  * 보여주는 내용 : unsafe response 중 harm severity distribution
  * 왜 중요한가 : 단순히 unsafe rate만 줄이는 것이 아니라, 실패했을 때도 더 덜 위험하게 실패한다는 점을 보여준다.

* **Figure 6–7**

  * 보여주는 내용 : biorisk prompt에서 safety/helpfulness 및 harm severity 분석
  * 왜 중요한가 : dual-use성이 매우 강한 frontier biorisk 영역에서도 safe-completion이 유효하다는 case study다.

* **Figure 8–9**

  * 보여주는 내용 : human evaluation에서 safety, helpfulness, balance와 safety rating distribution
  * 왜 중요한가 : 자동 평가뿐 아니라 인간 평가에서도 같은 경향이 확인된다.

### Ablation / 추가 분석

* 제거하거나 바꾼 요소 : refusal-oriented safety training을 safe-completion training으로 대체
* 결과 변화 :

  * safety는 유지 또는 개선
  * helpfulness는 증가
  * unsafe response의 harm severity는 낮아짐
* 해석 :

  * 모델이 “위험하면 무조건 거절” 또는 “괜찮아 보이면 완전 응답”이라는 양극단 대신, 정책 제약 내에서 안전한 수준의 정보를 제공하는 방식을 학습했기 때문으로 보인다.

### Negative results / 실패한 조건

* 논문은 뚜렷한 실패 사례를 많이 제시하지는 않는다.
* 다만 controlled experiments에서 일부 category, 특히 malicious illicit 쪽에서 작은 safety regression이 관찰된다고 언급한다.
* 저자의 설명 : 추가 failure가 더 낮은 severity의 harm일 가능성이 높다.
* 내가 보기엔 가능한 원인 :

  * safe-completion이 partial answer를 장려하기 때문에, policy boundary 근처에서 일부 response가 borderline unsafe로 판정될 수 있다.
  * indirect helpfulness를 높이려는 압력이 너무 강하면, 안전한 redirection과 operational hint 사이의 경계가 흐려질 수 있다.

---

## 장점 및 시사점 (Advantages & Learnings)

### 시사점 1

* 이 논문은 **intent classification 기반 refusal**만으로는 dual-use safety를 해결하기 어렵다는 점을 보여준다.
* 따라서 user prompt의 표면적 악의성보다 **output이 실제로 어떤 affordance를 제공하는가**가 중요하다.

### 시사점 2

* **출력의 detail level, actionability, specificity**는 단순한 문체 문제가 아니라 safety-critical variable이다.
* 같은 topic이라도 high-level overview는 허용 가능할 수 있고, quantities, thresholds, troubleshooting, procedural steps는 위험할 수 있다.

### 시사점 3

* Safe-completion은 safety를 refusal behavior가 아니라 **constrained helpfulness optimization**으로 재정의한다.
* 이는 alignment에서 “helpful vs harmless trade-off”를 보다 세밀하게 다루는 방향이다.

### 시사점 4

* 기존 방어 방식인 **keyword filtering, prompt intent classification, hard refusal tuning**에는 한계가 있다.
* 특히 biorisk, cybersecurity, chemistry, explosives처럼 legitimate use와 harmful use가 겹치는 영역에서는 binary policy가 과도한 over-refusal 또는 dangerous compliance를 만들 수 있다.

### 시사점 5

* 향후 safety evaluation은 단순 refusal rate나 violation rate뿐 아니라 다음 축을 포함해야 한다.

  * helpfulness conditioned on safety
  * residual failure severity
  * dual-use prompt handling
  * human-perceived balance
  * meaningful facilitation 여부

---

## 한계 및 의문점 (Limitations & Questions)

### 실험 범위의 한계

* 실험은 OpenAI 내부 모델과 OpenAI production data 중심이다.
* 따라서 다른 architecture, open-source model, non-reasoning model, tool-using agent 환경에서도 같은 효과가 나는지는 불확실하다.

### 가정의 한계

* 논문은 각 safety category에 대해 비교적 명확한 policy spec과 judge/RM을 구축할 수 있다고 가정한다.
* 실제 환경에서는 policy 자체가 모호하거나, jurisdiction/context에 따라 허용 범위가 달라질 수 있다.

### 평가 방식의 한계

* 주요 평가는 reasoning model autograder와 human evaluation에 의존한다.
* autograder는 policy에 잘 맞춰져 있을 수 있지만 benchmark overfitting 또는 judge bias 가능성이 있다.
* human evaluation은 policy-free라 현실적이지만, 평가자별 safety 기준 차이가 클 수 있다.

### 방법론적 한계

* 최종 reward가 `h_i · s_i`인 구조는 직관적이지만, safety score calibration이 매우 중요하다.
* safety RM이 borderline harmful content를 높게 평가하면 helpfulness reward가 unsafe detail을 강화할 수 있다.
* 반대로 safety RM이 과도하게 보수적이면 safe-completion이 다시 refusal-like behavior로 수렴할 수 있다.

### 방어 논의의 한계

* 논문은 safe-completion 자체가 robust mitigation이라고 주장하지만, adaptive adversary가 safe-completion boundary를 탐색하는 경우는 충분히 다루지 않는다.
* 예를 들어 사용자가 단계적으로 high-level answer를 구체화하거나, safe template의 빈칸을 메우도록 유도하는 multi-turn attack에 대한 분석은 부족하다.

### 질문

* Safe-completion reward는 multi-turn setting에서 누적 actionability를 어떻게 평가하는가?
* 답변 하나하나는 safe하지만 여러 답변을 조합하면 harmful procedure가 완성되는 경우는 어떻게 다루는가?
* Safety RM의 severity calibration은 어떤 데이터와 기준으로 이루어졌는가?
* “meaningful facilitation” threshold는 category별로 얼마나 일관되게 operationalized되는가?
* Safe-completion이 jailbreak robustness를 실제로 높이는지, 아니면 surface-level refusal behavior만 바꾸는지는 별도 adaptive attack evaluation이 필요하지 않은가?

---

## 내 판단 (My Assessment)

* **설득력** : 높음
* **중요도** : 높음
* **새로움** : 중간~높음
* **재현 가능성** : 중간
* **실제 위험성** : 높음

### 가장 강한 부분

* 기존 refusal paradigm의 실패 모드를 매우 명확히 잡아낸다.
* 특히 dual-use prompt에서 “prompt intent”가 아니라 “output actionability”를 중심으로 봐야 한다는 주장은 AI safety evaluation 관점에서 중요하다.
* 자동 평가, controlled ablation, production model comparison, human evaluation, biorisk case study를 함께 제시해 empirical story가 꽤 강하다.

### 가장 약한 부분

* OpenAI 내부 pipeline, data, policy spec, reward model에 크게 의존하기 때문에 외부 재현성이 제한적이다.
* safe-completion의 핵심인 safety/helpfulness RM 설계와 calibration 세부사항이 충분히 공개되어 있지 않다.
* adaptive jailbreak 또는 multi-turn composition attack에 대한 평가는 상대적으로 부족하다.

### 내가 특히 기억할 점

* 안전한 모델은 단순히 “잘 거절하는 모델”이 아니라, **위험한 세부사항을 제거하면서도 사용자의 legitimate goal에 도움이 되는 방향으로 응답하는 모델**이어야 한다.
* Safety는 input classification 문제가 아니라 output control 문제라는 framing이 중요하다.

### 이 논문을 인용한다면 어떤 목적으로 쓸 것인가

* 배경 설명 : refusal-based safety training의 한계와 dual-use brittleness 설명
* 관련 연구 비교 : Deliberative Alignment, RBR, RLHF/DPO, Safe-RLHF와 비교
* 방법론 참고 : safety reward와 helpfulness reward를 결합한 output-centric RL objective
* 취약점 사례 : intent가 모호한 prompt에서 refusal-trained model이 actionable detail을 제공하는 사례
* 방어 필요성 근거 : dual-use 및 biorisk 영역에서 hard refusal보다 safe-completion이 더 적절하다는 근거
* 벤치마크 / 평가 기준 : helpfulness given safe output, harm severity distribution, balance metric

---

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

### 직접적으로 연결되는 부분

* Jailbreak 및 safety evaluation 연구에서 “refusal 여부”만 보는 평가는 부족하다.
* 이 논문은 **unsafe completion severity**와 **safe-but-helpful redirection**을 평가 축으로 추가해야 함을 보여준다.
* 특히 compositional jailbreak, intention deception, multi-modal jailbreak 연구와 연결하면, 공격 성공률을 단순 policy violation이 아니라 **meaningful facilitation 정도**로 재정의할 수 있다.

### 가져다 쓸 수 있는 아이디어

* Jailbreak evaluation에서 response를 다음처럼 분류할 수 있다.

  * hard refusal
  * safe redirection
  * high-level safe answer
  * borderline operational answer
  * fully actionable harmful answer
* 공격 성공률도 binary ASR 대신 harm severity-weighted ASR로 측정할 수 있다.
* Defense 평가에서는 over-refusal rate뿐 아니라 **indirect helpfulness**도 측정할 수 있다.

### 비교 대상 / baseline으로 쓸 수 있는 부분

* Refusal-trained baseline vs safe-completion-trained model
* Binary safety classifier vs output actionability grader
* Hard refusal policy vs meaningful facilitation policy
* Safety-only metric vs safety-helpfulness balance metric

### 내가 확장해볼 수 있는 부분

* Multi-turn safe-completion robustness 평가
* Safe-completion을 우회하는 adaptive prompt attack
* Safe template을 단계적으로 operational detail로 바꾸는 compositional attack
* Modality가 추가된 경우, 예를 들어 image, code, file, tool output이 actionability를 높이는지 분석
* Safe-completion response의 latent leakage 또는 implicit procedural cue 분석

### 후속 연구 질문

* Safe-completion은 adversarial prompt optimization에 대해 refusal training보다 robust한가?
* Safe-completion 모델은 “부분적으로 안전한 답변”을 제공하다가 multi-turn에서 누적적으로 위험 정보를 제공하는가?
* Harm severity grader와 human-perceived danger는 얼마나 일치하는가?
* Output-centric training을 VLM, agent, tool-use setting에 적용하면 새로운 failure mode가 생기는가?

---

## Action Items

### 평가 관련

* 기존 jailbreak benchmark에 harm severity score를 추가한다.
* ASR을 binary violation rate뿐 아니라 severity-weighted ASR로 재정의한다.

### 분석 관련

* refusal-trained model과 safe-completion-style model의 response를 actionability dimension으로 비교한다.
* dual-use prompt에서 detail level, specificity, procedurality, troubleshooting 제공 여부를 annotation한다.

### 방어 실험 관련

* safe redirection이 multi-turn adversary에게 정보 scaffold로 활용되는지 테스트한다.
* high-level answer가 후속 질문과 결합될 때 harmful completion으로 누적되는지 평가한다.

### 재현 / 구현 관련

* 공개 모델 대상으로 간단한 safe-completion preference dataset을 만들어 DPO/RLHF style fine-tuning을 시도한다.
* safety score와 helpfulness score를 별도 LLM judge로 산출하고, `h · s` 또는 constrained optimization 형태로 비교한다.

### 후속 문헌 조사

* Deliberative Alignment
* Rule-Based Rewards for Language Model Safety
* Constitutional AI
* Safe-RLHF
* XSTest / over-refusal evaluation
* dual-use jailbreak 및 many-shot jailbreak 관련 연구
