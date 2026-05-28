---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.2. AI Safety, 1.1.2.2.2. Jailbreak]
title: "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack"
tags: [Jailbreak, Jailbreak Attack, VLM, AI Safety, Paper review, Safety training, Multi-turn attack, Black box attack, Safety alignment]
---
# Review Summary

## Basic information

* Published : arXiv:2404.01833v3, 2025-02-26
* Title : [Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack](https://arxiv.org/abs/2404.01833)
* Code / Project page : Crescendomation is released as part of PyRIT, `https://github.com/Azure/PyRIT`
* Main keywords : Multi-turn jailbreak, black-box attack, LLM safety alignment, Crescendo, automated red teaming, adaptive prompting, benign prompt escalation, Crescendomation, AdvBench, HarmBench 

## One-line summary

이 논문은 **LLM safety alignment가 단일-turn refusal에는 강하지만 multi-turn benign interaction을 통한 점진적 escalation에는 취약할 수 있음**을 다루며, **Crescendo라는 multi-turn jailbreak와 이를 자동화한 Crescendomation**을 통해 **GPT-4, Gemini-Pro, Claude, LLaMA 계열 등 여러 모델에서 높은 jailbreak 성공률과 기존 기법 대비 우수한 성능**을 보인다.

---

## 주요 기여 (Key Contributions)

### 기여 1: 문제 제기 / 취약점 발견

* 기존 jailbreak 연구가 주로 **single-turn prompt, adversarial suffix, 명시적 malicious instruction**에 집중한 반면, 이 논문은 **겉보기에는 benign한 multi-turn 대화만으로 safety boundary를 점진적으로 약화시킬 수 있음**을 보인다.
* 핵심 취약점은 모델이 **자신이 직전 turn에서 생성한 내용과 대화 맥락을 강하게 따르는 성향**에 있다.
* 특히 **chat history가 유지되는 일반적인 assistant setting**에서 이 문제가 발생한다.

### 기여 2: 방법론 / 공격 / 평가 프레임워크 제안

* 논문은 **Crescendo**라는 multi-turn jailbreak 방법을 제안한다.

* 핵심 아이디어는 다음과 같다.

  `추상적이고 benign한 질문 + 모델이 생성한 관련 맥락의 재사용 → 점진적 escalation → 최종 harmful / policy-violating output 유도`

* 기존 one-shot jailbreak나 adversarial suffix 방식과 비교했을 때, Crescendo는 **명시적 악성 문자열이 적고, white-box access가 필요 없으며, 사람이 읽기에 자연스러운 질문들로 구성**된다는 차이가 있다.

### 기여 3: 접근 권한 또는 threat model 완화

* 공격자는 **model weights, logits, gradients, system prompt, encoder access** 없이도 공격을 수행할 수 있다.
* 필요한 것은 사실상 **일반적인 black-box chat 또는 API access**뿐이다.
* 이는 실제 배포 환경에서 위험성을 높인다. 대부분의 상용 LLM 서비스는 사용자가 여러 turn에 걸쳐 대화할 수 있고, history를 유지하기 때문이다.

### 기여 4: 실험 조건 또는 trigger 유형 비교

* 논문은 다음 조건들을 비교한다.

  * Manual Crescendo vs automated Crescendomation
  * GPT-4, GPT-3.5, Gemini-Pro, Gemini-Ultra, Claude-2/3, LLaMA-2/3 등 여러 모델
  * AdvBench subset, HarmBench, 논문 자체 task set
  * 기존 jailbreak baseline: MSJ, PAIR, CoA, CIA
  * backtracking step 수, multi-turn round 수
  * model size: LLaMA-2 70B vs 7B
  * transferability: 한 모델에서 생성한 Crescendo sequence를 다른 모델에 적용
  * defense: Self-Reminder, Goal Prioritization
* 이를 통해 **single-turn 중심 alignment가 충분하지 않으며, multi-turn context accumulation과 model-generated context가 핵심 취약 원인**임을 분석한다.

### 기여 5: 정량적 결과

* AdvBench subset 50개 task에서 Crescendomation은 GPT-4에 대해 평균 ASR 56.2%, binary ASR 98.0%를 달성한다.
* Gemini-Pro에서는 평균 ASR 82.6%, binary ASR 100.0%를 보인다.
* 기존 기법 대비 GPT-4에서는 29–61%, Gemini-Pro에서는 49–71% 높은 성능을 보였다고 보고한다.
* HarmBench 100개 random task에서도 Crescendo는 평균 ASR 63.2%, binary success 91%로 MSJ의 평균 ASR 38.9%, binary success 70%보다 높다.

---

## 방법 요약 (Method Summary)

### 문제 설정

* 입력 : 사용자와 LLM 사이의 multi-turn 대화
* 출력 : 원래는 refusal되어야 할 harmful, unsafe, misinformation, hate, self-harm, violence, explicit content 등
* 공격자 / 평가자 목표 : 명시적으로 금지된 요청을 직접 던지지 않고, 모델이 점진적으로 관련 내용을 생성하도록 유도
* 가정 :

  * 공격자는 black-box access만 가진다.
  * 대화 history가 유지된다.
  * 모델은 이전 응답을 context로 사용한다.
* 제약 조건 :

  * prompt는 대체로 benign하고 human-readable해야 한다.
  * adversarial suffix, gradient optimization, weight access는 사용하지 않는다.
  * 자동화 버전은 attack LLM과 judge LLM에 의존한다.

### 핵심 메커니즘

논문의 방법은 크게 다음 단계로 구성된다.

1. **Benign opening**
   공격 목표와 느슨하게 관련된 일반적·역사적·추상적 질문으로 시작한다.

2. **Model-generated context accumulation**
   모델이 스스로 생성한 답변을 다음 turn의 기반으로 사용한다.

3. **Progressive escalation**
   다음 질문은 직전 답변의 일부를 참조하면서 조금 더 구체적이거나 강한 표현을 요청한다.

4. **Target behavior induction**
   충분한 관련 context가 쌓이면, 모델이 원래 직접 요청에는 거부했을 내용을 생성한다.

5. **Backtracking / rephrasing**
   모델이 거부하거나 필터가 작동하면 해당 질문을 제거하고 다른 방식으로 다시 질문한다.

### 모델 / 시스템 구조상 중요한 지점

* 취약점이 발생하는 위치 : **multi-turn context window와 instruction-following behavior**
* 관찰 또는 조작하는 representation : 직접 내부 representation을 조작하지 않고, **대화 history와 모델 자신의 prior output**을 조작한다.
* safety mechanism과 충돌하는 지점 : safety mechanism은 주로 **현재 user input의 명시적 위험성**에 반응하지만, Crescendo는 위험성을 여러 turn에 분산시킨다.
* 실패가 전파되는 경로 :
  `초기 benign 응답 → 관련 맥락 축적 → 모델이 자기 응답을 신뢰 / 이어쓰기 → refusal threshold 약화 → unsafe completion`

### 기존 방법과의 차이

* 기존 접근 :

  * one-shot jailbreak
  * DAN류 role-play prompt
  * adversarial suffix optimization
  * many-shot malicious examples
  * prompt optimization with attacker LLM
* 이 논문의 접근 :

  * 대화형, 점진적, multi-turn
  * 명시적으로 악성 prompt를 던지기보다 모델이 만든 맥락을 재활용
  * black-box setting에서 작동
* 실질적인 차이 :

  * 탐지하기 어렵다.
  * input filter만으로 막기 어렵다.
  * 실제 chat product의 interaction pattern과 더 가깝다.

---

## 실험 설정 및 결과 (Experiments & Results)

### 대상 모델 / 시스템

* GPT-4 / ChatGPT
* GPT-3.5
* Gemini-Pro
* Gemini-Ultra
* Claude-2
* Claude-3 / Claude-3 Opus
* Claude-3.5 Sonnet
* LLaMA-2 70B / 7B
* LLaMA-3 70B Chat

### 데이터셋 / 벤치마크

* 논문 자체 task set 15개

  * illegal activity
  * self-harm
  * misinformation
  * pornography
  * profanity
  * sexism
  * hate speech
  * violence
* AdvBench subset 50개 task
* HarmBench random 100개 task

### 평가 지표

* Judge LLM 기반 success flag
* Judge score, 0–100
* Attack Success Rate, ASR
* Binary ASR: 여러 번 시도 중 하나라도 성공하면 성공으로 간주
* Perspective API score
* Azure Content Filter score
* refusal count
* minimum successful turn count

### 주요 결과

* Manual Crescendo는 평가된 대부분의 모델과 task에서 성공했다.
* Crescendomation은 GPT-4와 Gemini-Pro에서 기존 baseline보다 높은 평균 ASR과 binary ASR을 보였다.
* 대부분의 task는 평균적으로 5 turn 이하에서 jailbreak된다고 보고한다.
* misinformation 계열 task, 특히 climate/election 관련 task에서 높은 성공률을 보인다.
* self-harm denial task도 여러 모델에서 매우 높은 성공률을 보인다.
* explicit content와 manifesto류 task는 상대적으로 더 어려운 경우가 있었다.
* LLaMA-2 70B와 7B의 취약성 패턴이 상당히 비슷하게 나타나, 이 논문에서는 model size와 Crescendo vulnerability가 단순히 비례하지 않는다고 해석한다.

### 중요한 Figure / Table

* **Figure 1** :

  * 보여주는 내용 : 직접 위험한 요청을 하면 refusal되지만, Crescendo 방식으로 대화하면 같은 목표에 도달할 수 있는 예시.
  * 왜 중요한가 : 논문의 핵심 직관, 즉 “직접 요청은 막히지만 multi-turn escalation은 통과한다”를 보여준다.

* **Table 2** :

  * 보여주는 내용 : manual Crescendo가 ChatGPT, Gemini, Claude, LLaMA 계열에서 여러 task에 성공하는지 여부.
  * 왜 중요한가 : Crescendo가 특정 모델에 국한된 취약점이 아니라 다양한 aligned LLM에서 재현되는 패턴임을 보여준다.

* **Figure 4 / Figure 5** :

  * 보여주는 내용 : 관련 context가 누적될수록 특정 target token 또는 compliance response의 확률이 증가하는 현상.
  * 왜 중요한가 : Crescendo가 단순한 prompt trick이 아니라 context accumulation에 의해 모델의 다음-token 분포가 변하는 현상과 관련 있음을 뒷받침한다.

* **Table 4 / Figure 6** :

  * 보여주는 내용 : AdvBench subset에서 Crescendo가 MSJ, PAIR, CoA, CIA보다 높은 ASR을 달성.
  * 왜 중요한가 : 기존 jailbreak baseline 대비 정량적 우위를 보여주는 핵심 결과다.

* **Figure 13** :

  * 보여주는 내용 : Self-Reminder와 Goal Prioritization 방어에 대한 성능.
  * 왜 중요한가 : 현재 방어가 일부 task에서는 ASR을 낮추지만, multi-turn adaptive attack을 안정적으로 막지는 못함을 보여준다.

### Ablation / 추가 분석

* 제거하거나 바꾼 요소 :

  * Crescendo sequence의 일부 turn 제거
  * 모델이 생성한 표현을 직접적인 user phrase로 대체
  * backtracking step 수 변경
  * round 수 변경
  * 가장 영향력 있는 sentence 제거
* 결과 변화 :

  * 중간 turn을 생략하면 성공 확률이 크게 낮아진다.
  * 모델이 생성한 referent를 쓰는 대신 사용자가 명시적으로 위험 단어를 쓰면 성공률이 크게 낮아진다.
  * backtracking은 task-dependent하게 성능 향상에 기여한다.
  * round 수 증가는 일부 보완 효과가 있지만 backtracking만큼 효과적이지 않은 경우가 있다.
* 해석 :

  * Crescendo의 핵심은 단일 prompt 문구가 아니라 **대화 맥락의 누적과 모델 self-conditioning**이다.

### Negative results / 실패한 조건

* 잘 작동하지 않은 방법 :

  * GPT-4 대상 일부 highly sensitive task에서는 모든 자동화 기법이 실패한 사례가 있다.
  * LLaMA-2 70B에서 Crescendomation은 Manifesto와 Explicit task에 실패했지만, manual Crescendo는 성공했다고 보고한다.
  * Self-Reminder / Goal Prioritization 적용 시 일부 task에서 ASR이 크게 낮아졌다.
* 저자의 설명 :

  * Crescendomation은 Crescendo의 한 자동화 구현일 뿐이며, 자동화 실패가 Crescendo 자체의 한계를 의미하지는 않는다.
  * 공격 모델과 judge 모델의 alignment가 자동화 성능을 제한할 수 있다.
* 내가 보기엔 가능한 원인 :

  * 자동화된 attack LLM이 안전 정책 때문에 충분히 공격적인 escalation을 만들지 못할 수 있다.
  * judge 기반 평가가 false positive / false negative를 만들 수 있다.
  * 특정 task는 content filter와 refusal boundary가 더 강하게 작동한다.
  * multi-turn context가 길어질수록 모델별 context handling 차이가 커진다.

---

## 장점 및 시사점 (Advantages & Learnings)

### 시사점 1

* 이 논문은 **single-turn refusal robustness**만으로는 **실제 chat setting의 safety**를 보장하기 어렵다는 점을 보여준다.
* 따라서 alignment 평가에는 **adaptive multi-turn adversarial interaction**이 반드시 포함되어야 한다.

### 시사점 2

* **대화 history와 모델이 스스로 생성한 context**는 단순한 부가 정보가 아니라, 모델 행동을 결정하는 핵심 control surface로 작동한다.
* 특히 safety mechanism이 user prompt만 강하게 검사하고 assistant-generated context를 덜 엄격하게 다루면 취약점이 생긴다.

### 시사점 3

* **composition과 transferability** 때문에 공격 sequence가 완전히 모델별로만 작동하는 것은 아니다.
* 일부 Crescendo sequence는 다른 모델에도 전이되며, 이는 red-teaming artifact가 재사용 가능할 수 있음을 시사한다.

### 시사점 4

* 기존 방어 방식인 **keyword filtering, direct refusal tuning, input-only moderation, single-turn benchmark**에는 한계가 있다.
* Crescendo는 개별 prompt가 benign해 보일 수 있기 때문에, 단일 입력 단위의 탐지는 구조적으로 어렵다.

### 시사점 5

* 이 논문은 향후 **multi-turn safety evaluation, conversation-level moderation, context-aware refusal, assistant-output-aware safety training**이 중요하다는 점을 보여준다.

---

## 한계 및 의문점 (Limitations & Questions)

### 실험 범위의 한계

* 실험 대상은 강력하지만, 평가 시점의 특정 모델 버전에 제한되어 있다.
* closed-source model은 지속적으로 업데이트되므로 결과가 장기적으로 그대로 유지된다고 보기 어렵다.
* 실제 제품 환경의 system prompt, memory, tool use, enterprise policy layer까지 포함한 end-to-end 평가와는 차이가 있다.

### 가정의 한계

* 논문은 **multi-turn history가 유지되는 chat setting**을 전제로 한다.
* history가 짧거나 turn별로 강한 independent moderation이 적용되는 시스템에서는 효과가 달라질 수 있다.
* Crescendomation은 target model API access뿐 아니라 attack model과 judge model access도 필요하다.

### 평가 방식의 한계

* 주요 평가는 LLM judge에 크게 의존한다.
* Secondary Judge와 manual review를 도입했지만, 여전히 false positive / false negative 가능성이 있다.
* Perspective API와 Azure Content Filter는 misinformation처럼 다루기 어려운 category를 충분히 포괄하지 못한다.

### 방법론적 한계

* Crescendomation은 GPT-4를 attack model로 사용한다.
* 따라서 자동화 성능은 attack model의 능력, safety policy, prompt-following quality에 의존한다.
* 논문이 보여준 것은 “Crescendo를 자동화할 수 있다”이지, 최적의 자동화 알고리즘을 제시했다는 것은 아니다.

### 방어 논의의 한계

* 논문은 prefiltering, Crescendo data를 활용한 alignment, input/output filtering 등을 논의하지만, 안정적이고 일반적인 방어법을 충분히 검증하지는 않는다.
* Self-Reminder와 Goal Prioritization도 실험하지만, conversation-level defense의 설계 공간은 더 넓다.

### 질문

* assistant-generated content를 safety-critical context로 간주해 재평가하면 Crescendo를 얼마나 줄일 수 있을까?
* turn-level moderation보다 conversation-level trajectory moderation이 얼마나 효과적일까?
* Crescendo에 대한 adversarial training은 일반 helpfulness를 얼마나 손상시킬까?
* attack model 없이 사람이 만든 Crescendo와 자동 Crescendomation의 실제 위험도 차이는 어느 정도일까?
* tool-using agent나 RAG system에서는 Crescendo가 retrieval, tool call, planner state를 통해 더 강해질 수 있을까?

---

## 내 판단 (My Assessment)

* **설득력** : 높음
* **중요도** : 높음
* **새로움** : 중간~높음
* **재현 가능성** : 중간
* **실제 위험성** : 높음

### 가장 강한 부분

* 이 논문은 LLM safety에서 자주 과소평가되는 **multi-turn interaction surface**를 매우 명확하게 보여준다.
* Crescendo의 강점은 “강한 adversarial string”이 아니라 **정상 대화처럼 보이는 점진적 escalation**에 있다.
* 특히 모델 자신의 output을 다음 공격 단계의 발판으로 삼는 구조가 중요하다.

### 가장 약한 부분

* LLM judge 기반 성공 판정에 의존하는 정도가 크다.
* 일부 결과는 모델 업데이트에 따라 빠르게 변할 수 있다.
* Crescendomation의 구체적 prompt와 자동화 세부 구현이 성능에 큰 영향을 줄 가능성이 높다.
* 방어는 상대적으로 얕게 다뤄져 있으며, 근본적 mitigation보다는 문제 제기에 가깝다.

### 내가 특히 기억할 점

* **Jailbreak는 prompt 하나의 문제가 아니라 conversation trajectory의 문제**다.
* safety boundary는 사용자의 현재 입력뿐 아니라 **모델이 이전에 생성한 안전하지 않은 intermediate context**에 의해서도 약화될 수 있다.
* multi-turn benchmark는 고정된 dataset만으로 만들기 어렵다. 다음 turn이 target model의 응답에 의존하기 때문이다.

### 이 논문을 인용한다면 어떤 목적으로 쓸 것인가

* 배경 설명 : single-turn jailbreak evaluation의 한계를 설명할 때
* 관련 연구 비교 : adversarial suffix, PAIR, MSJ, CoA, CIA와 multi-turn benign escalation을 비교할 때
* 방법론 참고 : adaptive multi-turn red-teaming framework를 설계할 때
* 취약점 사례 : assistant-generated context가 safety failure를 유발하는 사례로
* 방어 필요성 근거 : input-only filter와 single-turn refusal tuning의 한계를 주장할 때
* 벤치마크 / 평가 기준 : AdvBench / HarmBench 기반 multi-turn ASR 평가를 구성할 때

---

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

### 직접적으로 연결되는 부분

* AI Safety 관점에서 이 논문은 **alignment가 static property가 아니라 interaction-dependent property**임을 보여준다.
* 특히 배포된 assistant의 안전성은 개별 response classifier가 아니라 **대화 전체의 state transition**으로 평가해야 한다.

### 가져다 쓸 수 있는 아이디어

* Multi-turn red-teaming을 단순 prompt search가 아니라 **trajectory search problem**으로 모델링하기
* assistant-generated content를 별도의 위험 신호로 추적하기
* refusal을 단일 label이 아니라 “trajectory-level safety invariant”로 정의하기
* backtracking을 포함한 adaptive adversary를 benchmark에 포함하기

### 비교 대상 / baseline으로 쓸 수 있는 부분

* Crescendo / Crescendomation
* PAIR
* MSJ
* CoA
* CIA
* Self-Reminder
* Goal Prioritization

### 내가 확장해볼 수 있는 부분

* Crescendo에 대한 **defense-side state machine** 설계
* multi-turn conversation에서 위험도가 누적되는지 측정하는 **context risk score**
* model-generated harmful intermediate representation을 탐지하는 moderation
* RAG / tool-use / agent planning 환경에서 Crescendo-style escalation 평가
* Crescendo-resistant instruction hierarchy 또는 memory policy 설계

### 후속 연구 질문

* Conversation-level safety monitor는 어떤 granularity로 작동해야 하는가?
* 모델의 prior assistant messages를 “trusted context”로 취급하는 것이 얼마나 위험한가?
* Crescendo-style 공격은 helpfulness optimization과 어떤 방식으로 충돌하는가?
* multi-turn adversarial training은 over-refusal을 얼마나 증가시키는가?
* 모델 내부 activation에서 Crescendo escalation의 phase transition을 관찰할 수 있는가?

---

## Action Items

### 평가 관련

* 현재 사용 중인 safety benchmark에 **adaptive multi-turn jailbreak** 항목을 추가한다.
* single-turn ASR과 별도로 **trajectory-level ASR, turn-to-success, refusal recovery rate**를 측정한다.

### 분석 관련

* 모델이 언제부터 unsafe direction으로 기울어지는지 turn별로 logit / refusal probability / judge score를 추적한다.
* assistant-generated context와 user-generated context를 분리해 위험도 기여도를 분석한다.

### 방어 실험 관련

* input-only moderation, output-only moderation, conversation-level moderation을 비교한다.
* Self-Reminder류 suffix defense가 장기 multi-turn에서 얼마나 유지되는지 실험한다.
* assistant response 자체를 다음 turn의 risk feature로 넣는 detector를 테스트한다.

### 재현 / 구현 관련

* PyRIT 기반으로 Crescendomation-style red-team loop를 구성한다.
* 직접적인 harmful content 생성을 피하면서, policy-safe surrogate task로 escalation dynamics를 먼저 재현한다.
* judge model의 false positive / false negative를 줄이기 위해 human audit subset을 만든다.

### 후속 문헌 조사

* PAIR: Jailbreaking Black Box Large Language Models in Twenty Queries
* MSJ: Many-Shot Jailbreaking
* HarmBench
* Goal Prioritization defense
* Self-Reminder defense
* Multi-round automatic red-teaming / MART 계열 연구