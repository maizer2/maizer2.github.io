---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.2. AI Safety, 1.1.2.2.2. Jailbreak]
title: "Jailbreaking Frontier Foundation Models Through Intention Deception"
tags: [Jailbreak, Jailbreak Attack, VLM, AI Safety, Paper review, Safety training, Multi-turn attack]
---

# Review Summary

## Basic information

- **Published** : arXiv v1 submitted 2026-04-27; arXiv page notes "Accepted at CVPR 2026 Findings Track." ([arXiv][1])
- **Title** : [Jailbreaking Frontier Foundation Models Through Intention Deception](https://arxiv.org/abs/2604.24082)
- **Authors** : Xinhe Wang, Katia Sycara, Yaqi Xie. ([arXiv][1])
- **Code / Project page** : 공식 저자 코드 저장소는 확인되지 않음. arXiv의 Code/Data/Media 섹션은 외부 code finder 링크만 제공하며, CatalyzeX도 "Request Code" 상태로 보인다. ([arXiv][1])
- **Main keywords** : Multi-turn jailbreak, LVLM/VLM safety, safe completion, intention deception, para-jailbreaking, conversational consistency, black-box attack, harmful alternative leakage

## One-line summary

이 논문은 **safe-completion 기반 frontier LLM/VLM이 다중 턴 대화에서 표면상 선의적인 의도에 속아 harmful auxiliary information을 누출하는 취약점**을 다루며, **iDecep이라는 intention-deception 기반 explore-then-exploit 공격 프레임워크**를 통해 **직접 jailbreak뿐 아니라 "para-jailbreaking"이라는 간접 누출 실패 모드가 GPT-5, Claude-Sonnet-4.5 등 강한 모델에서도 유의미하게 발생함**을 보인다. ([arXiv][2])

---

## 주요 기여 (Key Contributions)

### Safe completion의 새로운 취약점: para-jailbreaking 제시

- 기존 safety training은 unsafe input에 대한 hard refusal에서 output-centric safe completion으로 이동하고 있으며, 논문은 이 전환이 "안전한 대안 답변"이라는 새로운 공격면을 만든다고 주장한다. ([arXiv][2])
- 특히 모델이 직접 harmful request에 답하지 않더라도, 대안 설명이나 보조 정보 `y_alt` 안에 exploitable information이 들어가면 외부 judge 관점에서는 실패가 된다. 논문은 이를 **para-jailbreaking**으로 정의한다. ([arXiv][2])

### iDecep: intention deception 기반 multi-turn 공격 프레임워크 제안

- 논문은 공격자가 실제 harmful goal `G`를 숨기고, 표면적으로는 방어적·보고서 작성·감사·교육 등 benign-looking intention `I`를 유지하면서 대화를 진행하는 **iDecep**을 제안한다. ([arXiv][2])
- 핵심 아이디어는 다음과 같다.

  `harmful goal G + benign cover intention I + multi-turn consistency pressure → direct 또는 para harmful disclosure 유도`

- Crescendo나 Chain-of-Attack처럼 처음에는 중립 질문으로 시작해 점진적으로 steering하는 방식과 달리, iDecep은 처음부터 목표 주제와 가까운 대화를 하되 그 목적을 선의적으로 framing한다. 논문은 이 차이가 safe-completion 모델에서 특히 효과적이라고 주장한다. ([arXiv][2])

### Black-box threat model에서도 작동

- 공격자는 victim model의 weights, gradients, internal states에 접근하지 않고 API-like black-box interaction만 사용한다. 관찰 가능한 것은 victim의 textual response이며, action은 다음 prompt 제출 또는 regeneration이다. ([arXiv][2])
- 이는 실제 배포 환경에서 위험도를 높인다. 공격자가 모델 내부를 몰라도, 대화 history와 모델의 이전 답변을 이용해 다음 query를 구성할 수 있기 때문이다.

### Text-only와 multimodal VLM 설정 비교

- 논문은 AdvBench text, ClearHarm, 그리고 benign image를 붙인 AdvBench-Vision을 평가한다. AdvBench-Vision은 harmful textual task마다 인터넷에서 수집한 benign image를 결합해 현실적인 multimodal interaction을 시뮬레이션한다. ([arXiv][2])
- 비교 조건은 크게 다음과 같다.
  - 공격 방법: Chain-of-Attack, Crescendo, iDecep
  - 공격 모델: Qwen-Plus, GPT-3.5-Turbo
  - victim model: GPT-4o, Gemini-2.5-Flash, Claude-Sonnet-4.5, GPT-5
  - 평가 환경: AdvBench text, ClearHarm, AdvBench-Vision
  - 성공 유형: Total SR, Direct SR, Para SR ([arXiv][2])

### 정량적으로 frontier 모델에서도 높은 성공률 보고

- AdvBench text에서 iDecep은 Qwen-Plus attacker 기준 GPT-4o 0.96, Gemini-2.5-Flash 0.98, Claude-Sonnet-4.5 0.59, GPT-5 0.63의 Total SR을 보고한다. 특히 GPT-5에서는 Direct SR 0.12, Para SR 0.51로, 성공의 대부분이 para-jailbreaking으로 나타난다. ([arXiv][2])
- ClearHarm에서도 Qwen-Plus attacker 기준 GPT-5 Total SR 0.63, Direct SR 0.11, Para SR 0.52가 보고되며, AdvBench-Vision에서는 GPT-5 Total SR 0.84, Direct SR 0.23, Para SR 0.61로 multimodal context가 위험을 더 키우는 경향을 보인다. ([arXiv][2])

---

## 방법 요약 (Method Summary)

### 문제 설정

- **입력** :
  - adversarial goal `G`
  - benign-looking intention `I`
  - multi-turn textual input `s_t`
  - 선택적으로 visual input `v_t`
  - 이전 대화 history `H_t`
- **출력** :
  - victim response `y_t`
  - direct answer component `y_direct`
  - alternative/safe-completion component `y_alt`
- **공격자 / 평가자 목표** :
  - 공격자는 victim이 직접 harmful answer를 내거나, 거절하면서도 alternative response 안에 harmful information을 포함하도록 유도한다.
  - 외부 judge는 response 전체를 보고 direct failure와 para failure를 분리해 평가한다. ([arXiv][2])
- **가정** :
  - victim은 safe-completion-style safeguard를 사용한다.
  - attacker는 black-box access만 가진다.
  - attacker의 query generator와 evaluator는 LLM/VLM prompting으로 구현된다. ([arXiv][2])
- **제약 조건** :
  - 내부 parameter, gradient, hidden state 접근 없음.
  - 제한된 multi-turn query budget.
  - 논문 실험은 주로 instructional harms에 초점을 둔다. ([arXiv][2])

### 핵심 메커니즘

논문의 방법은 크게 다음 단계로 구성된다.

1. **Benign intention 생성** — harmful goal `G`에 대해 겉보기에는 합법적·방어적·보고서형인 intention `I`를 생성한다.
2. **Exploration phase** — victim과 여러 턴 대화를 하며 benign cover를 강화하고, victim response에서 다음에 파고들 수 있는 candidate point를 수집한다.
3. **Candidate aggregation** — dialogue history를 요약하고, victim이 이미 제공한 safe-looking response 중 goal과 연결될 수 있는 부분을 branching node로 만든다.
4. **Exploitation / branching phase** — 각 candidate에 대해 follow-up query를 생성하고, internal evaluator가 성공 가능성을 판단하며, 필요하면 같은 턴에서 regenerate를 시도한다. 논문의 Algorithm 1/2가 이 explore-then-exploit 구조를 기술한다. ([arXiv][2])

### 모델 / 시스템 구조상 중요한 지점

- **취약점이 발생하는 위치** : 모델의 refusal 자체가 아니라, refusal과 함께 제공되는 "도움 되는 대안" 또는 safe-completion component에서 발생한다.
- **관찰 또는 조작하는 representation** : attacker는 hidden representation을 직접 보지 않고, 대화 history `H_t`, victim response `y_t`, 내부 attack state `Σ_t`를 이용한다. ([arXiv][2])
- **safety mechanism과 충돌하는 지점** : safe completion은 helpfulness를 유지하면서 policy를 지키려 한다. 하지만 "safe alternative"가 실제로는 harmful goal에 필요한 조각 정보를 제공할 수 있다. ([arXiv][2])
- **실패가 전파되는 경로** : benign narrative → 모델의 신뢰/일관성 압력 증가 → 이전 답변 일부를 bridge로 사용 → 더 구체적 follow-up → direct 또는 para harmful disclosure.

### 기존 방법과의 차이

- **기존 접근** : Crescendo류 공격은 초기에 malicious intent를 숨기고 중립적인 질문에서 시작한 뒤 점진적으로 harmful objective로 이동한다. Chain-of-Attack도 semantic-driven multi-turn steering을 사용한다. ([arXiv][2])
- **이 논문의 접근** : 처음부터 target topic 근처에 머물되, 역할·목적·문맥을 benign하게 구성한다. 또한 victim의 이전 응답에서 exploitable segment를 선택해 다음 query의 발판으로 삼는다. ([arXiv][2])
- **실질적인 차이** : 표면적으로는 대화가 topic-consistent하고 legitimate하게 보이므로, input-level refusal classifier나 단일 턴 safety filter로는 감지하기 어렵다. 동시에 safe-completion의 "helpful alternative" 경로까지 공격 대상으로 삼는다.

---

## 실험 설정 및 결과 (Experiments & Results)

### 대상 모델 / 시스템

- Victim models:
  - GPT-4o
  - Gemini-2.5-Flash
  - Claude-Sonnet-4.5
  - GPT-5
- Attacker models:
  - Qwen-Plus
  - GPT-3.5-Turbo ([arXiv][2])

### 데이터셋 / 벤치마크

- **AdvBench text** : 10개 harmful category, category당 10 task, 총 100 task.
- **ClearHarm** : Chemical, Biological, Nuclear, Cybersecurity 네 domain, domain당 25 task, 총 100 task.
- **AdvBench-Vision** : AdvBench task에 benign image를 결합한 multimodal benchmark. ([arXiv][2])

### 평가 지표

- **Total SR** : direct success + para success를 포함한 전체 attack success rate.
- **Direct SR** : 모델이 직접 harmful content를 출력한 비율.
- **Para SR** : 직접 답변은 거절하거나 안전해 보이지만, alternative response에 harmful information이 포함된 비율. ([arXiv][2])

### 주요 결과

- **Text setting** : iDecep은 기존 multi-turn baselines보다 훨씬 높은 Total SR을 보인다. 예를 들어 Qwen-Plus attacker 기준 AdvBench text에서 GPT-5는 Chain-of-Attack 0.02, Crescendo 0.02에 머무는 반면, iDecep은 0.63이다. ([arXiv][2])
- **Para-jailbreaking의 중요성** : GPT-5와 Claude-Sonnet-4.5처럼 direct completion이 억제된 모델에서 Para SR이 크게 나타난다. GPT-5의 경우 AdvBench text에서 Qwen-Plus attacker 기준 Para SR 0.51, GPT-3.5-Turbo attacker 기준 Para SR 0.60이다. ([arXiv][2])
- **Multimodal amplification** : AdvBench-Vision에서 iDecep은 Qwen-Plus attacker 기준 GPT-5 Total SR 0.84, Para SR 0.61을 기록한다. 이는 benign image가 단순 부가정보가 아니라 benign narrative를 강화하는 context artifact로 작동할 수 있음을 시사한다. ([arXiv][2])
- **Category-wise 결과** : AdvBench category-wise 분석에서 iDecep은 GPT-4o, Gemini-2.5-Flash에서 거의 saturation에 가까운 성공 수를 보이며, GPT-5와 Claude-Sonnet-4.5에서도 baseline 대비 큰 차이를 보인다. ([arXiv][2])
- **ClearHarm domain-wise 결과** : Cybersecurity domain은 상대적으로 더 높은 성공 수를 보이고, Biological/Chemical domain은 더 엄격한 safeguard 때문에 낮지만 여전히 non-trivial success가 보고된다. ([arXiv][2])

### 중요한 Figure / Table

- **Figure 2**
  - 보여주는 내용 : malicious goal `G`에서 benign intent `I`를 만들고, query generation, victim response, safeguard evaluation, self-evaluation, external judge가 연결되는 iDecep attacker 구조.
  - 왜 중요한가 : 이 논문의 핵심이 단순 prompt가 아니라 **feedback-guided multi-turn attack loop**라는 점을 시각적으로 보여준다.
- **Table 1**
  - 보여주는 내용 : AdvBench text와 ClearHarm에서 Total SR / Direct SR / Para SR 비교.
  - 왜 중요한가 : para-jailbreaking이 특히 GPT-5, Claude-Sonnet-4.5 같은 robust model에서 주요 실패 모드라는 주장의 핵심 근거다. ([arXiv][2])
- **Table 2**
  - 보여주는 내용 : AdvBench-Vision에서 Chain-of-Attack과 iDecep 비교.
  - 왜 중요한가 : benign image를 포함한 multimodal context에서 iDecep의 성공률이 크게 증가함을 보여준다. ([arXiv][2])
- **Table 8**
  - 보여주는 내용 : ClearHarm의 Chemical/Biological/Nuclear/Cybersecurity domain별 successful task 수.
  - 왜 중요한가 : high-stakes domain별 취약성이 균일하지 않으며, 방어 평가도 domain-specific decomposition이 필요함을 보여준다. ([arXiv][2])

### Ablation / 추가 분석

- **제거하거나 바꾼 요소** : 정식 ablation보다는 attacker model, victim model, modality, benchmark category를 바꾼 비교 분석에 가깝다.
- **결과 변화** : 강한 attacker인 Qwen-Plus뿐 아니라 약한 GPT-3.5-Turbo attacker에서도 iDecep이 유의미하게 작동한다. 논문은 이를 attacker model의 raw capability보다 intention-deception mechanism 자체가 중요하다는 근거로 해석한다. ([arXiv][2])
- **해석** : 공격 성공의 핵심은 "더 똑똑한 attacker"라기보다, safe-completion 모델이 유지하려는 helpfulness와 dialogue consistency를 악용하는 구조적 취약점이다.

### Negative results / 실패한 조건

- **잘 작동하지 않은 방법** : Chain-of-Attack과 Crescendo는 Claude-Sonnet-4.5와 GPT-5에서 대부분 실패하거나 매우 낮은 SR을 보인다. 예컨대 Table 1에서 Claude-Sonnet-4.5는 baseline들이 거의 0.00이고, GPT-5도 0.00–0.03 수준이다. ([arXiv][2])
- **저자의 설명** : 기존 공격은 safe-completion의 alternative leakage를 충분히 겨냥하지 못하고, robust model의 direct refusal에 막힌다.
- **내가 보기엔 가능한 원인** : 기존 방법은 "직접 답변 유도"에 가까운 성공 기준에 최적화되어 있어, refusal-with-alternative 상태를 exploit target으로 삼는 설계가 부족하다. 반면 iDecep은 바로 이 중간 상태를 공격 surface로 만든다.

---

## 장점 및 시사점 (Advantages & Learnings)

### Safe completion은 hard refusal보다 항상 안전한 것은 아니다

- 이 논문은 **input intent 판별**만으로는 jailbreak를 막기 어렵다는 기존 문제의식을 넘어, **output-centric safe completion**도 alternative answer 경로에서 실패할 수 있음을 보여준다. ([arXiv][2])
- 따라서 safety evaluation은 "거절했는가?"가 아니라 "거절하면서 무엇을 제공했는가?"까지 봐야 한다.

### Multi-turn context는 안전성 평가의 핵심 단위다

- 공격은 단일 prompt가 아니라 대화 전체의 narrative를 조작한다.
- 따라서 guardrail도 per-turn classifier가 아니라 session-level risk accumulation, intent hypothesis tracking, disclosure budget 같은 구조를 가져야 한다.

### 모델의 conversational consistency가 공격면이 된다

- iDecep은 victim의 이전 답변 일부를 bridge로 사용해 다음 query를 만든다. 즉, 모델이 coherent하고 helpful하게 대화를 이어가려는 성향이 나중의 harmful disclosure를 위한 stepping stone이 된다. ([arXiv][2])

### 멀티모달 입력은 benign narrative를 강화할 수 있다

- AdvBench-Vision 결과는 benign image가 공격을 약화시키기보다, 오히려 legitimacy를 강화하는 context로 작동할 수 있음을 보여준다. ([arXiv][2])
- VLM safety에서는 image 자체의 harmfulness뿐 아니라 image가 대화 의도 해석에 주는 framing effect도 평가해야 한다.

### Para-jailbreaking은 별도 metric과 방어가 필요하다

- direct harmful answer만 잡는 benchmark는 safe-completion 시대의 실패 모드를 과소평가한다.
- 앞으로는 answer decomposition, alternative-content auditing, refusal-adjacent leakage detection이 중요해진다.

---

## 한계 및 의문점 (Limitations & Questions)

### 실험 범위의 한계

- 실험은 AdvBench, ClearHarm, AdvBench-Vision의 100-task sampling에 기반한다. 논문도 specific goal types, 특히 instructional harms에 초점을 둔다고 명시한다. ([arXiv][2])
- 따라서 non-instructional harms, persuasion, fraud escalation, agentic tool-use 환경까지 일반화되는지는 추가 검증이 필요하다.

### 가정의 한계

- black-box access라는 점은 현실적이지만, attacker가 여러 턴 동안 충분히 대화를 이어가고 regeneration까지 사용할 수 있다는 조건은 실제 제품 UX, rate limit, monitoring 정책에 따라 달라질 수 있다.
- 또한 benign image를 인터넷에서 가져오는 설정은 현실적이지만, 이미지 선택 전략 자체의 ablation은 충분히 분리되어 있지 않다.

### 평가 방식의 한계

- 외부 judge가 direct/para harmfulness를 판정한다. 하지만 judge model의 calibration, false positive/false negative, domain별 민감도 차이가 최종 SR에 큰 영향을 줄 수 있다.
- para-jailbreaking은 정의상 "직접 답변은 아니지만 유해한 보조 정보"를 판정해야 하므로, judge prompt와 policy boundary가 특히 중요하다.

### 방법론적 한계

- query generator와 evaluator도 LLM/VLM prompting에 의존한다. 논문은 이 모듈들이 bias와 stochasticity를 도입할 수 있다고 인정한다. ([arXiv][2])
- 공식 코드가 없는 상태라 재현 시 prompt details, judge settings, sampling parameters, regeneration budget이 결과를 크게 흔들 수 있다.

### 방어 논의의 한계

- 논문은 para-jailbreaking을 막기 위한 dedicated mitigation 필요성을 주장하지만, 실제 방어 알고리즘을 체계적으로 제안하거나 검증하지는 않는다. 결론에서도 향후 연구로 direct/para-jailbreaking 방어를 제시한다고만 말한다. ([arXiv][2])

### 질문

- Para SR은 judge model이 바뀌면 얼마나 안정적인가?
- `y_alt`를 어떤 granularity로 분해해야 para-harm을 정확히 측정할 수 있는가?
- Safe-completion 모델에서 helpfulness reward를 낮추지 않고 para leakage만 줄일 수 있는가?
- Multi-turn intent tracking을 도입하면 false refusal가 얼마나 증가하는가?
- Benign image의 semantic relevance, realism, source credibility가 attack success에 미치는 영향은 무엇인가?

---

## 내 판단 (My Assessment)

- **설득력** : 높음
- **중요도** : 높음
- **새로움** : 중간-높음
- **재현 가능성** : 중간
- **실제 위험성** : 높음

### 가장 강한 부분

- "모델이 거절했으니 안전하다"는 관점을 무너뜨리고, safe-completion의 alternative channel을 별도 실패 모드로 정의한 점이 강하다.
- Direct SR과 Para SR을 분리한 metric 설계가 특히 유용하다. 이 분해는 future benchmark design에 바로 가져다 쓸 수 있다.

### 가장 약한 부분

- 공식 코드와 상세 재현 세팅이 부족해 보인다.
- judge reliability와 para-harm boundary가 명확히 검증되지 않으면, 결과가 평가자 모델의 정책 해석에 의존할 수 있다.
- 방어 실험이 거의 없어서 "취약점 발견 논문"으로는 강하지만 "해결책 논문"으로는 약하다.

### 내가 특히 기억할 점

- **Refusal is not enough; safe alternatives can be the leakage channel.**
- Safe-completion 모델의 핵심 위험은 직접 답변 실패보다, "도움 되려고 제공한 주변 정보"가 공격 목표에 충분히 유용해지는 경우일 수 있다.

### 이 논문을 인용한다면 어떤 목적으로 쓸 것인가

- **배경 설명** : safe-completion 시대의 jailbreak threat model이 hard refusal 우회에서 alternative leakage로 확장되고 있음을 설명할 때.
- **관련 연구 비교** : Crescendo, Chain-of-Attack 등 multi-turn jailbreak와 비교할 때.
- **방법론 참고** : multi-turn explore-then-exploit attacker, direct/para decomposition, conversation-level judge 설계 참고.
- **취약점 사례** : VLM/LLM이 benign narrative와 consistency pressure에 의해 harmful auxiliary content를 누출하는 사례.
- **방어 필요성 근거** : per-turn moderation이나 direct refusal 평가가 충분하지 않다는 근거.
- **벤치마크 / 평가 기준** : Total SR을 Direct SR과 Para SR로 분해하는 평가 기준.

---

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

### 직접적으로 연결되는 부분

- AI safety 관점에서 이 논문은 multi-turn alignment failure, intent inference, disclosure auditing, VLM safety evaluation에 직접 연결된다.
- 특히 frontier model safety case에서 "모델이 직접 harmful instruction을 생성하지 않는다"는 주장만으로는 부족하다는 근거로 쓸 수 있다.

### 가져다 쓸 수 있는 아이디어

- Direct vs Para failure taxonomy
- Refusal-with-alternative auditing
- Session-level harmfulness scoring
- Benign-context multimodal stress test
- Model response segment를 follow-up risk candidate로 추출하는 evaluator

### 비교 대상 / baseline으로 쓸 수 있는 부분

- Crescendo
- Chain-of-Attack
- Text-only vs image-conditioned multi-turn attacks
- Qwen-Plus vs weaker attacker model 비교
- GPT-5/Claude-Sonnet-4.5 같은 robust model에서 Direct SR보다 Para SR이 큰 현상

### 내가 확장해볼 수 있는 부분

- Para-jailbreaking을 방어 관점에서 benchmark화.
- Safe-completion output을 `direct refusal`, `safe alternative`, `domain explanation`, `procedural detail`, `actionable detail` 등 finer-grained label로 분해.
- Multi-turn risk memory를 가진 guardrail을 설계하고, iDecep-style narrative attack에서 방어 성능 측정.
- Benign image의 role을 더 세밀하게 ablation: irrelevant image, semantically aligned image, authority-framing image, document screenshot 등.

### 후속 연구 질문

- Safe-completion 모델에서 "helpful alternative"의 안전성을 어떻게 formal verification 또는 policy checking할 수 있는가?
- 모델이 대화 초반에 형성한 benign intent hypothesis를 언제, 어떤 증거로 업데이트해야 하는가?
- Para leakage를 줄이는 방어가 helpfulness를 얼마나 희생하는가?
- Multi-turn VLM safety에서 image가 intent inference에 미치는 causal effect를 어떻게 분리 측정할 수 있는가?

---

## Action Items

### 평가 관련

- 기존 jailbreak eval에 **Direct SR / Para SR / Total SR** 분해를 추가한다.
- refusal response만 따로 모아 "alternative leakage"를 judge하는 secondary evaluation pass를 만든다.

### 분석 관련

- 모델 답변을 `direct answer`, `refusal`, `alternative`, `background explanation`, `actionable detail`로 segment-level annotation한다.
- harmfulness judge가 segment별로 어떤 근거에서 para-harm을 판정하는지 calibration set을 만든다.

### 방어 실험 관련

- session-level intent tracker를 붙여, benign narrative가 누적될 때 risk score가 어떻게 변하는지 측정한다.
- safe alternative generator 앞단 또는 후단에 "goal-relevance leakage filter"를 추가해 Para SR 감소 여부를 본다.

### 재현 / 구현 관련

- 공식 코드가 없으므로 먼저 paper의 Algorithm 1/2 수준에서 safe, non-operational reproduction harness를 만든다.
- AdvBench/ClearHarm task를 그대로 쓰되, harmful operational content를 저장하지 않는 redacted logging pipeline을 설계한다.
- attacker model, judge model, temperature, max turns, regeneration budget을 고정하고 sensitivity analysis를 한다.

### 후속 문헌 조사

- Crescendo, Chain-of-Attack, Reveal 등 multi-turn/VLM jailbreak 연구와 비교 정리.
- Safe-completion / output-centric safety training 관련 논문을 따로 묶어, hard refusal → safe completion 전환의 장단점을 정리.
- Conversational intent tracking, cumulative risk scoring, refusal alternative auditing 관련 방어 문헌을 조사.

[1]: https://arxiv.org/abs/2604.24082 "[2604.24082] Jailbreaking Frontier Foundation Models Through Intention Deception"
[2]: https://arxiv.org/pdf/2604.24082 "Jailbreaking Frontier Foundation Models Through Intention Deception"

---

# Main Review

## Abstract

* 현존하는 안전 훈련 접근법: 모델이 refusal boundary를 학습한다.
  * Refusal boundary란? 사용자의 의도에 기반한 안전과 불안전의 경계

* Refusal boundary의 문제점: 사용자 의도를 신뢰할 수 있게 평가할 수 없을 때, 특히 공격자가 자신의 의도를 숨길 경우 refusal boundary가 모호해진다.

* 최근 학습 방법(Safe completion): 안전 제약을 준수하면서 helpfulness(유용성)을 극대화 한다.

* Safe completion의 문제점: Intent inversion(사용자가 자신의 의도를 benign한 것처럼 행동)할 때, 악용될 수 있다.