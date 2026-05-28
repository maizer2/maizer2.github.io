---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.2. AI Safety, 1.1.2.2.2. Jailbreak]
title: "JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models"
tags: [Jailbreak, Jailbreak Attack, VLM, AI Safety, Paper review]
---

# Review Summary

## Basic information

- **Published** : 18 Sept 2025, NeurIPS 2025 poster; Last Modified: 21 Apr 2026. ([OpenReview][1])
- **Title** : [JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models](https://openreview.net/forum?id=yg1yfaKolw)
- **Authors** : Jiaxin Song, Yixu Wang, Jie Li, Xuan Tong, Rui Yu, Yan Teng, Xingjun Ma, Yingchun Wang. ([OpenReview][1])
- **Code / Project page** : 공식 코드 페이지는 OpenReview 메타데이터에서 확인되지 않음. Supplementary Material ZIP은 제공됨. ([OpenReview][1])
- **Main keywords** : AI safety, adversarial attack and defense, jailbreak, multimodal learning; 추가적으로 latent safety boundary, ELK, cross-modal perturbation. ([OpenReview][1])

## One-line summary

이 논문은 **VLM의 fusion-layer latent representation에 존재하는 내부 safety decision boundary가 jailbreak에 악용될 수 있다는 문제**를 다루며, **Safety Boundary Probing + Safety Boundary Crossing을 통해 이미지 perturbation과 텍스트 suffix를 공동 최적화**하여 **white-box 및 black-box VLM에서 높은 ASR과 transferability를 보인다**. ([OpenReview][1])

---

## 주요 기여 (Key Contributions)

### 기여 1: VLM 내부 safety boundary를 jailbreak vector로 정의

- 기존 VLM jailbreak 연구가 주로 입력 이미지 또는 텍스트 perturbation 자체에 초점을 둔 반면, 이 논문은 **multimodal fusion layer의 latent space 안에 safe/unsafe를 가르는 내부 decision boundary가 있다**는 관점을 제시한다.
- 특히 ELK 관점에서, 모델이 최종 출력으로는 거부하더라도 내부 표현에는 unsafe response로 이어질 수 있는 safety-relevant signal이 남아 있다고 본다.

### 기여 2: JailBound = boundary probing + boundary crossing

- 논문은 **JailBound**라는 2단계 latent-space jailbreak framework를 제안한다.
- 핵심 아이디어는 다음과 같다.

  `unsafe image/text pair의 fusion representation + boundary normal direction 추정 → image/text perturbation을 jointly optimize → refusal boundary를 넘어 non-refusal / policy-violating output 유도`

- Safety Boundary Probing은 fusion layer별 logistic regression classifier로 decision hyperplane을 근사하고, Safety Boundary Crossing은 그 boundary 방향을 따라 이미지 perturbation과 텍스트 suffix를 함께 업데이트한다.

### 기여 3: black-box target으로의 transferability 제시

- 공격 생성은 white-box source VLM의 내부 representation 접근을 전제로 하지만, 최종 adversarial input은 GPT-4o, Gemini 2.0 Flash, Claude 3.5 Sonnet 같은 **black-box commercial VLM**으로도 transfer된다.
- 즉 target model의 weights, gradients, fusion layer access 없이도 transfer attack이 가능하다는 점이 실제 배포 환경 관점에서 중요하다.

### 기여 4: trigger / modality 조건 비교

- 논문은 다음 조건들을 비교한다.
  - `I0 + T0` : baseline, adversarial image/text 없음
  - `I0 + T1` : text-only attack
  - `I1 + T0` : image-only attack
  - `I1 + T1` : image/text를 따로 공격한 뒤 결합
  - `{I1, T1}` : iterative joint attack, 논문의 주 방법
- 결과적으로 단순 결합보다 **iterative joint optimization**이 더 강하며, 모델별로 text perturbation과 visual perturbation의 상대적 취약성이 다르다는 점을 보인다.

### 기여 5: 정량적 결과

- abstract 기준 평균 ASR은 **white-box 94.32%**, **black-box 67.28%**이며, 각각 SOTA 대비 **+6.17%p**, **+21.13%p** 높다고 주장한다. ([OpenReview][1])
- main experiment에서는 white-box Llama-3.2-11B, Qwen2.5-VL-7B, MiniGPT-4에서 각각 **94.38%, 91.40%, 97.19%** ASR을 보고한다.
- black-box transfer에서는 GPT-4o, Gemini 2.0 Flash, Claude 3.5 Sonnet에서 각각 **75.24%, 70.06%, 56.55%** ASR을 달성한다.

---

## 방법 요약 (Method Summary)

### 문제 설정

- **입력** : 원본 이미지 `Xv_raw`와 원본 텍스트 prompt `Xt_raw`.
- **출력** : adversarial image `Xv_adv = Xv_raw + δv`와 adversarial text `Xt_adv = [Xt_raw, Xt_suffix]`.
- **공격자 / 평가자 목표** : VLM의 fused representation `h = ϕ(xv, xt)`를 특정 target region으로 이동시켜 harmful / policy-violating output을 유도.
- **가정** : source white-box VLM에 대해 vision encoder, text embedding, fusion module의 representation을 관찰하고 gradient 기반 최적화를 수행할 수 있음.
- **제약 조건** : visual perturbation은 `L∞ ≤ 8/255`, text suffix는 20 tokens, optimization은 100–150 iterations로 설정됨.

### 핵심 메커니즘

논문의 방법은 크게 다음 단계로 구성된다.

1. MM-SafetyBench의 safe/unsafe multimodal sample을 VLM에 통과시켜 fusion layer별 hidden representation을 수집한다.
2. 각 fusion layer마다 logistic regression classifier를 학습해 safe/unsafe decision hyperplane을 근사한다.
3. hyperplane의 normal vector `v`와 boundary까지의 거리 `ε`를 계산해 perturbation 방향과 magnitude를 정의한다.
4. `Lalign`, `Lgeo`, `Lsem`을 결합한 total loss로 image perturbation과 text suffix를 jointly optimize한다.
5. 생성된 adversarial input을 white-box model 및 black-box target model에 평가한다.

### 모델 / 시스템 구조상 중요한 지점

- **취약점이 발생하는 위치** : vision encoder와 LLM backbone 사이의 multimodal fusion layer.
- **관찰 또는 조작하는 representation** : fused hidden representation `h = ϕ(xv, xt)`.
- **safety mechanism과 충돌하는 지점** : 모델의 출력 refusal는 안전하게 보일 수 있지만, 내부 latent state는 safe/unsafe boundary 근처에서 조작 가능하다는 점.
- **실패가 전파되는 경로** : image/text perturbation → fused representation shift → safety boundary crossing → decoder가 refusal 대신 non-refusal response 생성.

### 기존 방법과의 차이

- **기존 접근** : image-only 또는 text-only perturbation, 혹은 gradient 기반 최적화에 의존하되 명확한 target direction이 부족함.
- **이 논문의 접근** : fusion-layer boundary를 먼저 probe하고, 그 boundary normal direction을 따라 cross-modal perturbation을 유도함.
- **실질적인 차이** : 단순히 "출력이 harmful해지도록" 최적화하는 것이 아니라, 내부 safety boundary를 geometry로 모델링해 더 직접적인 방향성을 제공한다.

---

## 실험 설정 및 결과 (Experiments & Results)

### 대상 모델 / 시스템

- **White-box** : Llama-3.2-11B, Qwen2.5-VL-7B, MiniGPT-4.
- **Black-box transfer** : GPT-4o, Gemini 2.0 Flash, Claude 3.5 Sonnet.

### 데이터셋 / 벤치마크

- **MM-SafetyBench** 사용.
- 13개 prohibited content category와 1,719개 adversarial example로 구성되며, 각 sample은 unsafe visual content와 malicious prompt를 pairing한다.

### 평가 지표

- **ASR / Attack Success Rate** : 전체 attack attempt 중 성공 비율.
- 논문 appendix 기준으로는 model output에 predefined rejection template이 포함되지 않으면 성공으로 간주하는 string-matching 방식이다.
- **Semantic Preservation Score** : GPT-4o 및 human evaluation 기반 0–5 score. 다만 human annotator는 공저자이며, semantic utility 평가 시 factual correctness, ethical compliance, safety alignment를 의도적으로 무시하도록 설정되어 있다.

### 주요 결과

- White-box에서 `{I1, T1}` iterative joint attack은 전체 ASR 기준 Llama-3.2-11B **94.38%**, Qwen2.5-VL-7B **91.40%**, MiniGPT-4 **97.19%**를 기록한다.
- Black-box transfer에서는 GPT-4o **75.24%**, Gemini 2.0 Flash **70.06%**, Claude 3.5 Sonnet **56.55%**를 기록한다.
- multimodal baseline 비교에서는 UMK, FigStep, VAJM 대비 전반적으로 더 균형 잡힌 ASR을 보이며, 논문은 decision hyperplane modeling이 더 효율적인 adversarial optimization을 가능하게 한다고 해석한다.

### 중요한 Figure / Table

- **Figure 1** : JailBound 전체 pipeline. Initial unsafe image/text pair → boundary probing → perturbation constraint → boundary crossing → white-box/black-box jailbreak로 이어지는 구조를 보여준다. 중요성은 공격이 단순 prompt trick이 아니라 latent-space boundary manipulation이라는 점을 한눈에 보여준다는 데 있다.
- **Figure 2** : `Lalign`, `Lgeo`, `Lsem` 세 loss의 역할. `Lalign`은 target region crossing, `Lgeo`는 normal direction alignment, `Lsem`은 perturbation magnitude / suffix coherence 보존 역할을 한다.
- **Table 3** : white-box safety-critical category별 ASR. `{I1, T1}`가 대부분의 category와 model에서 가장 높다.
- **Table 4** : black-box transferability. GPT-4o/Gemini/Claude에서 transfer가 유지되지만 category와 target model에 따라 편차가 큼을 보여준다.
- **Figure 4** : ablation. loss component 제거가 ASR과 semantic preservation에 미치는 영향을 보여준다.

### Ablation / 추가 분석

- **제거하거나 바꾼 요소** : `Lalign`, `Lgeo`, `Lsem`.
- **결과 변화** : `Lalign` 제거 시 ASR은 150 iterations에서 **82.67%**, `Lgeo` 제거 시 **85.79%**로 감소한다. `Lsem` 제거는 ASR을 **95.21%**로 올리지만 semantic score가 **3.48**로 하락한다. full loss의 semantic score는 **4.67**로 가장 높다.
- **해석** : `Lalign`과 `Lgeo`는 attack direction과 convergence 안정성에 핵심이고, `Lsem`은 공격 성공률 자체보다 응답 relevance / semantic coherence를 유지하는 역할이 크다.
- **classifier configuration analysis** : full-layer classifier supervision은 100 iterations 내 **91.4%+ ASR**, last 10 layers는 약 **88.2%**, last layer only는 약 **82.8%**로, dense layer-wise guidance가 더 빠르고 강한 공격을 만든다.

### Negative results / 실패한 조건

- **잘 작동하지 않은 방법** : `w/o Lalign`, `w/o Lgeo`, last-layer-only guidance.
- **저자의 설명** : directional guidance가 줄어들면 boundary crossing이 불안정해지고, layer-wise semantic diversity를 활용하지 못해 convergence가 느려진다.
- **내가 보기엔 가능한 원인** : fusion representation의 safety signal은 단일 final layer보다 여러 depth에 분산되어 있을 가능성이 높다. 따라서 last-layer-only objective는 gradient signal이 너무 coarse하고 transferability도 약해진다.
- **black-box category 편차** : Claude 3.5 Sonnet에서 일부 category의 ASR이 낮게 나타나는데, 이는 target model의 refusal style, category-specific moderation, 또는 rejection-template metric과의 상호작용 때문일 수 있다.

---

## 장점 및 시사점 (Advantages & Learnings)

### 시사점 1: Latent safety boundary를 직접 평가해야 한다

- 이 논문은 **출력 refusal behavior만으로는 VLM safety를 충분히 평가하기 어렵다**는 점을 보여준다.
- 따라서 내부 fusion representation에서 safe/unsafe separability가 어떻게 형성되는지 보는 probing-based safety evaluation이 중요하다.

### 시사점 2: Cross-modal fusion은 부가 요소가 아니라 핵심 attack surface다

- image와 text는 독립적인 입력 channel이 아니라, fusion layer에서 하나의 safety-relevant context로 결합된다.
- 이 때문에 text-only / image-only evaluation은 실제 multimodal 위험을 과소평가할 수 있다.

### 시사점 3: Transferability는 shared multimodal pipeline의 취약성을 시사한다

- JailBound가 black-box VLM으로 transfer된다는 점은 서로 다른 VLM이 유사한 fusion-stage vulnerability를 공유할 가능성을 시사한다.
- 이는 benchmark나 red-teaming에서 "source model에서 만든 adversarial multimodal input이 다른 target에 얼마나 옮겨가는가"를 별도 축으로 봐야 함을 의미한다.

### 시사점 4: Output moderation / refusal tuning만으로는 부족하다

- ASR 평가가 refusal phrase 부재에 기반한다는 한계는 있지만, 논문이 보여주는 핵심은 output-level safety가 latent representation manipulation을 충분히 막지 못한다는 것이다.
- 따라서 output filtering, keyword filtering, text-only refusal tuning만으로는 cross-modal latent attack에 취약할 수 있다.

### 시사점 5: Defense research의 단위가 "prompt"에서 "representation trajectory"로 이동해야 한다

- 향후 방어는 prompt-level detector보다 fusion-layer representation shift, boundary margin, latent safety concept robustness를 함께 다뤄야 한다.
- 특히 adversarial training, representation regularization, cross-modal consistency check, latent refusal margin widening 같은 방향이 자연스럽다.

---

## 한계 및 의문점 (Limitations & Questions)

### 실험 범위의 한계

- white-box source model은 Llama-3.2-11B, Qwen2.5-VL-7B, MiniGPT-4 중심이고, black-box target은 세 commercial VLM으로 제한된다.
- 따라서 더 다양한 architecture, 더 최신 closed-source system, 실제 product moderation stack까지 일반화되는지는 불확실하다.

### 가정의 한계

- 핵심 공격 생성은 source VLM의 fusion representation과 gradients에 접근하는 white-box 조건을 전제로 한다.
- 실제 attacker가 이런 source model을 확보하지 못하거나 target과 architecture mismatch가 클 경우 transfer ASR은 달라질 수 있다.

### 평가 방식의 한계

- ASR은 rejection template string matching에 의존하므로, refusal이 paraphrase되거나 안전한 non-refusal이 포함되는 경우 false positive / false negative가 생길 수 있다.
- Semantic preservation human evaluation은 공저자 annotator 기반이며, scoring에서 factual correctness와 safety alignment를 제외한다. 이는 "공격 응답의 유용성"을 측정하려는 목적에는 맞지만, safety evaluation으로는 편향 가능성이 있다.
- NeurIPS checklist에서도 error bar나 confidence interval, statistical significance testing이 없다고 명시되어 있다.

### 방법론적 한계

- 제안 방법은 fusion layer representation을 관찰하고 layer-wise classifier를 학습하는 구조에 의존한다.
- end-to-end closed system, encrypted / hidden representation system, tool-augmented VLM, retrieval-augmented VLM에서는 동일하게 적용되기 어렵다.

### 방어 논의의 한계

- 저자도 fusion-layer latent attack에 특화된 방어를 충분히 탐구하지 않았고, perturbation budget도 fixed setting 위주라고 인정한다.
- 따라서 논문은 강한 공격 논문에 가깝고, 일반적 방어법까지 제시한 논문은 아니다.

### 질문

- fusion-layer boundary가 safety fine-tuning 이후 어떻게 이동하는가?
- representation-level adversarial training으로 `Lalign` / `Lgeo` 방향을 무력화할 수 있는가?
- refusal-template 기반 ASR 대신 human / model-judge / policy-grounded rubric을 쓰면 결과가 얼마나 달라지는가?
- black-box transfer가 architecture similarity 때문인지, shared training data / instruction tuning 때문인지 분리할 수 있는가?
- 더 강한 commercial moderation layer가 있는 production VLM에서도 같은 transfer pattern이 유지되는가?

---

## 내 판단 (My Assessment)

- **설득력** : 높음-중간
- **중요도** : 높음
- **새로움** : 중간-높음
- **재현 가능성** : 중간
- **실제 위험성** : 높음, 단 source white-box access가 필요한 생성 단계와 black-box transfer 단계는 구분해야 함

### 가장 강한 부분

- latent safety boundary라는 관점을 VLM jailbreak에 연결하고, 이를 실제 attack objective로 operationalize한 점.
- image/text를 따로 다루지 않고 fusion-centric joint optimization으로 설계한 점.
- white-box뿐 아니라 black-box transfer까지 보여줘 실제 risk framing이 강한 점.

### 가장 약한 부분

- ASR metric이 rejection-template string matching에 크게 의존한다.
- defense 실험이 부족하다.
- official code / project page가 명확하지 않고, 통계적 유의성이나 error bar가 없다.

### 내가 특히 기억할 점

- "안전성은 출력에서만 평가할 것이 아니라, fusion-layer latent boundary margin으로도 평가해야 한다."
- `Lsem`을 제거하면 ASR은 올라가지만 semantic preservation이 무너지는 trade-off가 있다.
- dense layer-wise guidance가 last-layer-only guidance보다 강력하다.

### 이 논문을 인용한다면 어떤 목적으로 쓸 것인가

- **배경 설명** : VLM safety alignment가 visual modality와 fusion representation 때문에 LLM보다 더 넓은 attack surface를 가진다는 근거.
- **관련 연구 비교** : image-only, text-only, typographic prompt, visual adversarial example 계열과 대비되는 latent-boundary-based attack.
- **방법론 참고** : fusion-layer probing, layer-wise classifier, boundary normal direction을 활용한 representation-space attack objective.
- **취약점 사례** : black-box VLM에도 transfer되는 cross-modal jailbreak 사례.
- **방어 필요성 근거** : output-level refusal tuning만으로는 latent-space manipulation을 막기 어렵다는 근거.
- **벤치마크 / 평가 기준** : MM-SafetyBench 기반 ASR 평가의 장점과 string-matching metric의 한계를 함께 논의하는 사례.

---

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

### 직접적으로 연결되는 부분

- AI safety 관점에서 이 논문은 **representation-level safety failure**를 다룬다.
- 특히 "모델이 내부적으로 무엇을 encode하는가"와 "출력 policy가 무엇을 말하게 하는가" 사이의 gap을 공격 표면으로 본다는 점이 alignment / interpretability 연구와 직접 연결된다.

### 가져다 쓸 수 있는 아이디어

- fusion layer별 safe/unsafe linear separability 측정.
- safety boundary margin을 robustness metric으로 정의.
- adversarial attack뿐 아니라 defense evaluation에서 `boundary distance`, `normal alignment`, `cross-modal consistency`를 측정.
- text-only refusal robustness와 multimodal latent robustness를 분리 평가.

### 비교 대상 / baseline으로 쓸 수 있는 부분

- `I0+T0`, `I0+T1`, `I1+T0`, `I1+T1`, `{I1,T1}` modality ablation.
- UMK, FigStep, VAJM과의 multimodal jailbreak 비교.
- last layer only vs last 10 layers vs full-layer classifier guidance.

### 내가 확장해볼 수 있는 부분

- defense-side probing: unsafe direction을 찾은 뒤, 그 방향으로 representation이 이동하지 않도록 regularization.
- transferability decomposition: source-target architecture similarity, tokenizer similarity, safety policy similarity를 분리.
- ASR metric 개선: rejection-template matching 대신 policy-grounded judge + human adjudication + harmfulness severity score.
- latent boundary shift 분석: safety fine-tuning 전후 boundary normal과 margin이 어떻게 바뀌는지 측정.

### 후속 연구 질문

- VLM safety boundary는 instruction tuning, RLHF/RLAIF, safety fine-tuning 단계 중 언제 가장 크게 형성되는가?
- 내부 boundary를 넓히는 defense가 model utility를 얼마나 손상시키는가?
- multimodal chain-of-thought, tool use, retrieval context가 추가되면 boundary crossing이 더 쉬워지는가?
- black-box transfer를 줄이려면 source-target shared representation을 줄이는 게 효과적인가, 아니면 output policy를 강화하는 게 효과적인가?

---

## Action Items

### 평가 관련

- MM-SafetyBench 또는 유사 safety benchmark에서 refusal-template ASR, model-judge harmfulness, human severity rating을 함께 비교.
- category별 ASR variance와 confidence interval을 추가해 통계적 안정성 확인.

### 분석 관련

- fusion layer별 safe/unsafe linear separability를 probe하고, early/mid/late layer에서 boundary margin을 시각화.
- safety fine-tuning 전후 boundary normal vector의 cosine similarity와 margin 변화를 측정.

### 방어 실험 관련

- representation regularization: unsafe direction으로의 movement를 penalty로 주는 defense 실험.
- adversarial training: JailBound-style perturbation으로 학습한 뒤 clean utility와 robust ASR 변화 측정.
- cross-modal consistency detector: image perturbation과 text suffix가 fusion layer에서 비정상적으로 alignment되는지 탐지.

### 재현 / 구현 관련

- 우선 Qwen2.5-VL-7B 또는 MiniGPT-4에서 layer-wise classifier probing만 재현.
- 그다음 `I0+T1`, `I1+T0`, `{I1,T1}`를 작은 subset으로 비교.
- 공식 코드가 명확하지 않으므로, supplementary ZIP과 논문 hyperparameter를 기준으로 최소 구현부터 시작.

### 후속 문헌 조사

- ELK / CCS 기반 latent knowledge probing.
- VLM jailbreak: FigStep, VAJM, UMK, compositional adversarial attacks.
- representation-level safety defense, refusal direction, activation steering, adversarial robustness for multimodal fusion layers.

[1]: https://openreview.net/forum?id=yg1yfaKolw "JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models — OpenReview"
