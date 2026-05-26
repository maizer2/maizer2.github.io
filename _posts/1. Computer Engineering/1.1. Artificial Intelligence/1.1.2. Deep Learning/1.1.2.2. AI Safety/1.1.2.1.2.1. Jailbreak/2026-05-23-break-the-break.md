---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.2. AI Safety, 1.1.2.1.2.1. Jailbreak]
title: "Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization"
tags: [Jailbreak, Jailbreak Attack, VLM, AI Safety, Safety Alignment, Transferability, Gradient-based jailbreak, Paper review]
---

# Review Summary

* **Basic information** :

  * Published : arXiv preprint, submitted 2026-05-11. Comments: 17 pages, 8 figures, 6 tables. ([arXiv][1])
  * Title : *Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization* ([arXiv][1])
  * Code / Project page : 공개 구현은 확인되지 않음. CatalyzeX에는 “Request Code”만 표시됨. ([CatalyzeX][2])
  * Main keywords : VLM jailbreak, untargeted attack, entropy maximization, high-entropy decision tokens, KL regularization, transferability, multimodal safety.

* **One-line summary** :

  * 이 논문은 **기존 gradient-based VLM image jailbreak의 낮은 cross-model transferability가 실제 취약점 부재 때문인지, 아니면 targeted/prefix-constrained objective 때문인지**를 다루며, **refusal이 집중되는 high-entropy decision token의 entropy를 높이고 low-entropy structural token은 KL로 안정화하는 UJEM-KL**을 통해 **untargeted jailbreak가 white-box 성능과 transferability를 동시에 개선할 수 있음을 보인다**. ([arXiv][3])

---

## 주요 기여 (Key Contributions)

* **기여 1: 낮은 transferability에 대한 재해석**

  * 기존 연구가 “VLM universal image jailbreak는 잘 transfer되지 않는다”는 결론을 냈지만, 이 논문은 그 원인이 **모델 간 공유 취약점 부재**라기보다 **fixed prefix / response pattern을 강제하는 과도하게 constrained objective**일 수 있다고 주장한다. ([arXiv][3])
  * 특히 “Sure, here is …” 같은 특정 prefix를 강제하는 targeted jailbreak는 target model에서 동일한 token trajectory를 재현해야 하므로 architecture, tokenizer, decoding behavior 차이에 취약하다고 본다. ([arXiv][3])

* **기여 2: strictly untargeted multimodal jailbreak threat model 제안**

  * 논문은 fixed target string이나 fixed response format 없이, **외부 safety evaluator가 unsafe로 판정하는 어떤 유용한 응답이든 유도하면 성공**으로 보는 untargeted 설정을 formalize한다. ([arXiv][3])

  * 핵심 아이디어는 다음과 같다.

    `이미지 perturbation + 유해 instruction 유지 → refusal decision token의 local distribution flattening → non-refusal outcome 유도`

  * 기존 targeted method 대비 **출력 형태 제약이 약하고**, model별로 다른 non-refusal trajectory가 나와도 성공으로 인정되므로 transfer 가능성이 높아진다.

* **기여 3: refusal mechanism의 token-level 관찰**

  * 논문은 harmful prompt에서 refusal-indicative token이 전체 sequence에 균일하게 퍼져 있는 것이 아니라, **소수의 high-entropy decision token에 집중된다**고 관찰한다. ([arXiv][3])
  * 또한 attack 전에도 top-ranked candidate 안에 non-refusal token이 이미 존재하며, attack 후에는 예컨대 최상위 후보가 refusal token에서 non-refusal token으로 바뀔 수 있음을 보인다. ([arXiv][3])
  * 해석상 이 공격은 “새 취약점을 생성”한다기보다, **모델 내부에 이미 존재하는 non-refusal probability mass를 decision boundary에서 끌어올리는 방식**이다.

* **기여 4: UJEM-KL 방법론**

  * UJEM은 clean reference trajectory에서 entropy가 높은 content token subset `Sρ`를 decision set으로 선택하고, 그 위치의 entropy를 maximize한다. ([arXiv][3])
  * UJEM-KL은 여기에 KL regularization을 추가하여 decision set의 complement인 low-entropy structural positions `Rρ`에서 perturbed distribution이 clean distribution과 크게 달라지지 않도록 한다. ([arXiv][3])
  * 즉, **refusal decision은 흔들되 문장 구조와 fluency는 보존**하려는 설계다.

* **기여 5: 정량적 결과**

  * JailBreakV-28K와 SafeBench의 1,000-sample subset에서 Qwen2.5-VL-7B-Instruct, InternVL3.5-4B, LLaVA-1.5-7B를 평가한다. 평가는 Llama Guard, GPT-4o judge, HarmBench classifier가 모두 unsafe로 판정해야 성공으로 세는 conservative multi-judge intersection protocol을 사용한다. ([arXiv][3])
  * Main white-box 결과에서 UJEM-KL은 JailBreakV-28K 기준 Qwen 82.23, InternVL 83.67, LLaVA 88.32 ASR을 기록하고, SafeBench 기준 Qwen 67.39, InternVL 70.24, LLaVA 72.21 ASR을 기록한다. ([arXiv][3])
  * Cross-model transfer에서도 UJEM-KL은 거의 모든 source-target pair에서 UJEM과 baseline을 개선하며, 논문은 KL stabilization이 transfer setting에서 더 큰 효과를 보인다고 해석한다. ([arXiv][3])

---

## 방법 요약 (Method Summary)

* **문제 설정**

  * 입력 : image `x_img` + harmful text instruction `x_txt`.
  * 출력 : VLM의 autoregressive response.
  * 공격자 / 평가자 목표 : text prompt는 그대로 두고 image만 bounded perturbation하여, model이 refusal 대신 unsafe but usable response를 생성하도록 유도.
  * 가정 : source model에 대한 white-box access가 기본이며, transfer 실험에서는 source에서 만든 adversarial image를 target model에 직접 적용.
  * 제약 조건 : image perturbation은 `L∞` budget `ε = 8/255` 안에서 수행되며, main experiment는 100 optimization iterations를 사용한다. ([arXiv][3])

* **핵심 메커니즘**

  1. Clean input에서 reference response trajectory를 decoding한다.
  2. Teacher forcing으로 각 token position의 next-token entropy를 계산한다.
  3. content token 중 entropy 상위 `ρ = 0.2`를 decision set `Sρ`로 선택한다.
  4. `Sρ`에서는 entropy를 maximize하여 refusal/non-refusal margin을 무너뜨리고, 나머지 structural set `Rρ`에서는 KL regularization으로 clean distribution과의 drift를 억제한다. ([arXiv][3])

* **모델 / 시스템 구조상 중요한 지점**

  * 취약점이 발생하는 위치 : autoregressive decoding 중 refusal 여부가 갈리는 high-entropy token positions.
  * 관찰 또는 조작하는 representation : teacher-forced next-token distribution과 token-level entropy.
  * safety mechanism과 충돌하는 지점 : refusal token이 non-refusal token보다 약간 우세한 decision boundary.
  * 실패가 전파되는 경로 : decision token에서 refusal이 깨지면 이후 generation이 non-refusal trajectory로 이어지며, structural token이 불안정하면 반복·파편화·비문이 발생한다.

* **기존 방법과의 차이**

  * 기존 접근 : fixed harmful prefix, fixed response pattern, targeted response steering.
  * 이 논문의 접근 : 외부 judge 기준 unsafe이면 성공으로 보는 untargeted objective.
  * 실질적인 차이 : target model에서 동일 문구를 재현할 필요가 없고, refusal boundary만 넘기면 되므로 transferability가 개선된다. ([arXiv][3])

---

## 실험 설정 및 결과 (Experiments & Results)

* **대상 모델 / 시스템**

  * Qwen2.5-VL-7B-Instruct
  * InternVL3.5-4B
  * LLaVA-1.5-7B ([arXiv][3])

* **데이터셋 / 벤치마크**

  * JailBreakV-28K
  * SafeBench
  * Appendix에서는 HarmBench 결과도 추가로 다룸. ([arXiv][3])

* **평가 지표**

  * ASR, Attack Success Rate.
  * Main ASR은 Llama Guard, GPT-4o judge, HarmBench classifier가 모두 unsafe로 판정해야 성공으로 인정하는 intersection metric이다. 이 설계는 single judge가 fragmented 또는 borderline response를 과대평가하는 문제를 줄이기 위한 것이다. ([arXiv][3])

* **주요 결과**

  * UJEM-KL은 UJEM보다 일관되게 높다. SafeBench InternVL3.5에서는 UJEM 60.12에서 UJEM-KL 70.24로 약 +10.12 point 개선된다. ([arXiv][3])
  * Cross-model transfer에서 Qwen→LLaVA는 JailBreakV-28K 기준 UJEM 48.50에서 UJEM-KL 56.63, InternVL→LLaVA는 52.61에서 60.77, LLaVA→InternVL은 54.34에서 62.44로 오른다. ([arXiv][3])
  * Defense setting에서는 SafeBench 평균 기준 UJEM-KL이 No defense 70.1, SafeDecoding 65.3, adversarial training 61.2, UniGuard 32.7, R-TOFU 40.8을 기록한다. post-hoc guardrail이나 unlearning 계열에서는 공격 성공률이 크게 낮아진다. ([arXiv][3])

* **중요한 Figure / Table**

  * Figure 3 / 4 :

    * 보여주는 내용 : non-refusal token이 attack 전에도 top candidates에 있고, refusal mass가 high-entropy token set에 집중됨.
    * 왜 중요한가 : UJEM-KL의 핵심 전제, 즉 “refusal boundary는 소수 high-entropy decision point에서 흔들 수 있다”는 주장을 뒷받침한다. ([arXiv][3])

  * Table 1 :

    * 보여주는 내용 : white-box ASR에서 UJEM-KL이 대체로 baseline과 UJEM을 능가함.
    * 왜 중요한가 : untargeted entropy objective가 targeted jailbreak보다 약한 방법이 아니라, 오히려 competitive하거나 더 강할 수 있음을 보여준다. ([arXiv][3])

  * Table 2 :

    * 보여주는 내용 : source model에서 만든 adversarial image를 target model에 적용한 cross-model transfer ASR.
    * 왜 중요한가 : 논문 핵심 주장인 transferability 개선을 직접 검증한다. ([arXiv][3])

  * Table 5 :

    * 보여주는 내용 : KL weight `λ_KL`이 너무 작거나 너무 크면 성능이 떨어지고, 논문은 `0.01`을 사용한다.
    * 왜 중요한가 : entropy maximization과 structural preservation 사이의 trade-off가 실제로 존재함을 보여준다. ([arXiv][3])

* **Ablation / 추가 분석**

  * 제거하거나 바꾼 요소 : anti-refusal suppression, early stopping, KL stabilization.
  * 결과 변화 : UJEM + AR은 불안정하고 일부 모델에서 오히려 악화된다. Early stopping은 개선되며, KL stabilization이 가장 일관된 개선을 보인다. ([arXiv][3])
  * 해석 : explicit refusal-token suppression은 특정 token set에 과적합될 수 있고, KL은 출력 구조를 보존하면서 jailbreak signal을 유지하는 데 더 유리하다.

* **Negative results / 실패한 조건**

  * 잘 작동하지 않은 방법 : temperature-only manipulation은 sampling에서는 일부 효과가 있지만 greedy decoding에서는 argmax가 보존되어 한계가 있다. ([arXiv][3])
  * entropy-only UJEM은 refusal을 깨뜨릴 수 있지만, 반복·비문·fragmentary output을 만들 수 있다. ([arXiv][3])
  * 내가 보기엔 가능한 원인 : entropy를 모든 목적처럼 다루면 token distribution의 decision boundary뿐 아니라 discourse-level coherence까지 무너진다. 따라서 “attack success”와 “usable harmful completion”은 분리해서 평가해야 한다.

---

## 장점 및 시사점 (Advantages & Learnings)

* **시사점 1**

  * 이 논문은 **fixed prefix 기반 jailbreak 평가**만으로는 VLM의 transfer vulnerability를 과소평가할 수 있음을 보여준다.
  * 따라서 transferability를 논할 때는 targeted objective와 untargeted objective를 분리해야 한다.

* **시사점 2**

  * **High-entropy decoding positions**는 단순 uncertainty indicator가 아니라, safety refusal이 실제로 결정되는 boundary region일 수 있다.

* **시사점 3**

  * **Transferability**는 동일한 harmful text를 재현하는 능력이 아니라, 여러 모델에 공유되는 refusal-margin collapse 현상으로도 나타날 수 있다.

* **시사점 4**

  * 기존 방어 방식인 **logit filtering, constrained decoding, refusal heuristic**은 high-entropy decision token의 margin을 직접 공격하는 방법에 취약할 수 있다. 논문은 더 강한 방어로 unsafe probability mass 자체를 줄이는 unlearning/fine-tuning 또는 post-hoc filtering을 논의하지만, 각각 utility 손실·false positive·latency·distribution shift 문제가 있다고 본다. ([arXiv][3])

* **시사점 5**

  * 향후 VLM safety evaluation은 output-level refusal 여부뿐 아니라 **token-level entropy, refusal mass, decision-token margin, structural drift** 같은 내부적 지표를 포함해야 한다.

---

## 한계 및 의문점 (Limitations & Questions)

* **실험 범위의 한계**

  * 실험 대상은 3개 open-source VLM과 2개 주요 safety benchmark 중심이다. closed-source proprietary VLM, 더 큰 frontier VLM, tool-using multimodal agents로 일반화되는지는 불확실하다.
  * 논문 자체도 cross-model transfer가 여전히 어렵고, visual encoder, fusion mechanism, tokenizer, decoding behavior 차이가 transfer를 제한한다고 인정한다. ([arXiv][3])

* **가정의 한계**

  * 기본 공격은 source model white-box access를 전제로 한다. 실제 closed-source target만 존재하는 black-box 환경에서는 surrogate 선택, query budget, decoding configuration mismatch가 핵심 변수가 된다.
  * text instruction은 유지하고 image만 perturb한다는 threat model은 깔끔하지만, 실제 공격자는 text prompt와 image를 동시에 조작할 수 있으므로 더 넓은 threat model에서는 결과가 달라질 수 있다.

* **평가 방식의 한계**

  * multi-judge intersection은 precision을 높이지만 recall을 낮출 수 있다. 즉, 실제로 유해한 response가 하나의 judge에서만 누락되면 실패로 처리될 수 있다.
  * 반대로 GPT-4o judge를 포함한 LLM-as-a-judge는 version drift, policy drift, hidden calibration 변화에 취약할 수 있다.

* **방법론적 한계**

  * UJEM-KL은 teacher-forced reference trajectory, entropy ranking, tokenizer-dependent token positions에 의존한다.
  * 논문이 제안하는 future direction처럼, decision-token margin statistics나 coarse semantic alignment 같은 더 tokenizer-agnostic signal이 필요해 보인다. ([arXiv][3])

* **방어 논의의 한계**

  * defense 실험은 representative defenses를 포함하지만, adaptive defense나 jointly trained multimodal safety head, input-side perturbation detection, adversarial image purification에 대한 충분한 검증은 제한적이다.
  * UniGuard와 R-TOFU에서 ASR이 크게 낮아지는 점은 중요하지만, 이 방어들이 utility를 얼마나 손상하는지 같은 safety-utility frontier 분석은 더 필요하다.

* **질문**

  * high-entropy decision token은 model family 간 얼마나 정렬되는가?
  * refusal token mass를 낮추는 fine-tuning이 benign uncertainty calibration을 망가뜨리지 않을 수 있는가?
  * KL stabilization은 fluency를 보존하지만 harmfulness specificity도 같이 보존하거나 강화하는가?
  * closed-source VLM API에서 surrogate-only UJEM-KL이 어느 정도 transfer되는가?
  * image perturbation detection이나 preprocessing defense를 거치면 entropy landscape가 어떻게 변하는가?

---

## 내 판단 (My Assessment)

* **설득력** : 높음

* **중요도** : 높음

* **새로움** : 중간~높음

* **재현 가능성** : 중간

* **실제 위험성** : 중간~높음

* **가장 강한 부분**

  * “transfer가 안 된다”는 기존 결론을 objective design 관점에서 재해석한 점이 강하다. 특히 high-entropy decision token이라는 mechanistic hook을 통해 attack objective와 transferability를 연결한 점이 좋다.

* **가장 약한 부분**

  * closed-source black-box 환경으로의 일반화가 아직 약하다. white-box source optimization이 필요하고, tokenizer / decoding / judge 설정에 민감할 가능성이 있다.

* **내가 특히 기억할 점**

  * VLM refusal은 전체 response의 global property처럼 보이지만, 실제로는 소수 high-entropy decision token에서 결정될 수 있다.
  * Entropy-only attack은 refusal을 깨지만 fluency를 망칠 수 있고, KL stabilization은 “unsafe but usable” completion을 만들기 위한 핵심 장치다.

* **이 논문을 인용한다면 어떤 목적으로 쓸 것인가**

  * 배경 설명 : VLM jailbreak transferability 논쟁.
  * 관련 연구 비교 : targeted prefix jailbreak vs untargeted entropy-based jailbreak.
  * 방법론 참고 : token-level entropy 기반 decision point selection.
  * 취약점 사례 : multimodal image perturbation이 refusal margin을 붕괴시키는 사례.
  * 방어 필요성 근거 : surface-level refusal/logit heuristic 방어의 한계.
  * 벤치마크 / 평가 기준 : multi-judge intersection ASR protocol.

---

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

* **직접적으로 연결되는 부분**

  * AI safety 관점에서 이 논문은 refusal behavior를 output-level이 아니라 **token distribution geometry** 관점에서 봐야 한다는 근거를 제공한다.
  * 특히 refusal token margin, entropy concentration, structural drift는 mechanistic evaluation feature로 쓸 수 있다.

* **가져다 쓸 수 있는 아이디어**

  * Safety evaluation에서 “refusal/non-refusal top-k competition”을 metric화.
  * Fine-tuning 또는 decoding defense 전후로 high-entropy decision token의 location과 margin 변화를 추적.
  * Attack-free diagnostic으로 harmful prompt에서 non-refusal token mass가 top-k에 얼마나 남아 있는지 측정.

* **비교 대상 / baseline으로 쓸 수 있는 부분**

  * FigStep, UJA, SEA, Force와 UJEM/UJEM-KL을 targeted vs untargeted axis에서 비교 가능.
  * Temperature-only, UJEM entropy-only, UJEM-KL을 각각 decoding-only / entropy-only / entropy+stability baseline으로 둘 수 있다.

* **내가 확장해볼 수 있는 부분**

  * Closed-source VLM API에서 surrogate transfer 실험.
  * tokenizer-agnostic decision-token alignment metric 설계.
  * defense-aware entropy margin regularization.
  * harmfulness judge가 아닌 human-rated usability 기준과의 correlation 분석.

* **후속 연구 질문**

  * refusal decision token은 prompt family별로 안정적인가, 아니면 category-specific한가?
  * safety fine-tuning은 high-entropy token의 entropy를 낮추는가, 아니면 refusal token mass만 올리는가?
  * multimodal fusion layer의 어느 지점에서 image perturbation이 language token entropy로 전파되는가?

---

## Action Items

* **평가 관련**

  * UJEM-KL을 재현할 수 있다면, ASR뿐 아니라 refusal-token margin, entropy shift, output fluency score를 함께 기록.
  * single judge와 multi-judge intersection의 disagreement case를 따로 분석.

* **분석 관련**

  * harmful category별 high-entropy decision token 위치 분포를 추출.
  * clean vs perturbed에서 top-k token 후보가 어떻게 바뀌는지 시각화.

* **방어 실험 관련**

  * logit filtering / SafeDecoding류 방어가 refusal margin을 실제로 얼마나 넓히는지 측정.
  * unlearning 또는 targeted safety fine-tuning 후 benign task utility degradation도 함께 평가.

* **재현 / 구현 관련**

  * 공개 코드가 아직 확인되지 않으므로, 논문의 Algorithm / appendix 설정을 바탕으로 최소 재현부터 시작.
  * 우선 Qwen2.5-VL 또는 LLaVA 중 하나로 single-model white-box UJEM → UJEM-KL 순서로 구현하는 것이 현실적이다.

* **후속 문헌 조사**

  * prefix-targeted VLM jailbreak transfer 실패 연구.
  * entropy-guided adversarial attack 계열.
  * SafeDecoding, UniGuard, R-TOFU 같은 decoding-time / post-hoc / unlearning defense 계열.

[1]: https://arxiv.org/abs/2605.10764 "[2605.10764] Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization"
[2]: https://www.catalyzex.com/paper/break-the-brake-not-the-wheel-untargeted "Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization"
[3]: https://arxiv.org/pdf/2605.10764 "Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization"
