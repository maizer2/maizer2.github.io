---
layout: post
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.2. Deep Learning, 1.1.2.2. AI Safety, 1.1.2.1.2.1. Jailbreak]
title: "Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models"
tags: [Jailbreak, Jailbreak Attack, VLM, AI Safety, Safety Alignment, Paper review]
---

# Review Summary

## Basic information

- **Published** : ICLR 2024 Spotlight. OpenReview 기준 Published: 16 Jan 2024, Last Modified: 11 Mar 2024. ([OpenReview][1])
- **Title** : *Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models* ([OpenReview][1])
- **Code / Project page** : GitHub `erfanshayegani/Jailbreak-In-Pieces`; repo에는 `src/adv_image.py` 기반 CLIP embedding-space adversarial image optimization 코드가 포함되어 있음. ([GitHub][2])
- **Main keywords** : Adversarial attacks, Vision encoders, Jailbreak, Prompt Injection, Security, Embedding space attacks, Black-box, LLM, VLM, Cross-Modality alignment. ([OpenReview][1])

## One-line summary

이 논문은 **VLM에서 텍스트 alignment만으로는 막기 어려운 cross-modality jailbreak 취약점**을 다루며, **악성 의미를 이미지 embedding에 숨기고 benign textual prompt와 조합하는 compositional adversarial attack**을 통해 **LLaVA와 LLaMA-Adapter V2에서 높은 jailbreak 성공률과 prompt injection 가능성**을 보인다.

---

## 주요 기여 (Key Contributions)

### Cross-modality safety alignment 취약점 제기

- 기존 text-only jailbreak 방어는 명시적 유해 텍스트 prompt를 refusal / filtering으로 막을 수 있지만, VLM에서는 이미지 modality가 joint embedding context로 들어가면서 다른 attack surface가 생긴다.
- 논문은 benign-looking image + generic text prompt 조합이 textual alignment gate를 우회할 수 있음을 보인다. 저자들은 이를 "cross-modality alignment vulnerability"로 해석한다.

### Embedding-space adversarial image attack 제안

- 논문은 malicious trigger의 embedding에 가까운 adversarial image를 생성하는 방법을 제안한다.
- 핵심 아이디어는 다음과 같다.

  `benign generic prompt + adversarial image whose CLIP embedding is close to malicious trigger → VLM이 이미지 context를 유해 주제로 해석 → harmful generation`

- 기존 end-to-end white-box VLM attack과 달리 LLM output logits나 full VLM gradient가 아니라, 주로 vision encoder embedding matching을 사용한다.

### 더 약한 threat model: LLM access 없이 vision encoder access만 필요

- 공격자는 full white-box VLM, LLM weights, output logits에 접근하지 않아도 된다.
- 논문은 공격 생성이 보통 off-the-shelf vision encoder, 예컨대 CLIP에 대한 접근만으로 가능하다고 주장한다. 이는 closed-source VLM이 공개 vision encoder를 재사용할 때 특히 위험하다.

### 네 가지 malicious trigger 유형 비교

- 논문은 다음 조건들을 비교한다.
  - **Textual trigger** : harmful text를 CLIP text encoder로 embedding.
  - **OCR textual trigger** : 이미지 안에 harmful text/OCR 요소를 넣어 vision encoder로 embedding.
  - **Visual trigger** : harmful visual object/image를 vision encoder로 embedding.
  - **Combined OCR textual + visual trigger** : OCR text와 visual object를 함께 사용.
- 실험 결과, textual trigger는 거의 작동하지 않았고, image-based trigger 세 가지가 훨씬 높은 ASR을 보였다. 저자들은 textual trigger 실패를 CLIP joint embedding의 **modality gap**과 연결한다.

### 정량적 결과

- LLaVA에서 평균 ASR은 textual trigger 0.007에 불과한 반면, OCR textual 0.849, visual 0.849, combined 0.870이다.
- LLaMA-Adapter V2에서도 textual trigger 0.006에 비해 OCR textual 0.604, visual 0.608, combined 0.633으로 높다.
- 세 명의 annotator agreement는 Fleiss' Kappa = 0.8969로 보고된다.

---

## 방법 요약 (Method Summary)

### 문제 설정

- **입력** : 이미지 `x_i`와 텍스트 prompt `x_t`를 함께 받는 VLM.
- **출력** : VLM의 textual response.
- **공격자 / 평가자 목표** : 명시적으로 유해한 텍스트 prompt를 쓰지 않고, 이미지 embedding에 유해 context를 숨겨 VLM이 harmful response를 생성하게 만드는 것.
- **가정** : 공격자는 vision encoder 또는 유사한 CLIP encoder에 접근할 수 있다. LLM weights, full VLM gradients, output logits 접근은 필요하지 않다.
- **제약 조건** : adversarial image는 겉보기에는 benign해야 하며, embedding space에서는 malicious trigger와 가깝게 위치해야 한다.

### 핵심 메커니즘

논문의 방법은 크게 다음 단계로 구성된다.

1. 유해 prompt를 **generic textual instruction**과 **malicious trigger**로 분해한다.
2. malicious trigger를 네 가지 방식 중 하나로 joint embedding space의 target으로 만든다.
3. adversarial image를 최적화하여 그 embedding이 malicious trigger embedding에 가까워지도록 한다.
4. 최종 inference에서 adversarial image와 benign/generic prompt를 함께 입력하여 VLM이 이미지 context를 유해 주제로 해석하게 만든다.

### 모델 / 시스템 구조상 중요한 지점

- **취약점이 발생하는 위치** : vision encoder output이 projection layer를 거쳐 LLM context로 들어가는 joint embedding interface.
- **관찰 또는 조작하는 representation** : CLIP-style vision embedding 또는 projected visual representation.
- **safety mechanism과 충돌하는 지점** : text-only refusal/alignment는 generic prompt를 안전한 질의로 보지만, LLM은 visual context를 통해 유해 subject를 회수한다.
- **실패가 전파되는 경로** : adversarial image → vision encoder embedding → projected joint embedding → LLM context contamination → harmful continuation.

### 기존 방법과의 차이

- **기존 접근** : output-side target을 정하고 full VLM/LLM gradients를 통해 특정 harmful output likelihood를 높이는 white-box attack.
- **이 논문의 접근** : output을 직접 최적화하지 않고, input-side embedding space에서 malicious trigger와 adversarial image의 거리를 줄인다.
- **실질적인 차이** : prompt마다 새로 최적화하지 않아도 하나의 adversarial image가 여러 generic prompt와 조합될 수 있고, 같은 prompt도 여러 malicious image와 조합될 수 있다.

---

## 실험 설정 및 결과 (Experiments & Results)

### 대상 모델 / 시스템

- LLaVA.
- LLaMA-Adapter V2.

### 데이터셋 / 벤치마크

- OpenAI moderation 기준 8개 prohibited scenario를 사용한다.
- 각 category마다 8개 adversarial image, 네 가지 malicious trigger strategy, 2개 generic prompt, 25회 반복을 사용하여 총 6400 queries를 구성한다.
- Appendix에서는 AdvBench harmful prompt subset에 대해서도 추가 평가한다.

### 평가 지표

- Human-evaluated Attack Success Rate.
- Perspective API toxicity score.
- Toxic BERT / Toxic RoBERTa classifier score.
- Hidden prompt injection에서는 prompt divergence 또는 instruction following 성공 여부를 ASR로 측정한다.

### 주요 결과

- **Jailbreak ASR** : image-based triggers가 textual trigger보다 압도적으로 높다. LLaVA에서는 combined trigger가 평균 0.870, LLaMA-Adapter V2에서는 0.633을 기록한다.
- **Toxicity evaluation** : LLaMA-Adapter V2 output 기준 combined trigger가 Toxic RoBERTa 43.04, Perspective toxicity 46.74, severe toxicity 13.97로 가장 강한 harmful signal을 보인다.
- **Indirect hidden prompt injection** : LLaVA 평균 ASR은 visual 0.75, OCR 0.80, combined 0.81이고, LLaMA-Adapter V2는 각각 0.36, 0.40, 0.42이다.
- **Direct hidden prompt injection** : 일부 조건에서는 작동하지만 jailbreak / indirect prompt injection보다 낮고, temperature 0.1에서 더 잘 관찰된다. 예를 들어 "Never Stop" scenario에서 LLaVA 0.79, LLaMA-Adapter V2 0.12이다.

### 중요한 Figure / Table

- **Figure 1**
  - 보여주는 내용 : 네 가지 trigger, embedding matching optimization, benign prompt와 adversarial image의 compositional jailbreak 구조.
  - 왜 중요한가 : 논문 전체의 attack decomposition을 가장 명확하게 보여준다.
- **Table 1**
  - 보여주는 내용 : 네 trigger 유형별 jailbreak ASR.
  - 왜 중요한가 : textual trigger는 거의 실패하고 image-based trigger가 성공한다는 핵심 주장을 정량적으로 뒷받침한다.
- **Table 2**
  - 보여주는 내용 : toxicity classifier / Perspective API 기반 harmfulness score.
  - 왜 중요한가 : human ASR뿐 아니라 automatic toxicity metric에서도 combined trigger가 가장 강한 경향을 보인다.
- **Figure 2 / Figure 3**
  - 보여주는 내용 : context contamination과 extreme bias activation.
  - 왜 중요한가 : 한 번 jailbreak가 발생하면 이후 대화 context까지 오염되어 safety failure가 확산될 수 있음을 보여준다.
- **Figure 19 / Table 10**
  - 보여주는 내용 : one image-many prompts, one prompt-many images, 그리고 universal malicious trigger 평가.
  - 왜 중요한가 : 공격이 단발성이 아니라 compositional / reusable attack primitive로 작동할 수 있음을 보여준다.

### Ablation / 추가 분석

- **제거하거나 바꾼 요소** : malicious trigger type을 textual / OCR textual / visual / combined로 바꿈.
- **결과 변화** : textual trigger는 거의 실패, image-based trigger는 높은 ASR, combined trigger가 가장 강한 경향.
- **해석** : VLM의 safety failure는 단순히 "이미지에 유해 내용이 있다"가 아니라, cross-modal embedding alignment와 modality gap에서 발생한다.

### Negative results / 실패한 조건

- **Textual trigger 기반 adversarial image** : 거의 작동하지 않는다. 저자들은 CLIP의 image/text embedding이 분리되어 있어 textual target으로 optimized image가 model 입장에서는 out-of-distribution처럼 보일 수 있다고 해석한다.
- **Direct hidden prompt injection** : abstract instruction을 이미지에 숨기는 것은 real object를 숨기는 것보다 어렵다. 저자들은 CLIP training distribution이 주로 실제 객체 이미지 중심이고, VLM도 이미지를 instruction이 아니라 description 대상으로 학습하기 때문이라고 설명한다.
- **Bing/Bard 같은 closed-source system** : vision encoder와 fusion mechanism을 모르면 공격이 거의 random해지고, output-side monitoring filter도 추가 방어로 작동할 수 있다.

---

## 장점 및 시사점 (Advantages & Learnings)

### Text-only alignment의 한계

- 이 논문은 **textual refusal tuning / text-only jailbreak evaluation**만으로는 VLM safety를 보장하기 어렵다는 점을 보여준다.
- 따라서 safety alignment는 modality별로 따로 보는 것이 아니라, **full multimodal input pipeline** 전체에서 평가되어야 한다.

### Representation이 safety boundary가 된다

- 이미지 입력은 단순 부가 정보가 아니라 LLM이 답변할 subject와 context를 결정하는 핵심 representation이다.
- 특히 visual embedding이 LLM context로 주입되는 interface가 사실상 새로운 attack surface가 된다.

### Compositionality 때문에 공격 재사용성이 생긴다

- 하나의 adversarial image가 여러 prompt와 결합될 수 있고, 하나의 prompt가 여러 adversarial image와 결합될 수 있다.
- 이는 prompt-specific attack보다 운영 비용이 낮고, 평가 coverage가 어려워지는 방향이다.

### 기존 방어 방식의 한계

- keyword filtering이나 perplexity-based text filter는 generic prompt만 보면 악성을 탐지하기 어렵다.
- embedding-based safety filter도 unsafe concept coverage, threshold 선택, false positive 문제 때문에 일반 방어로 충분하지 않다.

### 향후 safety evaluation 축 추가

- VLM/MLLM 평가에는 다음 축이 필요하다.
  - benign text + adversarial image 조합 평가.
  - visual context contamination 평가.
  - OCR/visual/combined trigger별 modality-specific robustness.
  - long-context continuation에서 jailbreak persistence 측정.

---

## 한계 및 의문점 (Limitations & Questions)

### 실험 범위의 한계

- 실험 대상은 주로 LLaVA와 LLaMA-Adapter V2에 제한되어 있다.
- closed-source VLM에 대해서는 vision encoder 접근 불가능성과 output monitoring filter 때문에 직접적인 일반화가 제한된다.

### 가정의 한계

- 논문은 공격자가 vision encoder 또는 유사한 encoder에 접근할 수 있다는 가정을 둔다.
- 실제 배포 시스템이 proprietary encoder, image preprocessing, ensemble filtering, output moderation을 쓰면 효과가 달라질 수 있다.

### 평가 방식의 한계

- 주요 ASR은 human evaluation에 의존한다.
- automatic toxicity classifier는 harmfulness의 proxy일 뿐이며, safety violation의 맥락적 판단을 완전히 대체하기 어렵다.

### 방법론적 한계

- CLIP embedding matching이 핵심이므로, CLIP과 다른 vision encoder를 쓰는 모델에 대한 transferability는 불확실하다.
- abstract concept이나 instruction-like content를 숨기는 direct hidden prompt injection은 성공률이 낮다.

### 방어 논의의 한계

- embedding filter, adversarial training, preprocessing/postprocessing, output monitoring 등이 논의되지만, 안정적이고 일반적인 방어법으로 충분히 검증되지는 않는다.
- 특히 embedding filter는 threshold를 낮추면 false positive가 많아지고, 높이면 adversarial sample을 놓치는 trade-off가 있다.

### 질문

- CLIP이 아닌 SigLIP, EVA-CLIP, proprietary vision encoder에서도 동일한 compositionality가 유지되는가?
- multimodal instruction tuning 단계에서 image-as-context와 image-as-instruction을 분리하면 방어가 가능한가?
- adversarial image detection보다, LLM-side에서 visual evidence의 "safety provenance"를 추적하는 방식이 더 효과적인가?
- context contamination을 long-horizon dialogue safety benchmark로 정량화할 수 있는가?

---

## 내 판단 (My Assessment)

- **설득력** : 높음
- **중요도** : 높음
- **새로움** : 높음
- **재현 가능성** : 중간
- **실제 위험성** : 높음

### 가장 강한 부분

- "유해 prompt를 텍스트로 직접 쓰지 않고, 이미지 embedding이 subject/context를 공급한다"는 decomposition이 매우 강하다.
- 특히 Table 1의 ASR 차이는 modality gap과 cross-modal alignment failure를 설득력 있게 보여준다.

### 가장 약한 부분

- closed-source, production-grade VLM으로의 일반화는 제한적이다.
- 또한 방어 실험이 충분히 체계적이지 않아, 어떤 defense layer가 실제로 가장 효과적인지는 아직 명확하지 않다.

### 내가 특히 기억할 점

- VLM safety에서는 "prompt가 안전한가?"보다 "모든 modality가 결합된 context가 어떤 의미 영역을 형성하는가?"가 더 중요하다.
- 이미지가 단순 input이 아니라 LLM의 latent subject provider로 작동한다.

### 이 논문을 인용한다면 어떤 목적으로 쓸 것인가

- **배경 설명** : multimodal alignment가 text-only alignment보다 어렵다는 근거.
- **관련 연구 비교** : full white-box VLM attack과 embedding-space / encoder-only attack의 차이.
- **방법론 참고** : malicious trigger decomposition과 CLIP embedding matching.
- **취약점 사례** : benign text + adversarial image 조합의 jailbreak 사례.
- **방어 필요성 근거** : text filtering, refusal tuning, output-only moderation의 한계.
- **벤치마크 / 평가 기준** : trigger type별 ASR, context contamination, hidden prompt injection 평가.

---

## 내 연구 / 관심사와의 연결 (Relevance to My Work)

### 직접적으로 연결되는 부분

- multimodal safety evaluation에서 **input decomposition**과 **representation-level attack surface**를 분리해 봐야 한다는 점.
- VLM alignment failure를 "model output"이 아니라 "vision encoder → projection → LLM context" pipeline failure로 분석하는 관점.

### 가져다 쓸 수 있는 아이디어

- benign prompt + unsafe visual context 조합을 평가 suite로 만들기.
- textual / OCR / visual / combined trigger taxonomy를 다른 MLLM benchmark에 적용하기.
- context contamination 이후 후속 turn에서 safety degradation을 측정하기.

### 비교 대상 / baseline으로 쓸 수 있는 부분

- Textual trigger baseline.
- OCR textual trigger vs visual trigger vs combined trigger.
- CLIP embedding-space matching attack as a weak-access baseline.
- AdvBench prompt decomposition baseline.

### 내가 확장해볼 수 있는 부분

- encoder mismatch 조건에서 transferability 평가.
- image preprocessing, denoising, cropping, compression에 대한 robustness.
- safe visual grounding 또는 visual-context attribution을 이용한 defense.
- multimodal refusal training이 context contamination을 얼마나 줄이는지 평가.

### 후속 연구 질문

- "unsafe visual context"를 LLM이 답변 subject로 사용하는 순간을 mechanistic하게 탐지할 수 있는가?
- multimodal model에서 refusal은 text decoder에만 넣으면 충분한가, 아니면 encoder/projection 단계에도 safety boundary가 필요한가?
- adversarial image가 아니라 자연 이미지, meme, screenshot, document OCR에서도 유사한 compositional jailbreak가 발생하는가?

---

## Action Items

### 평가 관련

- LLaVA 계열 최신 모델, GPT-4V류 closed-source API, Qwen-VL / InternVL / LLaVA-NeXT 등으로 trigger taxonomy 재평가.
- single-turn ASR뿐 아니라 multi-turn context contamination ASR을 별도 metric으로 정의.

### 분석 관련

- vision embedding과 text embedding 사이 modality gap을 모델별로 시각화.
- adversarial image가 projection 이후 LLM token embedding space에서 어떤 unsafe concept cluster와 가까운지 분석.

### 방어 실험 관련

- input image denoising, JPEG compression, resizing, random crop, CLIP-embedding safety filter, output moderation을 layer별로 ablation.
- visual context를 답변 subject로 쓰기 전 safety classifier를 거치는 gating mechanism 실험.

### 재현 / 구현 관련

- GitHub repo의 `adv_image.py`를 기준으로 CLIP embedding matching pipeline을 재현.
- 동일 trigger set에서 seed, initialization image, threshold τ, optimizer setting 변화에 따른 ASR variance 측정.

### 후속 문헌 조사

- Multimodal jailbreak / visual prompt injection 후속 연구.
- Cross-modal safety alignment와 multimodal unlearning 연구.
- Vision encoder robustness와 embedding-space safety filter 관련 연구.

[1]: https://openreview.net/forum?id=plmBsXHxgR "Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models — OpenReview"
[2]: https://github.com/erfanshayegani/Jailbreak-In-Pieces "GitHub - erfanshayegani/Jailbreak-In-Pieces: [ICLR 2024 Spotlight  ] - [ Best Paper Award SoCal NLP 2023 ] - Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models · GitHub"
