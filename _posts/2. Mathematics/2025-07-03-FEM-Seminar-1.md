---
layout: post
title: "FEM Seminar 1 - FEM 기초"
categories: [1. Computer Engineering, 2. Mathematics]
tags: [2.1. Pure mathematics, 2.1.1. Mathematical analysis, 2.1.2. Linear Algebra, 2.2. Applied Mathematics, 2.2.2. Mathematical Statistics, 2.2.1. Mathematical Optimization, 2.0. Paper Review]
---

## 📚 공부 주제: Reference Triangle과 Affine Mapping을 활용한 FEM의 기하학적 모델링 구조

### 1. FEM(유한 요소법) 이란?
FEM, 유한요소법은 복잡한 공학 및 무리 문제를 컴퓨터를 이용해 수치적으로 해결하는 강력한 해석 기법이다. 눈에 보이지 않는 힘의 작용, 열의 이동, 유체의 흐름 등 어려운 미분방정식으로 표현되는 현상들을 눈으로 확인 가능한 형태로 시뮬레이션하여 보여준다.
  
핵심 원리는 **분할과 정복**에 있다. 해석하려는 복잡한 형상의 대상물을 '유한(Finite)'개의 단순한 기하학적 모양(삼각형, 사각형 등)인 '요소(Element)'로 잘게 나눠, 각 요소의 거동(움직임)을 비교적 간단한 수학적 관계로 근사화하여 계산하고, 이를 전체적으로 **통합**하여 예측하는 방식이다.

1. 분할 (Divide)
    * 주어진 문제를 더 이상 나눌 수 없을 때까지 비슷한 유형의 여러 작은 문제로 쪼갠다.

2. 정복 (Conquer)
    * 나누어진 작은 문제들은 크기가 작고 단순하기 때문에 해결하기가 쉬워, 각 문제를 개별적으로 해결한다.

3. 통합 (Combine)
    * 해결된 작은 문제들의 답을 다시 원래의 큰 문제에 맞게 합쳐, 최종적인 해답을 얻는다.

### 2. 핵심 키워드
* 근사 해 : $u_h = \sum^{DOF}_{i=1}\alpha_{i}\psi_{i}$
    * $u_{h}$: FEM으로 계산된 **근사 해 (approximate solution)**. 실제 해 $u$를 완벽하게 알 수 없기 때문에, $u$ 대신 $u_{h}$로 계산.

    * $\alpha_{i}$ : **해의 계수 (또는 자유도 값, DOF 값)**. 이 값은 FEM 해석 과정에서 계산되는 수치 결과(변수). 예: 변위, 온도 등

    * $\psi_{i}$ : **기저 함수(basis function)** 또는 **형상 함수(shape function)**. 각 요소나 노드에서 해를 근사하기 위해 사용되는 함수.

    * $DOF$ : 전체 시스템에서의 자유도(Degree of Freedom) 개수. FEM 시스템의 변수 개수로 사용.

* Reference Triangle, $T_{R}$ : 

* Physical Triangle, $T_{P}$ :

### 3. 이해 어려웠던 부분
- Precision과 Recall의 trade-off
- ROC Curve와 AUC 개념

### 4. 적용 예시
- 불균형 데이터셋에서는 Precision보다 Recall이 중요할 수 있음 (예: 스팸 탐지)

### 5. 느낀 점 및 다음 계획
- 모델을 비교할 때는 하나의 지표로 판단하지 말 것
- 다음에는 ROC Curve 직접 그려보기
