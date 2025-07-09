---
layout: post
title: "FEM Seminar 1 - FEM 기초"
categories: [2. Mathematics]
tags: [2.1. Pure mathematics, 2.1.1. Mathematical analysis, 2.1.2. Linear Algebra, 2.1.1.1. Calculus]
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

### 2. 핵심 키워드 설명
* 근사 해 : $u_h = \sum_{i=1}^{DOF}\alpha_{i}\psi_{i}$
    * $u_{h}$: FEM으로 계산된 **근사 해 (approximate solution)**. 실제 해 $u$를 완벽하게 알 수 없기 때문에, $u$ 대신 $u_{h}$로 계산.

    * $\alpha_{i}$ : **해의 계수 (또는 자유도 값, DOF 값)**. 이 값은 FEM 해석 과정에서 계산되는 수치 결과(변수). 예: 변위, 온도 등

    * $\psi_{i}$ : **기저 함수(basis function)** 또는 **형상 함수(shape function)**. 각 요소나 노드에서 해를 근사하기 위해 사용되는 함수.

    * $DOF$ : 전체 시스템에서의 자유도(Degree of Freedom) 개수, FEM 시스템의 변수 개수로 사용.

    * 식의 의미 : FEM은 해 $u(x)$를 정확히 구할 수 없기 때문에, 기저 함수들의 선형 결합으로 근사한다. 즉, $u_h(x) = \alpha_1 \psi_1(x) + \alpha_2 \psi_2(x) + \cdots + \alpha_n \psi_n(x)$ 이처럼 $\psi_{i}(x)$라는 간단한 함수들을 여러 개 합쳐서, 복잡한 해 $u(x)$를 근사한다.

![Visualization for 1d linear element](https://raw.githubusercontent.com/maizer2/gitblog_img/refs/heads/main/2.%20Mathematics/2.1.%20Pure%20mathematics/2.1.1.%20Mathematical%20analysis/2.1.1.1.%20Calculus/FEM-Seminar-1/FEM_1d_visualization.png)
<p align="center"><strong>Figure 1.</strong> 차원별 유한요소 타입.</p>

* 요소(Element)란?
    * FEM에서 복잡한 연속체(예: 구조물, 기계부품, 뼈, 날개 등)를 작은 조각들로 분할해서 분석할 때, 작은 조각 하나하나를 **요소(Element)** 라고 부른다.
    * 각 차원에서의 요소는?
        * 1차원 -> 선분(line segment or line element or linear element)
            * 두 노드로 구성된 1차원 요소
        * 2차원 -> 삼각형(triangular element) / 사각형(quadrilateral element)
            * 세 노드로 구성된 평면 요소 / 네 노드로 구성된 평면 요소
        * 3차원 -> 사면체(tetrahedral element) / 육면체(hexahedral element or brick element)
            * 네 꼭짓점의 3D 요소 / 
    * 특징
        * **기저 함수(basis function)** 또는 **형상 함수(shape function)** 가 1차 다항식
        * 각 요소에는 **노드(Node)** 가 2개만 있음
        * 해는 두 노드 사이를 **직선으로 연결**해서 보간

* Reference Triangle, $T_{R}$ : 
    * FEM에서 모든 삼각형 요소를 동일한 방식으로 해석하기 위해 사용하는 표준 삼각형. 일반적으로 다음 세 꼭짓점을 갖는 단위 삼각형으로 정의됨:

$$ T_R = \left\{ (\hat{x}, \hat{y}) \mid \hat{x} \ge 0,\ \hat{y} \ge 0,\ \hat{x} + \hat{y} \le 1 \right\} $$

    * 이 정규화된 삼각형 위에서 형상 함수(shape function), 수치 적분(Gauss quadrature) 등을 정의하고, 모든 실제 삼각형 요소는 이 기준 삼각형으로부터의 **어파인 사상(affine mapping)**을 통해 변환됨.

* Physical Triangle, $T_{P}$ :
    *  실제 해석 대상의 **물리 공간(physical domain)**에 존재하는 **임의의 삼각형 요소**이다.   예를 들어, 2D 구조물의 메쉬를 구성하는 삼각형들이 여기에 해당된다.
    * 이 물리 삼각형은 세 꼭짓점의 좌표 $ (x_1, y_1), (x_2, y_2), (x_3, y_3) $로 정의되고,   계산을 위해 기준 삼각형 $ T_R $에서 물리 삼각형 $ T_P $로의 **어파인 변환**이 적용된다:

$$ (x, y) = x_1 + (x_2 - x_1)\hat{x} + (x_3 - x_1)\hat{y} $$

$$ (y, y) = y_1 + (y_2 - y_1)\hat{x} + (y_3 - y_1)\hat{y} $$

### 🔁  $T_{R}$ 와 $T_{P}$ 비교

| 항목 | Reference Triangle $ T_R $ | Physical Triangle $ T_P $ |
|------|------------------------------|-----------------------------|
| 정의 | 정규화된 기준 삼각형 | 실제 해석 대상 삼각형 |
| 좌표계 | $ (\hat{x}, \hat{y}) \in [0,1] $ | $ (x, y) \in \mathbb{R}^2 $ |
| 목적 | 형상 함수 정의, 수치 적분 통일 | 실제 도메인의 물리 정보 표현 |
| 변환 | 없음 | $ T_R \to T_P $ 로 어파인 사상 적용됨 |

### 3. 1차 선형 요소의 해

* 1차 선형 요소란?
    * 수식 표현
        * $u_{h}(x) = \sum_{i=1}^{2}\alpha_{i}\psi_{i} = \alpha_{1}\psi_{1}(x) + \alpha_{2}\psi_{2}(x)$

        * 요소 구간 : $x\in[0,1]$
            * 요소 구간이 $[0, 1]$인 이유
                * 정규 좌표계(natural coordinate)를 사용하기 위해
                * 실제 요소 구간이 [2, 5]일지라도, 정규 좌표계를 사용함으로써 모든 요소를 **동일하게 해석**할 수 있고, 한 번 정의한 형상 함수 및 적분 등의 계산을 모든 요소에 **재사용** 가능하다.
                * 실제 요소 구간(실제 좌표계, physical coordinates)을 정규화된 요소 구간 $\hat{x}\in[0,1]$으로 변환하여 단순하게 해석한다.
                    * 실제 요소 -> 정규화된 요소 : $x = \alpha + (b - a)\hat{x}$
                    * 정규화된 요소 -> 실제 요소 : $\hat{x} = (x - a) / (b - a)$

        * 노드(Node) : $ x = 0 $ 과 $ x = 1 $

        * 형상 함수(Shape function) : 
            * $\psi_{1}(x) = 1 - x$
            * $\psi_{2}(x) = x$

        * 해의 근사: $u_{h}(x) = \alpha_{1}(1-x) + \alpha_{2}(x)$
        * 해석적 의미 :
            * $\alpha_{1}$ 과 $\alpha_{2}$ 는 각각 노드에서의 물리량(예: 변위)
            * 해는 두 점 사이를 직선으로 보간한 형태
        
* 시각화 코드

![Visualization for 1d linear element](https://raw.githubusercontent.com/maizer2/gitblog_img/refs/heads/main/2.%20Mathematics/2.1.%20Pure%20mathematics/2.1.1.%20Mathematical%20analysis/2.1.1.1.%20Calculus/FEM-Seminar-1/various_types_of_finite_elements.png)
<p align="center"><strong>Figure 2.</strong> 1차 선형 요소 시각화.</p>

``` python
#Figure 1 코드

import numpy as np
import matplotlib.pyplot as plt

# 노드 위치
x = np.linspace(0, 1, 100)

# 계수 (자유도 값)
alpha1 = 2
alpha2 = 5

# 형상 함수
psi1 = 1 - x
psi2 = x

# 근사 해
u_h = alpha1 * psi1 + alpha2 * psi2

# 시각화
plt.plot(x, psi1, '--', label='ψ₁(x) = 1 - x')
plt.plot(x, psi2, '--', label='ψ₂(x) = x')
plt.plot(x, u_h, label='uₕ(x) = α₁ψ₁ + α₂ψ₂', linewidth=2)
plt.scatter([0, 1], [alpha1, alpha2], c='red', zorder=5, label='노드 값 (α₁, α₂)')

# plt.title('1차 선형 요소의 FEM 근사 해 시각화')
plt.title('Visualization of FEM Approximation with Linear Elements')
plt.xlabel('x')
plt.ylabel('uₕ(x)')
plt.legend()
plt.grid(True)
plt.show()
```
### 4. 2차 선형 요소의 해

2차 요소는 각 변 중간에 **추가 노드(midpoint node)**를 배치하여, 해를 **2차 다항식으로 보간**하는 방식이다. 이는 해의 정확도를 높이기 위한 방법이다.

#### 📌 요소 구성:
- 노드 개수: 총 **6개**
  - 세 꼭짓점 (vertex nodes)
  - 세 변의 중점 (mid-side nodes)

#### 📐 형상 함수 예시 (1D 기준)
요소 구간이 $ x \in [0, 1] $일 때, 노드는 $ x = 0, 0.5, 1 $에 위치한다.

형상 함수:

$$ 
\begin{align*}
\psi_1(x) &= 2(x - 0.5)(x - 1) \\
\psi_2(x) &= 4x(1 - x) \\
\psi_3(x) &= 2x(x - 0.5)
\end{align*}
$$

- $\psi_1(x)$: 왼쪽 끝 노드 (x=0)에서 1
- $\psi_2(x)$: 중간 노드 (x=0.5)에서 1
- $\psi_3(x)$: 오른쪽 끝 노드 (x=1)에서 1

#### 📈 근사 해 표현:

$$ u_h(x) = \alpha_1 \psi_1(x) + \alpha_2 \psi_2(x) + \alpha_3 \psi_3(x) $$

#### 🔍 해석적 의미:

- 1차 요소는 직선 보간만 가능하지만,
- 2차 요소는 **곡선 형태(포물선)**로 해를 더 정밀하게 근사 가능
- 경계 조건이 더 세밀하게 반영되며, 수렴 속도도 향상됨

> ✅ 2차 요소는 더 많은 계산량이 필요하지만, 더 높은 정확도를 제공

#### 🖼️ 예시 시각화 (1D 기준)

- $ x = 0, 0.5, 1 $ 세 점을 기준으로 한 포물선 곡선
- 중간 노드가 반영되기 때문에 곡선이 휘는 형태로 보간됨

---


### 5. 과제 및 추가 연구
- 모델을 비교할 때는 하나의 지표로 판단하지 말 것
- 다음에는 ROC Curve 직접 그려보기
