---
layout: post
title: "FEM Seminar 1 - FEM 기초"
categories: [2. Mathematics]
tags: [2.1. Pure mathematics, 2.1.1. Mathematical analysis, 2.1.2. Linear Algebra, 2.1.1.1. Calculus]
---

## 📚 공부 주제: 유한 요소법 기초

### 1. FEM(유한 요소법) 이란?

FEM, 유한요소법은 복잡한 공학 및 무리 문제를 컴퓨터를 이용해 수치적으로 해결하는 강력한 해석 기법이다. 눈에 보이지 않는 힘의 작용, 열의 이동, 유체의 흐름 등 어려운 미분방정식으로 표현되는 현상들을 눈으로 확인 가능한 형태로 시뮬레이션하여 보여준다.

> 여담으로, 저저가 생각했던 FEM은 미분/적분과 같이 큰 값을 구하기 위해 작은 것들의 집합을 하나로 합치면 결국 구하고자하는 값과 유사한 결과를 도출 할 수 있다고 생각했다. <br><br> 틀린말은 아니지만, "결국 FEM을 통해 하고자 하는게 뭐야?" 라는 질문에는 "음.. 어떤 실체를 표현하려는 것?"이라는 애매한 답변을 할 수 밖에 없었다. <br><br> 이런 생각을 하게 된 이유를 생각해보면, 아래에서 설명하게 될 **근사 해 방정식**을 정확히 이해하지 못해서 인 것 같다. <br><br> **근사 해 방정식**을 보면 차원에 해당하는 각 좌표에 대응하는 **형상 함수**가 있는데, 이를 정확히 이해하고 넘어가는게 중요하다고 생각된다.

FEM의 핵심 원리는 **분할과 정복**에 있다. 해석하려는 복잡한 형상의 대상물을 '유한(Finite)'개의 단순한 기하학적 모양(삼각형, 사각형 등)인 '요소(Element)'로 잘게 나눠, 각 요소의 거동(움직임)을 비교적 간단한 수학적 관계로 근사화하여 계산하고, 이를 전체적으로 **통합**하여 예측하는 방식이다.

1. 분할 (Divide)
    * 주어진 문제를 더 이상 나눌 수 없을 때까지 비슷한 유형의 여러 작은 문제로 쪼갠다.

2. 정복 (Conquer)
    * 나누어진 작은 문제들은 크기가 작고 단순하기 때문에 해결하기가 쉬워, 각 문제를 개별적으로 해결한다.

3. 통합 (Combine)
    * 해결된 작은 문제들의 답을 다시 원래의 큰 문제에 맞게 합쳐, 최종적인 해답을 얻는다.

---

### 2. 핵심 키워드 설명

#### 2.1. 근사 해 방정식 

$$ u_h = \sum_{i=1}^{DOF}\alpha_{i}\psi_{i} $$

**FEM에서 근사 해를 구하는 이유/목적은?**

* 일반적인 물리 문제는 보통 **미분방정식**으로 표현(모델링)된다.

* 하지만 이 문제는 너무 복잡하기 때문에 수학적으로 결과를 계산하기 어렵다.

* 따라서 FEM은 계산하기 어려운 **미분방정식** 대신, 해결하고자 하는 물리 문제를 보다 단순한 **대수 방정식**의 형태로 변환하여 컴퓨터를 이용해 수치적으로 해결할 수 있게 해준다.

* 즉, $u_h(x) = \alpha_1 \psi_1(x) + \alpha_2 \psi_2(x) + \cdots + \alpha_n \psi_n(x)$ 이처럼 $\psi_{i}(x)$라는 간단한 함수들을 여러 개 합쳐서, 복잡한 해 $u(x)$를 근사한다.

<br>

**기호 설명**

* $u_{h}$
    * FEM으로 계산된 **근사 해 (approximate solution)**. 
    * 실제 해 $u$를 완벽하게 알 수 없기 때문에, $u$ 대신 $u_{h}$로 계산.

          
* $\alpha_{i}$
    * **해의 계수 (또는 자유도 값, DOF 값)**. 
    * 이 값은 FEM 해석 과정에서 계산되는 수치 결과(변수). 예: 변위, 온도 등


* $\psi_{i}$
    * **기저 함수(basis function)** 또는 **형상 함수(shape function)**. 
    * 각 요소나 노드에서 해를 근사하기 위해 사용되는 함수.


* $DOF$(Degree of Freedom)
    * 자유도란, 구조물이나 물체가 ‘독립적으로 움직일 수 있는 방향의 수’를 말한다. 
    * 즉, 물체가 자유롭게 이동하거나 회전할 수 있는 독립적인 운동의 수 이다.<br>

    | 구조 차원 | 가능한 운동 방향 | 자유도 수 | 예시 |
    | -------- | --------------- | -------- | ---- |
    | **1D 구조** | X 방향 (좌우 이동) | 1개 | 스프링, 직선 막대 |
    | **2D 구조** | X 방향 이동<br>Y 방향 이동 | 2개 | 평면 트러스, 보 구조 |
    | **3D 구조** | X, Y, Z 방향 이동<br>X, Y, Z 축 회전 | 6개 | 입체 구조물, 빔-프레임 시스템 |

<br>

#### 2.2. 요소(Element)

![Various types of finite elements](https://raw.githubusercontent.com/maizer2/gitblog_img/refs/heads/main/2.%20Mathematics/2.1.%20Pure%20mathematics/2.1.1.%20Mathematical%20analysis/2.1.1.1.%20Calculus/FEM-Seminar-1/various_types_of_finite_elements.png)
<p align="center"><strong>Figure 1.</strong> 차원별 유한요소 종류</p>

**요소란?**

* FEM에서 복잡한 연속체(예: 구조물, 기계부품, 뼈, 날개 등)를 작은 조각들로 분할해서 분석할 때, 작은 조각 하나하나를 **요소(Element)** 라고 부른다.

<br>

**각 차원에서의 요소는?**

* 1차원 -> 선분(line segment or line element or linear element)
    * 두 노드로 구성된 1차원 요소


* 2차원 -> 삼각형(triangular element) / 사각형(quadrilateral element)
    * 세 노드로 구성된 평면 요소 / 네 노드로 구성된 평면 요소


* 3차원 -> 사면체(tetrahedral element) / 육면체(hexahedral element or brick element)
    * 네 꼭짓점의 3D 요소 / 여덟 꼭짓점과 여섯 면으로 구성된 벽돌 형태의 3D 요소

<br>

#### 2.3. 정규화/역정규화 방법

**아핀변환(Affine Transformation)**

* **선형 변환(linear transformation)**과 **이동(translation)**을 합친 좌표계 변환 방법으로, 이 두 변환을 조합하면 도형이나 좌표를 선형적으로 왜곡하고 동시에 위치도 바꿀 수 있는 굉장히 유연한 변환 방식이다.

* FEM에서는 정규화 방식으로써, "기준 요소"를 "실제 요소"로 바꾸거나, 그 반대로 이동시킬 때 아핀변환을 사용한다.

<br>

#### 2.4. 자코비안(Jacobian)

좌표계 간의 변환율을 나타내는 도함수 행렬이다. 

FEM에선 "참조 좌표 $\hat{x}$"에서 "실제 좌표 $x$"로 변환할 때, **한 점 근처의 변화를 얼마나 늘이거나 줄이는지를 나타내는 값**이다.

<br>

#### 2.5. 정규 좌표계(natural coordinate)

정규 좌표계는 유한 요소 내부의 위치를 표현하기 위해 사용하는 표준화된 참조 좌표계로써, 요소의 실제 위치나 모양과는 무관하게, 요소 내부의 모든 좌표를 고정된 범위 안에서 표현할 수 있게 만들어준다.

<br>

| 차원 | 요소 종류 | 정규 좌표계 | 좌표 기호 사용 용도 |
| ---- | -------- | ---------- | ------------------ |
| **1D** | 선분 요소  | $\hat{x} \in [0, 1]$ 또는 $\hat{\xi} \in [-1, 1] | - $\hat{x}$: FEM 기본 해석에서 직관적 표현<br>- $\hat{\xi}$: Gaussian 적분 등 수치 계산에 자주 사용  |
| **2D** | 삼각형 요소 | $(\hat{\xi}, \hat{\eta}) \in T\_R = { \hat{\xi} \ge 0, \hat{\eta} \ge 0, \hat{\xi} + \hat{\eta} \le 1 }$ | - $\hat{\xi}$: x 방향 상대 좌표<br>- $\hat{\eta}$: y 방향 상대 좌표 |
| | 사각형 요소 | $(\hat{\xi}, \hat{\eta}) \in [-1, 1] \times [-1, 1]$ | - $\hat{\xi}$: x 방향 상대 좌표<br>- $\hat{\eta}$: y 방향 상대 좌표 |
| **3D** | 사면체 요소 | $(\hat{\xi}, \hat{\eta}, \hat{\zeta})$ ∈ 정규화된 사면체 (예: $(0,0,0)$, $(1,0,0)$, $(0,1,0)$, $(0,0,1)$을 꼭짓점으로 사용) | - $\hat{\xi}$: x 방향 상대 좌표<br>- $\hat{\eta}$: y 방향 상대 좌표<br>- $\hat{\zeta}$: z 방향 상대 좌표 |

<br>

---

### 3. 1차원 유한 요소법, 1D Finite Element Method

**1차원 유한 요소법이란?**

* 유한 요소법(FEM)의 가장 기초적인 형태로, 주로 막대(bar), 빔(beam), 열전달 로드(rod) 등의 구조나 시스템을 해석하는 데 사용된다.

<br>

**수식 표현**

* 선분 요소(Line Element)
    * $u_{h}(x) = \sum_{i=1}^{2}\alpha_{i}\psi_{i}(x) = \alpha_{1}\psi_{1}(x) + \alpha_{2}\psi_{2}(x)$
    
<br>

**정규화/역정규화 (아핀변환, Affine Transformation)**

* **아핀변환(Affine Transformation)**을 적용한다.

* 정규 좌표계 $\hat{x} \in [0, 1]$ $\leftrightarrow$ 실제 좌표계 $x \in [a, b]$로 선형 스케일링 + 이동한 형태이다.

* 기호 설명
    * $a$: 실제 요소 구간의 **시작점** (좌측 끝)

    * $b$: 실제 요소 구간의 **끝점** (우측 끝)

    * $\hat{x}$: 정규 좌표계 상의 위치 (0과 1 사이의 값) 

    * $x$: 실제 좌표계에서의 위치

* 실제 요소 $\to$ 정규화된 요소 : 

$$x = \alpha + (b - a)\hat{x}$$

* 정규화된 요소 $\to$ 실제 요소 : 

$$\hat{x} = \frac{x - a}{b - a}$$

* 궁금한 점
    * **정규화란 a와 b를 0~1로 바꾸는 게 아닌가?**
        * FEM에서 말하는 정규화는 좌표계(=입력 값)를 바꾸는 것이지, 요소 자체(=구간 [a,b])를 바꾸는 게 아니다.
        * "정규화"는 a, b 자체를 바꾸는 게 아니라, 요소 내부의 '점' $x$들을 새로운 좌표 $\hat{x}$로 다시 표현하는 것이다.
        * 예시: 실제 요소 $[a=2, b=5]$ 정규화 $\to$ $[x=2 \to \hat{x} = 0, x=5 \to \hat{x} = 1]$
        
* 노드(Node) : 
    * $ x = 0 $ 과 $ x = 1 $

* 형상 함수(Shape function) : 
    * $\psi_{1}(x) = 1 - x$
    * $\psi_{2}(x) = x$

* 해의 근사 : 
    * $u_{h}(x) = \alpha_{1}(1-x) + \alpha_{2}(x)$

* 해석적 의미 :
    * $\alpha_{1}$ 과 $\alpha_{2}$ 는 각각 노드에서의 물리량(예: 변위)
    * 해는 두 점 사이를 직선으로 보간한 형태
        
<!-- * 시각화 코드

![Visualization for 1d linear element](https://raw.githubusercontent.com/maizer2/gitblog_img/refs/heads/main/2.%20Mathematics/2.1.%20Pure%20mathematics/2.1.1.%20Mathematical%20analysis/2.1.1.1.%20Calculus/FEM-Seminar-1/FEM_1d_visualization.png)
<p align="center"><strong>Figure 2.</strong> 1차 선형 요소 시각화.</p>

``` python
#Figure 2 코드

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

* 1차원 FEM에서의 Metrics $F(\hat{x})$ -->

---

### 4. 2차원 유한 요소법, 2D Finite Element Method

* 2차원 유한 요소법이란?
    * 2차원 FEM은, 1차원 FEM을 확장한 개념으로, 해석하려는 영역(예: 구조물의 단면, 판, 평면 영역 등)을 삼각형 또는 사각형 요소들로 분할하고, 각 요소 내에서 해(변위, 온도 등)를 보간하여 전체 해를 근사한다.

* 수식 표현
    * 선형 삼각형 요소(Linear Triangle Element)
        * $u_{h}(x, y) = \sum_{i=1}^{3}\alpha_{i}\psi_{i}(x, y) = \alpha_1 \psi_1(x, y) + \alpha_2 \psi_2(x, y) + \alpha_3 \psi_3(x, y)$
    
    * 선형 사각형 요소(Linear Quadrilateral Element)
        * $ u_{h}(x, y) = \sum_{i=1}^{4}\alpha_{i}\psi_{i}(x, y) = \alpha_1 \psi_1(x, y) + \alpha_2 \psi_2(x, y) + \alpha_3 \psi_3(x, y) + \alpha_4 \psi_4(x, y) $
        
* 아핀 변환(Affine Transformation)
    * 왜 2차원 유한 요소법에서 아핀 변환이 사용되는가?
        * 일반적으로 유한 요소법에서는 참조 요소(reference element)를 사용하여 정형화된 공간에서 계산이 이뤄진다. 이미 계산돼 있는 참조 요소를 **아핀 변환**을 적용하여 실제 요소(physical element)로 변환한다. 이를 통해 구하고자하는 영역을 쉽게 계산할 수 있다.
        
* 삼각형 요소, Triangle Element
    * 선형 삼각형 요소, Reference Triangle, $T_{R}$ : 
        * FEM에서 모든 삼각형 요소를 동일한 방식으로 해석하기 위해 사용하는 표준 삼각형. 일반적으로 세 꼭짓점을 갖는 단위 삼각형으로 정의됨

        * 이 정규화된 삼각형 위에서 형상 함수(shape function), 수치 적분(Gauss quadrature) 등을 정의하고, 모든 실제 삼각형 요소는 이 기준 삼각형으로부터의 **아핀 사상(affine mapping)**을 통해 변환됨.

$$ 
T_{R} = \{ (\hat{x}, \hat{y}) \mid{\hat{x} \ge{0},\ \hat{y} \ge{0},\ \hat{x} + \hat{y} \le{1}} \} 
$$

*
    * 실제 삼각형 요소, Physical Triangle, $T_{P}$ :
        *  실제 해석 대상의 **물리 공간(physical domain)**에 존재하는 **임의의 삼각형 요소**이다. 예를 들어, 2D 구조물의 메쉬를 구성하는 삼각형들이 여기에 해당된다.
        * 이 실제 삼각형은 세 꼭짓점의 좌표 $ (x_1, y_1), (x_2, y_2), (x_3, y_3) $로 정의되고,   계산을 위해 기준 삼각형 $ T_R $에서 물리 삼각형 $ T_P $로의 **아핀 변환**이 적용된다:

$$ (x, y) = x_1 + (x_2 - x_1)\hat{x} + (x_3 - x_1)\hat{y} $$

$$ (y, y) = y_1 + (y_2 - y_1)\hat{x} + (y_3 - y_1)\hat{y} $$

*   * 🔁  $T_{R}$ 와 $T_{P}$ 비교

        | 항목 | Reference Triangle $ T_R $ | Physical Triangle $ T_P $ |
        |------|------------------------------|-----------------------------|
        | 정의 | 정규화된 기준 삼각형 | 실제 해석 대상 삼각형 |
        | 좌표계 | $ (\hat{x}, \hat{y}) \in [0,1] $ | $ (x, y) \in \mathbb{R}^2 $ |
        | 목적 | 형상 함수 정의, 수치 적분 통일 | 실제 도메인의 물리 정보 표현 |
        | 변환 | 없음 | $ T_R \to T_P $ 로 어파인 사상 적용됨 |

---

### 5. 2차원 FEM의 삼각형 요소에 아핀 변환 적용

세 꼭짓점을 다음과 같이 정의한다:

| 노드 | 실제 좌표 $(x_i, y_i)$ | 참조 좌표 $(\hat{\xi}_i, \hat{\eta}_i)$ |
| -- | ------------------ | ----------------------------------- |
| 1  | $(x_1, y_1)$       | $(0, 0)$                            |
| 2  | $(x_2, y_2)$       | $(1, 0)$                            |
| 3  | $(x_3, y_3)$       | $(0, 1)$                            |

이때 아핀 행렬 $A$와 이동 벡터 $\mathbf{b}$는 다음과 같이 정의된다:

$$
A =
\begin{bmatrix}
x_2 - x_1 & x_3 - x_1 \\
y_2 - y_1 & y_3 - y_1
\end{bmatrix}, \quad
\mathbf{b} =
\begin{bmatrix}
x_1 \\
y_1
\end{bmatrix}
$$

따라서 전체 매핑은 다음과 같다:

$$
\begin{bmatrix}
x \\
y
\end{bmatrix}
=
\begin{bmatrix}
x_2 - x_1 & x_3 - x_1 \\
y_2 - y_1 & y_3 - y_1
\end{bmatrix}
\cdot
\begin{bmatrix}
\hat{\xi} \\
\hat{\eta}
\end{bmatrix}
+
\begin{bmatrix}
x_1 \\
y_1
\end{bmatrix}
$$

---

### 6. 역변환 (실제 → 참조)

아핀 변환은 선형이므로 행렬 $A$가 가역적이면 역변환도 가능하다:

$$
\begin{bmatrix}
\hat{\xi} \\
\hat{\eta}
\end{bmatrix}
=
A^{-1}
\cdot
\left(
\begin{bmatrix}
x \\
y
\end{bmatrix}
-
\begin{bmatrix}
x_1 \\
y_1
\end{bmatrix}
\right)
$$

이 역변환은 **적분점 또는 물리 좌표를 참조 좌표로 매핑할 때** 매우 중요하게 사용된다.

---

### 7. 3차원 유한 요소법, 3D Finite Element Method

* **3차원 유한 요소법(3D FEM)**이란?

  * 3D FEM은 해석하려는 물리적 구조(예: 고체, 덩어리 구조물, 부피를 가진 시스템)를 **3차원 요소들(예: 사면체, 육면체 등)**로 분할하고, 각 요소 내에서 물리량(변위, 응력 등)을 보간하여 전체 해를 근사하는 방법이다.

* **기본 요소 형태**

  * 사면체(Tetrahedron), 육면체(Hexahedron), 프리즘(Prism) 등이 사용되며, 이 중 **사면체(Tetrahedral element)**가 가장 기본적인 3차원 요소로 많이 쓰임.

* **기저 함수 표현 예**

  * 선형 사면체 요소에서는 다음과 같이 4개의 shape function을 사용하여 해를 근사한다:

  $$
  u_h(x, y, z) = \sum_{i=1}^{4} \alpha_i \, \psi_i(x, y, z)
  $$

---

### 8. 3차원 FEM의 사면체 요소에 아핀 변환 적용

#### 기준 사면체(Reference Tetrahedron), $T_R$

* 보통 다음의 정규화된 점들을 꼭짓점으로 사용:

  * $(0, 0, 0)$
  * $(1, 0, 0)$
  * $(0, 1, 0)$
  * $(0, 0, 1)$

#### 실제 사면체 요소(Physical Tetrahedron), $T_P$

* 실제 해석 대상에 존재하는 사면체 요소는 다음과 같은 꼭짓점 좌표를 가짐:

  * $(x_1, y_1, z_1)$
  * $(x_2, y_2, z_2)$
  * $(x_3, y_3, z_3)$
  * $(x_4, y_4, z_4)$

#### 아핀 행렬 $A$ 및 이동 벡터 $\mathbf{b}$

기준점 $\hat{\mathbf{x}} = (\hat{\xi}, \hat{\eta}, \hat{\zeta})$를 실제 공간 $\mathbf{x} = (x, y, z)$로 매핑하는 식:

$$
\mathbf{x} = A \cdot \hat{\mathbf{x}} + \mathbf{b}
$$

여기서:

* $A \in \mathbb{R}^{3 \times 3}$: 3D 선형 변환 행렬
* $\mathbf{b} \in \mathbb{R}^{3}$: 이동 벡터 (기준점 이동)

$$
A =
\begin{bmatrix}
x_2 - x_1 & x_3 - x_1 & x_4 - x_1 \\
y_2 - y_1 & y_3 - y_1 & y_4 - y_1 \\
z_2 - z_1 & z_3 - z_1 & z_4 - z_1
\end{bmatrix}, \quad
\mathbf{b} =
\begin{bmatrix}
x_1 \\
y_1 \\
z_1
\end{bmatrix}
$$

따라서 전체 매핑은:

$$
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
=
A
\cdot
\begin{bmatrix}
\hat{\xi} \\
\hat{\eta} \\
\hat{\zeta}
\end{bmatrix}
+
\begin{bmatrix}
x_1 \\
y_1 \\
z_1
\end{bmatrix}
$$

---

### 9. 역변환 (실제 → 참조)

아핀 변환 $\mathbf{x} = A \hat{\mathbf{x}} + \mathbf{b}$는 선형이므로, 행렬 $A$가 가역적일 경우 다음과 같이 역변환이 가능하다:

$$
\hat{\mathbf{x}} =
A^{-1}
\cdot
\left(
\mathbf{x} - \mathbf{b}
\right)
$$

즉,

$$
\begin{bmatrix}
\hat{\xi} \\
\hat{\eta} \\
\hat{\zeta}
\end{bmatrix}
=
A^{-1}
\cdot
\left(
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
-
\begin{bmatrix}
x_1 \\
y_1 \\
z_1
\end{bmatrix}
\right)
$$

이 역변환은 다음과 같은 상황에서 사용된다:

* **적분점이 실제 요소 상에 주어졌을 때**, 기준 사면체 좌표로 변환해 shape function 등을 평가할 때
* **해석 해를 시각화하거나 후처리할 때**, 기준 공간에서의 정보를 물리 공간으로 변환하기 위해

---

### 10. 과제 및 추가 연구

물론입니다! 지금까지의 내용을 바탕으로 한 **10. 과제 및 추가 연구** 항목을 아래에 구성해드렸습니다. 이 항목은 개념 정리, 실제 계산 적용, 프로그래밍 구현, 심화 학습으로 이어지는 단계적 학습 목표와 탐구 과제를 포함합니다.

---

### 10. 과제 및 추가 연구 -> ChatGPT 요청 결과

#### 기본 개념 복습

1. **기초 용어 정리**

   * 유한 요소법(FEM), 요소(Element), 자유도(DOF), 기저 함수(Shape Function), 자코비안(Jacobian), 아핀 변환(Affine Transformation)의 정의와 역할을 스스로 설명해보기
   * 1D, 2D, 3D 요소의 차이점 요약 정리하기

2. **수식 유도 및 손 계산**

   * 1차원 선형 요소에서 해 $u_h(x)$의 수식 유도 및 손계산으로 근사값 구하기
   * 선형 삼각형 요소에서의 아핀 변환 행렬 $A$와 역변환 수식 유도해 보기

---

#### 계산 및 수치 예제

3. **1D FEM 수치 구현 과제**

   * 구간 [0,1]을 2개 요소로 나눈 FEM 모델에서 각 요소의 강성 행렬을 계산하고 전체 시스템 방정식 조립해보기
   * Dirichlet 경계 조건을 주고 자유도 해 $\alpha_i$를 구하는 시스템 구현

4. **2D 삼각형 요소의 자코비안 계산**

   * 세 꼭짓점이 주어진 실제 삼각형에서 아핀 행렬 $A$ 구성
   * 자코비안 $J = \det(A)$, $J^{-1}$을 계산하고 도함수 변환에 활용

5. **3D 사면체 요소의 아핀 변환 적용**

   * 주어진 사면체의 좌표 데이터를 가지고 아핀 변환 $A$ 및 $A^{-1}$을 계산
   * 기준 좌표 $(\hat{\xi}, \hat{\eta}, \hat{\zeta})$에서 실제 좌표 $(x, y, z)$, 반대 방향 변환도 수행

---

#### 프로그래밍

6. **Python으로 FEM 구현**

   * 1D 또는 2D FEM 해석 프로그램 구성 (예: 열전달, 변위 해석)
   * 요소 조립, 형상 함수, 자코비안, 적분, 경계조건 적용 등 전체 흐름 포함
   * [선택] `matplotlib`를 이용한 형상 함수 시각화

7. **기저 함수 시각화**

   * 1D: $\psi_1(x) = 1 - x, \ \psi_2(x) = x$
   * 2D: 선형 삼각형 요소의 $\psi_1(\hat{x}, \hat{y}) = 1 - \hat{x} - \hat{y}$, 등고선 또는 색상 맵으로 시각화

---

#### 📚 심화 학습 주제

8. **고차 요소 탐색**

   * 2차 또는 3차 기저 함수의 정의 및 활용 방법 정리
   * 6-node 삼각형 요소, 10-node 사면체 요소 등 고차 보간 요소 구조 학습

9. **수치 적분(Gauss quadrature)**

   * 1D, 2D, 3D에서 사용되는 Gaussian Quadrature 규칙 학습
   * 정규 요소 기준으로 수치 적분 공식 직접 구현

10. **응용 분야 조사**

    * 구조 해석, 전자기 해석, 유체 흐름 등 FEM이 사용되는 실제 산업 사례 조사
    * 상업용 FEM 소프트웨어(예: ANSYS, Abaqus, COMSOL 등) 기능 비교

---

### 목표 정리

| 단계      | 목표                              |
| ------- | ------------------------------- |
| ① 개념 이해 | FEM, 아핀변환, 자코비안 등 핵심 용어 및 수식 숙지 |
| ② 계산 능력 | 요소 단위의 수식 및 자코비안 계산 역량          |
| ③ 프로그래밍 | Python/MATLAB 기반 FEM 코드 구현      |
| ④ 확장 탐구 | 고차 요소, 수치 적분, 산업 응용 이해          |
