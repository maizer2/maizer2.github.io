---
layout: post
title: "Mathematical theory for Diffusion model 01"
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 1.2.2.7. Diffusion, 2.2. Pobability and Statistics]
---

벡터나 행렬의 크기를 나타내는 노름(norm)은 벡터나 행렬의 크기나 길이를 측정하는 함수로, 다양한 종류가 있습니다. 벡터의 노름을 구하는 방법에는 다음과 같은 것들이 있습니다.

$L_1$ 노름: 벡터의 각 요소의 절댓값을 더한 값입니다. 수식으로는 $\Vert x \Vert_{1} = \sum_i \vert x_i \vert$ 입니다.

$L_2$ 노름: 벡터의 각 요소의 제곱을 더한 값에 루트를 씌운 값입니다. 수식으로는 $\Vert x \Vert_{2} = \sqrt{\sum_i x_i^2}$ 입니다. 이 노름은 Euclidean distance로서 자주 사용됩니다.

$L_p$ 노름: 벡터의 각 요소의 p제곱을 더한 값에 1/p승을 취한 값입니다. 수식으로는 $\Vert x \Vert_{p} = (\sum_i \vert x_i \vert^p)^{1/p}$ 입니다.

행렬의 노름을 구하는 방법으로는 Frobenius 노름과 spectral 노름이 있습니다.

Frobenius 노름: 행렬의 각 요소의 제곱을 더한 값에 루트를 씌운 값입니다. 수식으로는 $\Vert A \Vert_{F} = \sqrt{\sum_{i,j} A_{i,j}^2}$ 입니다.

Spectral 노름: 행렬의 특잇값 중에서 가장 큰 값입니다. 수식으로는 $\Vert A \Vert_{2} = \sigma_{\max}(A)$ 입니다. 이 노름은 행렬의 연산의 안정성과 관련이 있습니다.

노름은 벡터나 행렬의 크기나 길이를 측정하는 방법으로 다양한 분야에서 사용됩니다. 예를 들어, 머신 러닝에서는 노름을 사용하여 가중치 파라미터의 크기를 조절하여 모델의 일반화 성능을 향상시키는 정규화(regularization)을 수행하는 등의 다양한 용도로 활용됩니다.
