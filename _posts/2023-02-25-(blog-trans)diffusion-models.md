---
layout: post
title: "(Blog-translation)What are Diffusion Models?"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.9. Blog Translation]
---

##### [lilianweng.github.io - What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#what-are-diffusion-models)

> So far, I&rsquo;ve written about three types of generative models, <a href="https://lilianweng.github.io/posts/2017-08-20-gan/">GAN</a>, <a href="https://lilianweng.github.io/posts/2018-08-12-vae/">VAE</a>, and <a href="https://lilianweng.github.io/posts/2018-10-13-flow-models/">Flow-based</a> models. They have shown great success in generating high-quality samples, but each has some limitations of its own. GAN models are known for potentially unstable training and less diversity in generation due to their adversarial training nature. VAE relies on a surrogate loss. Flow models have to use specialized architectures to construct reversible transform.
>> 지금까지, 나는 세가지 generative model(생성 모델)들인, GAN, VAE 그리고 Flow-based 모델들을 포스팅 했다. 그것들은 성공적 high-quality 결과들을 잘 생성함을 보여준다. 하지만 모델들 각각에는 한계가 존재한다. GAN은 adversarial 훈련의 성격상 결과를 생성할 때 근본적으로 불안전한 훈련과 적은 다양성으로 알려져있다. VAE는 surrogate loss<sup>[1]</sup>에 의존한다. Flow-based model은 가역적 변환(reversible transform)<sup>[2]</sup>을 구성하기위해 특별한 아키텍쳐를 사용해야한다.

1. VAE의 surrogate loss란?
    * VAE의 surrogate loss란, VAE에서 학습을 위해 사용되는 대체 손실 함수를 말합니다. VAE에서는 생성 과정에서 발생하는 손실 함수를 직접 최적화할 수 없기 때문에, 대신 최적화 가능한 다른 손실 함수를 사용합니다. 이러한 대체 손실 함수를 surrogate loss function이라고 합니다. [1]
2. Reversible transform
    * Flow-based 모델에서 Reversible transform은 역변환 가능한 가역함수<sup>[4]</sup>로, 입력과 출력을 모두 재생성할 수 있습니다. 이러한 역변환 가능한 특성을 이용하여 데이터 분포를 학습하고 생성하는데 활용됩니다. 이는 Generative model에서도 많이 활용되는데, 이를 통해 실제같은 이미지 생성이 가능해집니다. Reversible transform은 1x1 Convolution, NICE(Non-linear Independent Components Estimation), RealNVP(Real-valued Non-Volume Preserving) 등 다양한 방식으로 구현될 수 있습니다.[2][3] Flow-based 모델에서 Reversible transform은 NICE를 기반으로한 Reversible Architecture 등 다양한 방식으로 활용되며, 이를 이용하여 고해상도 이미지 생성 및 효율적인 샘플링, 데이터의 특성을 조작하는 등의 다양한 응용이 가능합니다.[4]
3. Invertible function
    * 역함수(inverse function)는 한 함수의 입력과 출력의 순서를 바꿔주는 함수입니다. 즉, 한 함수가 a를 b로 변환시키면, 역함수는 b를 a로 변환시킵니다. 이러한 역함수를 가지려면 두 가지 조건을 만족해야 한다고 합니다. 첫째, 일대일 대응(one-to-one)이어야 합니다. 이는 함수가 서로 다른 입력 값에 대해 서로 다른 출력 값을 내놓는 것을 의미합니다. 둘째, 전사(surjective)여야 합니다. 이는 모든 출력 값이 적어도 하나의 입력 값과 연관되어야 한다는 것을 의미합니다. [5] 예를 들어, y = x²와 같은 함수는 모든 출력 값에 대해 두 개의 입력 값이 연관되므로 역함수를 가질 수 없습니다. 따라서 역함수를 가지려면 함수가 일대일 대응이며 전사 함수여야 합니다. 함수 f(x)의 역함수는 f^-1(x)로 표기합니다. 역함수의 공식을 찾는 방법은 f(x) = y로 정의된 함수에서 x에 대해 풀면 됩니다. 이 때, 결과 식을 y = x로 두고, x와 y를 바꿔주면 역함수의 공식이 됩니다. [6]

> Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).
>> Diffusion models은 비평형 열역학(non-equilibrium thermodynamics)<sup>[4]</sup>에서 영감을 받았다. 그들은 데이터에 랜덤 노이즈를 천천히 추가하는 Markov chain of diffusion steps를 정의한 후에 노이즈로부터 원하는 데이터 샘플(input $x$)를 구성하기 위해 diffusion process를 역순으로 학습한다. VAE와 Flow-based 모델들과 달리, Diffusion모델은 fixed procedure(고정된 순서)대로 학습되고 latent vairable(잠재 변수)는 고차성(원본 데이터와 같은)을 가진다.

4. Non-equilibrium thermodynamics
    * 비평형 열역학(Non-equilibrium thermodynamics)은, 시간에 따라 변하는 비평형(non-equilibrium)<sup>[5]</sup> 상태에서 열과 운동 에너지, 질량 등의 흐름을 연구하는 학문 분야입니다. 열역학은 원래 열과 열역학적 평형 상태에 대한 연구를 중심으로 한 학문 분야이지만, 현실 세계에서는 시간에 따라 변하는 비평형 상태가 매우 많기 때문에, 비평형 열역학의 중요성이 높아지고 있습니다.
5. Non-equilibrium(비평형)
    * 시간에 따라 변하는 상태를 뜻합니다. 열역학에서 "평형 상태(equilibrium state)"란, 열, 운동, 화학 등의 에너지 교환이 더 이상 일어나지 않는 상태를 의미합니다.

![Fig 1](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png)

<figcaption>Fig. 1. Overview of different types of generative models.</figcaption>

# What are Diffusion Models?

> Several diffusion-based generative models have been proposed with similar ideas underneath, including <em>diffusion probabilistic models</em> (**DPM**; <a href="https://arxiv.org/abs/1503.03585">Sohl-Dickstein et al., 2015</a>), <em>noise-conditioned score network</em> (<strong>NCSN</strong>; <a href="https://arxiv.org/abs/1907.05600">Yang &amp; Ermon, 2019</a>), and <em>denoising diffusion probabilistic models</em> (<strong>DDPM</strong>; <a href="https://arxiv.org/abs/2006.11239">Ho et al. 2020</a>).</p>
>> 몇몇의 diffusion-based 생성 모델들은  <em>diffusion probabilistic models</em> (**DPM**; <a href="https://arxiv.org/abs/1503.03585">Sohl-Dickstein et al., 2015</a>), <em>noise-conditioned score network</em> (<strong>NCSN</strong>; <a href="https://arxiv.org/abs/1907.05600">Yang &amp; Ermon, 2019</a>), 그리고 <em>denoising diffusion probabilistic models</em> (<strong>DDPM</strong>; <a href="https://arxiv.org/abs/2006.11239">Ho et al. 2020</a>)들을 포함한 유사한 아이디어들을 바탕으로 제안되었다.

## Forward diffusion process

> Given a data point sampled from a real data distribution $\mathbf{x}_0 \sim q(\mathbf{x})$, let us define a <em>forward diffusion process</em> in which we add small amount of Gaussian noise to the sample in $T$ steps, producing a sequence of noisy samples $\mathbf{x}_1, \dots, \mathbf{x}_T$. The step sizes are controlled by a variance schedule $\{\beta_t \in (0, 1)\}_{t=1}^T$.
>> 실제 data의 분포 $\mathbf{x}_0 \sim q(\mathbf{x})$로부터 sample된 data point가 주어졌을 때, 우리는 $T$ step들 마다 sample에 매우 작은 양의 Gaussian noise를 추가하여, noisy(노이즈가 추가된) sample $\mathbf{x}_1, \dots, \mathbf{x}_T$ sequence<sup>[6]</sup>를 생산하는 *forward diffusion process*를 정의한다.

6. Sequence in deep learning?
    * 딥러닝에서 sequence는 연속적인 데이터 집합을 의미합니다. 이 데이터 집합은 시간적으로 연속되는 데이터 또는 공간적으로 연속되는 데이터일 수 있습니다. 예를 들어, 자연어 처리 분야에서는 문장이나 단어들의 나열을 시퀀스 데이터로 다룹니다. 이미지 처리 분야에서는 이미지의 픽셀 값들이 공간적으로 연속된 데이터로 다룹니다. 이와 같이 시퀀스 데이터는 딥러닝에서 다양한 분야에서 다루어지며, 주로 순환 신경망(RNN)이나 변환 모델(Transformer) 등을 이용하여 학습됩니다.

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$

> The data sample $\mathbf{x}_0$ gradually loses its distinguishable features as the step $t$ becomes larger. Eventually when $T \to \infty$, $\mathbf{x}_T$ is equivalent to an isotropic Gaussian distribution.
>> $t$ 단계가 커짐에 따라 데이터 샘플 $\mathbf{x}_0$는 점차 구별 가능한 feature들을 잃는다. 결국 $T \to \infty$일 때, $\mathbf{x}_T$는 isotropic Gaussian distribution(등방성 가우시안 분포)와 같다.

7. isotropic Gaussian distribution(등방성 가우시안 분포)
    * 이소토로픽(isotropic)은 '동일한 방향(iso)으로'를 의미하며, 가우시안 분포(Gaussian distribution)는 정규분포(normal distribution)의 다차원 버전인 다변량 정규분포(multivariate normal distribution)로 알려져 있습니다. 다변량 정규분포는 한 선형 결합(linear combination)으로 이루어진 k개의 구성요소(component)가 각각 단변량 정규분포(univariate normal distribution)를 따르면, k차원 벡터(random vector)가 k변량 정규분포를 따른다고 정의됩니다 [<a href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution" target="_new">7</a>]. 가우시안 함수(Gaussian function)는 이동과 회전에 대해 불변성(invariance)을 가지는 정규분포로, 확률밀도함수(probability density function)가 모든 방향에서 동일한 경우를 이소토로픽 가우시안 분포라고 합니다 [<a href="https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic" target="_new">1</a>][<a href="https://en.wikipedia.org/wiki/Gaussian_function" target="_new">3</a>]. 반면, 이소토로픽 가우시안 분포가 아닌 경우, 공분산행렬(covariance matrix)의 대각성분(diagonal elements)이 같지 않습니다 [<a href="https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic" target="_new">8</a>].

![Fig 2](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)

<figcaption>Fig. 2. The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source: <a href="https://arxiv.org/abs/2006.11239" target="_blank">Ho et al. 2020</a> with a few additional annotations)</figcaption>

> A nice property of the above process is that we can sample $\mathbf{x}_t$ at any arbitrary time step $t$ in a closed form using <a href="https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick">reparameterization trick.</a> Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$:
>> 위 프로세스에서 좋은 변수는 reparameterization trick을 사용한 closed form의 임의의 time step $t$에서 $\mathbf{x}_t$을 samle할 수 있다.

8. Reparameterization trick
    * reparameterization trick은 Variational Autoencoder(VAE)에서 많이 사용되는 트릭 중 하나입니다. VAE는 생성 모델 중 하나로, 데이터의 잠재 변수(latent variable)를 학습하여 새로운 데이터를 생성하는 모델입니다. 이때, 잠재 변수의 분포를 학습하는데 사용하는 파라미터는 평균과 분산으로 이루어진 벡터입니다. reparameterization trick은 이 분포에서 무작위 샘플링을 하기 위해 사용됩니다. 일반적으로 무작위 샘플링을 하면 미분이 불가능해지기 때문에, 이를 해결하기 위해 reparameterization trick을 사용합니다. 이를 통해, 미분 가능한 함수로 잠재 변수를 학습할 수 있으며, 학습이 안정적으로 이루어지고 생성된 샘플의 품질이 향상됩니다. [[<a href="https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic" target="_new">9</a>][<a href="https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic" target="_new">10</a>]]

$$
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$

##### Reference

[1] "An example of a surrogate loss function could be $psi (h (x)) = max (1 - h (x), 0)$ (the so-called hinge loss in SVM), which is convex and easy to optimize using conventional methods. This function acts as a proxy for the actual loss we wanted to minimize in the first place. Obviously, it has its disadvantages, but in some cases a surrogate ..."
URL: https://stats.stackexchange.com/questions/263712/what-is-a-surrogate-loss-function

[2] "We introduce Glow, a reversible generative model which uses invertible 1x1 convolutions. It extends previous work on reversible generative models and simplifies the architecture. Our model can generate realistic high resolution images, supports efficient sampling, and discovers features that can be used to manipulate attributes of data."
URL: https://openai.com/blog/glow/

[3] "These can also be described as flow-based models, reversible generative models, or as performing nonlinear independent component estimation. ... This transform is where Invertible 1x1 ..."
URL: https://medium.com/ai-ml-at-symantec/introduction-to-reversible-generative-models-4f47e566a73

[4] "Reversible Architectures are a family of neural network architectures that are based on the NICE [12,13] reversible transformation model which are the precursors of the mod-ern day generative flow based image generation architec-tures [29,36]. Based on the NICE invertible transforma-tions, Gomez et al. [22] propose a Reversible ResNet ar-"
URL: https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf

[5] "Formally speaking, there are two conditions that must be satisfied in order for a function to have an inverse. 1) A function must be injective (one-to-one). This means that for all values x and y in the domain of f, f (x) = f (y) only when x = y. So, distinct inputs will produce distinct outputs. 2) A function must be surjective (onto)."
URL: https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:composite/x9e81a4f98389efdf:invertible/v/determining-if-a-function-is-invertible

[6] "Learn how to find the formula of the inverse function of a given function. For example, find the inverse of f (x)=3x+2. Inverse functions, in the most general sense, are functions that reverse each other. For example, if f f takes a a to b b, then the inverse, f^ {-1} f −1, must take b b to a a."
URL: https://www.khanacademy.org/math/algebra-home/alg-functions/alg-finding-inverse-functions/a/finding-inverse-functions

[7] URL: "https://en.wikipedia.org/wiki/Multivariate_normal_distribution"

[8] URL: "https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic"

[9] "The reparameterization trick is a powerful engineering trick. We have seen how it works and why it is useful for the VAE. We also justified its use mathematically and developed a deeper understanding on top of our intuition. Autoencoders, more generally, is an important topic in machine learning."
URL: https://www.baeldung.com/cs/vae-reparameterization

[10] "VAE network with and without the reparameterization trick . where, 𝜙 representations the distribution the network is trying to learn. The epsilon remains as a random variable (sampled from a standard normal distribution) with a very low value thereby not causing the network to shift away too much from the true distribution."
URL: https://towardsdatascience.com/reparameterization-trick-126062cfd3c3