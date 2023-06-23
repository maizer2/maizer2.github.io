---
layout: post
title: "(VITON)Towards Scalable Unpaired Virtual Try-On via Patch-Routed Spatially-Adaptive GAN"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.7. Paper Review]
---

### [VITON Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/08/01/paper-of-VITON.html)

### $$\begin{align*}&\mathbf{Towards\;Scalable\;Unpaired\;Virtual\;Try-On\;via\;Patch-Routed\;Spatially-Adaptive\;GAN}\end{align*}$$

#### $$\begin{align*}&\mathbf{Zhenyu\;Xie,\;Zaiyu\;Huang,\;Fuwei\;Zhao,\;Haoye Dong  Michael\;Kampffmeyer,\;Xiaodan\;Liang}\\&\mathbf{Shenzhen\;Campus\;of\;Sun\;Yat-Sen\;University}\\&\mathbf{UiT\;The\;Arctic\;University\;of\;Norway,\;Peng\;Cheng\;Laboratory}\end{align*}$$

### $$\mathbf{Abstract}$$

Propose a texture preserving end-to-end network, PASTA-GAN, that facilitates real-world unpaired virtual try-on.

* Retaining garment texture and shape characteristics.
    * PASTA-GAN consists of **Patch-routed disentanglement module**


### $\mathbf{1.\;Introduction}$

* Problem of Paried training dataset
    * Unable to exchange garments directly between two person images, thus largelylimiting their application scenarios.

* Problem of UnParied training dataset
    * These models are usually trained by reconstructing the same person image, which is prone to over-fitting,
    * And thus underperform when handling garment transfer during testing.
    * The performance discrepancy is mainly reflected in the garment synthesis results, in particular the shape and texture, which we argue is caused by the entanglement of the garment style and spatial representations in the synthesis network **during the reconstruction process**.
    * None of the existing unpaired try-on methods consider the problem of coupled style and spatial garment information directly, which is crucial to obtain accurate garment transfer results in the unpaired and unsupervised virtual try-on scenario.

* What was PASTA-GAN trying to solve?
    * **Patch-routed disentanglement module** that decouples the garment style
    * **Spatially-adaptive residual module** to mitigate the problem of feature misalignment.
    * **By separating the garments into normalized patches** with the inherent spatial information largely reduced, the patch-routed disentanglement module encourages the style encoder to learn spatial-agnostic garment features. These features enable the synthesis network to generate images with accurate garment style regardless of varying spatial garment information.
    * **Given the target human pose**, the normalized patches can be easily reconstructed to the warped garment complying with the target shape, without requiring a warping network or a 3D human model.
    * **The spatially-adaptive residual module extracts the warped garment feature and adaptively inpaints the region** that is misaligned with the target garment shape. Thereafter, the inpainted warped garment features are embedded into the intermediate layer of the synthesis network, guiding the network to generate try-on results with realistic garment texture.

### $\mathbf{2.\;Related\;Work}$

* **Paired Virtual Try-on**
    * These methods require paired training data and are incapable of exchanging garments between two person images.
* **Unpaired Virtual Try-on**
    * Existing unpaired methods require either cumbersome data collecting or extensive online optimization, extremely harming their scalability in real scenarios.

### $\mathbf{3.\;PASTA-GAN}$

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-08-09-(VITON)PASTA-GAN/Figure-2.PNG)

**First of all, Simply explan PASTA-GAN structure and then explain it in detail.**

#### Section (a)

* Given Data
    * $I_{s}$ : A source image
    * $I_{t}$ : A target person image

Given Data $(I_{s}, I_{t})$ extract the source garment $G_{s}$ and the source pose $J_{s}$ and the target pose $J_{t}$.

* Extract Given Data
    * $I_{s}\to{}(G_{s},J_{s})$
    * $I_{t}\to{}J_{t}$

And then send to the **patch-routed disentanglement module** to **yield the normalized garment patches** $P_{n}$ and the **warped garment** $G_{t}$.

* Patch-routed disentanglement module
    * $PD(G_{s}, J_{s}, J_{t})\to{}(P_{n}, G_{t})$

#### Section (b)

The modified conditional StyleGAN2 first collaboratively exploits the disentangled style code $w$, projected from $P_{n}$

* $w$ from Style Encoder and Mapping Network
    * $\mathbf{MappingNetwork}(\mathbf{StyleEncoder}(P_{n}))\to{}w$

And the person identity feature $f_{id}$, encoded from target head and pose $(H_{t}, J_{t})$

* $f_{id}$ from Identity Encoder
    * $\mathbf{IdentityEncoder}(H_{t}, J_{t})\to{}f_{id}$

Coarse try-on result $\tilde{I}_{t}'$ and $M_{g}$ was Synthesized through the person identity feature $f_{id}$ and style code $w$.

* Through $4\times{}4$ Synthesis Block
    * $\mathbf{SynthesisBlock}(f_{id}, w)\to{}(\tilde{I}_{t}', M_{g})$

It then leverages the warped garment feature $f_{g}$ in the texture synthesis branch to generate the final try-on result $I_{t}'$.

1. $\mathbf{GarmentEncoder}(M_{g}\odot{}G_{t})\to{}f_{g}'$

2. $f_{g}'\odot{}(1-M_{misalign})+A(f_{g}'\odot{}M_{align})\odot{}M_{misalign}\to{}f_{g}$
    * It explain 3.2 section more detailly

$f_{g}$ in the texture synthesis branch to generate the final try-on result $I_{t}'$.

#### $mathbf{3.1\;Patch-routed\;Disentanglement\;Module}$

#### $mathbf{3.2\;Attribute-decoupled\;Conditional\;StyleGAN2}$

#### $mathbf{3.3\;Spatially-adaptive\;Residual\;Module}$

#### $mathbf{3.4\;Loss\;Functions\;and\;Training\;Details}$

### $mathbf{4\;Experiments}$

---

$$\begin{align}M_{align}=M_{g}\cap{}M_{t},\end{align}$$

$$\begin{align}M_{misalign}=M_{g}-M_{align},\end{align}$$

$$\begin{align}f_{g}'=E_{g}(G_{t}\odot{}M_{g}),\end{align}$$

$$\begin{align}f_{g}=f_{g}'\odot{}(1-M_{misalign})+A(f_{g}'\cdot{}M_{align})\odot{}M_{misalign}\;,\end{align}$$

