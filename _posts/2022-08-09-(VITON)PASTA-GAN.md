---
layout: post
title: "(VITON)Towards Scalable Unpaired Virtual Try-On via Patch-Routed Spatially-Adaptive GAN"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.7. Literature Review]
---

### [VITON Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/08/01/Literature-of-VITON.html)

### [$$\mathbf{Towards\;Scalable\;Unpaired\;Virtual\;Try-On\;via\;Patch-Routed\;Spatially-Adaptive\;GAN}$$](https://arxiv.org/pdf/2111.10544v1.pdf)

##### $$\begin{align*}&\mathbf{Zhenyu\;Xie,\;Zaiyu\;Huang,\;Fuwei\;Zhao,\;Haoye Dong  Michael\;Kampffmeyer,\;Xiaodan\;Liang}\\&\mathbf{Shenzhen\;Campus\;of\;Sun\;Yat-Sen\;University}\\&\mathbf{UiT\;The\;Arctic\;University\;of\;Norway,\;Peng\;Cheng\;Laboratory}\end{align*}$$


### $$\mathbf{Abstract}$$

Propose a texture preserving end-to-end network, PASTA-GAN, that facilitates real-world unpaired virtual try-on.

* Retaining garment texture and shape characteristics.
    * PASTA-GAN consists of **Patch-routed disentanglement module**


### $\mathbf{1\;Introduction}$

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

![Figure 2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/img/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-08-09-(VITON)PASTA-GAN/Figure-2.PNG)

* Given Data
    * $I_{s}$ : A source image
    * $I_{t}$ : A target person image

$$\begin{align}\end{align}$$