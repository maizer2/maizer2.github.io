---
layout: post
title: "(VITON)Person Image Synthesis via Denoising Diffusion Model HumanDiffusion"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.7. Literature Review]
---

### [VITON Literature List](https://maizer2.github.io/1.%20computer%20engineering/2022/08/01/Literature-of-VITON.html)

# Person Image Synthesis via Denoising Diffusion Model

## Abstract

The pose-guided person image generation task requires synthesizing photorealistic images of humans in arbitrary poses. 

The existing approaches use generative adversarial networks that do not necessarily maintain realistic textures or need dense correspondences that struggle to handle complex deformations and severe occlusions. 

In this work, we show how denoising diffusion models can be applied for high-fidelity person image synthesis with strong sample diversity and enhanced mode coverage of the learnt data distribution. 

Our proposed Person Image Diffusion Model (PIDM) disintegrates the complex transfer problem into a series of simpler forward-backward denoising steps.

This helps in learning plausible source-to-target transformation trajectories that result in faithful textures and undistorted appearance details. 

We introduce a ‘texture diffusion module’ based on cross-attention to accurately model the correspondences between appearance and pose information available in source and target  images. 

Further, we propose ‘disentangled classifier-free guidance’ to ensure close resemblance between the conditional inputs and the synthesized output in terms of both pose and appearance  information. 

Our extensive results on two large-scale benchmarks and a user study demonstrate the photorealism of our proposed approach under challenging scenarios. We also show how our generated images can help in downstream tasks.

Our code and models will be publicly released.

## 1. Introduction

The Pose-guided person image synthesis task [19,23,30] aims to render a person’s image with a desired pose and appearance. 

Specifically, the appearance is defined by a given source image and the pose by a set of keypoints. 

Having control over the synthesized person images in terms of pose and style is an important requisite for applications such as ecommerce, virtual reality, metaverse and content generation for the entertainment industry. 

Furthermore, the generated images can be used to improve performance on downstream tasks such as person re-identification [30]. 

The challenge is to generate photorealistic outputs tightly conforming with the given pose and appearance information. 

In the literature, person synthesis problem is generally tackled using Generative Adversarial Networks (GAN) [4] which try to generate a person in a desired pose using a single forward pass. 

However, preserving a coherent structural, appearance and global body composition in the new pose is a challenging task to achieve in one shot. 

The resulting outputs commonly experience deformed textures and unrealistic body shapes, especially when synthesizing occluded body parts (see Fig. 1). 

Further, GANs are prone to unstable training behaviour due to adversarial min-max objective and lead to limited diversity in the generated samples. Similarly, Variational Autoencoder [8] based solutions have been explored that are relatively stable, but suffer from blurry details and offer low-quality outputs than GANs due to their dependence on a surrogate loss for optimization. 

In this work, we frame the person synthesis problem as a series of diffusion steps that progressively transfer a person in the source image to the target pose. Diffusion models [6] are motivated from non-equilibrium thermodynamics that define a Markov chain of slowly adding noise to the input samples (forward pass) and then reconstructing the desired samples from noise (reverse pass). 

In this manner, rather than modeling the complex transfer characteristics in a single go, our proposed person synthesis approach PIDM breaks down the problem into a series of forward-backward diffusion steps to learn the plausible transfer trajectories. 

Our approach can model the intricate interplay of the person’s pose and appearance, offers higher diversity and leads to photorealistic results without texture deformations (see Fig. 1). 

In contrast to existing approaches that deal with major pose shifts by requiring parser maps denoting human body parts [14,23,28], or dense 3D correspondences [9,10] to fit human body topology by warping the textures, our approach can learn to generate realistic and authentic images without such detailed annotations. 

Our major contributions are as follows:

• We develop the first diffusion-based approach for pose-guided person synthesis task which can work under challenging pose transformations while preserving appearance, texture and global shape characteristics.

• To effectively model the complex interplay between appearance and pose information, we propose a texture diffusion module. This module exploits the correspondences between source and target appearance and pose details, hereby obtaining artefact free images. 

• In the sampling procedure, we introduce disentangled classifier-free guidance to tightly align the output image style and pose with the source image appearance and target pose, respectively. It ensures close resemblance between the conditions that are input to the generative model and the generated output.

• Our results on DeepFashion [11] and Market1501 [27] benchmarks set new state of the art. We also report a user study to evaluate the qualitative features of generated images. Finally, we demonstrate that synthesized images can be used to improve performance in downstream tasks e.g., person re-identification.

## 2. Related work

Pose-guided Person Image Synthesis: The problem of human pose transfer has been studied extensively during the recent years, especially with the unprecedented success of GAN-based models [15] for conditional image synthesis. 

An early attempt [13] proposes a coarse-to-fine approach to first generate a rough image with the target pose and then refine the results adversarially. 

The method simply concatenates the source image, the source pose, and the target pose as inputs to obtain the target image, which leads to feature misalignment. 

To address this issue, Essner et al. [3] attempt to disentangle the appearance and the pose of person images using VAE-based design and a UNet based skipconnection architecture. 

Siarohin et al. [20] improve the model by introducing deformable skip connections to spatially transform the textures, which decomposes the overall deformation by a set of local affine transformations. 

Subsequently, some works [9, 10, 19] use flow-based deformation to transform the source information to improve pose alignment. 

Ren et al. [19] propose GFLA that obtains the global flow fields and occlusion mask, which are used to warp local patches of the source image to match the required pose. 

Another group of works [9, 10] use geometric models that fit a 3D mesh human model onto the 2D image, and subsequently predict the 3D flow, which finally warps the source appearance. 

On the other hand, without any deformation operation, Zhu et al. [30] propose to progressively transform the source image by a sequence of transfer blocks. 

However, useful information can be lost during multiple transfers, which may result in blurry details. ADGAN [14] uses a texture encoder to extract style vectors for human body parts and gives them to several AdaIN residual blocks to synthesize the final image. 

Methods such as PISE [23], SPGnet [12] and CASD [28] make use of parsing maps to generate the final image. CoCosNet [25, 29] extracts dense correspondences between cross-domain images with attention-based operation. 

Recently, Ren et al. [18] propose a framework NTED based on neural texture extraction and distribution operation, which achieves superior results.

Diffusion Models: The existing GAN-based approaches attempt to directly transfer the style of the source image to a given target pose, which requires the architecture to model complex transformation of pose. 

In this work, we present a diffusion-based framework named PIDM that breaks the pose transformation process into several conditional denoising diffusion steps, in which each step is relatively simple to model. 

Diffusion models [6] are recently proposed generative models that can synthesize high-quality images. 

After success in unconditional generation, these models are extended to work in conditional generation settings, demonstrating competitive or even better performance than GANs. 

For class-conditioned generation, Dhariwal et al. [2] introduce classifier-guided diffusion, which is later adapted by GLIDE [16] to enable conditioning over CLIP textual representations. 

Recently, Ho et al. [7] propose a ClassifierFree Guidance approach that enables conditioning without requiring pretraining of the classifiers. 

In this work, we develop the first diffusion-based approach for poseguided person synthesis task. 

We also introduce disentangled classifier-free guidance to tightly align the output image style and pose with the source image appearance and target pose, respectively.


## 3. Proposed Method

Motivation: The existing pose-guided person synthesis methods [12,14,18,19,23,25,28] rely on GAN-based frameworks where the model attempts to directly transfer the style of the source image into a given target pose in a single forward pass. 

It is quite challenging to directly capture the complex structure of the spatial transformation, therefore current CNN-based architectures often struggle to transfer the intricate details of the cloth texture patterns. 

As a result, the existing methods yield noticeable artifacts, which become more evident when the generator needs to infer occluded body regions from the given source image. 

Motivated by these observations, we advocate that instead of learning the complex structure directly in a single step, deriving the final image using successive intermediate transfer steps can make the learning task simpler. 

To enable the above progressive transformation scheme, we introduce Person Image Diffusion Model or PIDM, a diffusionbased [6] person image synthesis framework that breaks down the generation process into several conditional denoising diffusion steps, each step being relatively simple to model. 

A single step in the diffusion process can be approximated by a simple isotropic Gaussian distribution. 

We observe that our diffusion-based texture transfer technique PIDM can bring the following benefits: 

(1) High-quality synthesis: Compared to previous GAN-based approaches, PIDM generates photo-realistic results when dealing with complex cloth-textures and extreme pose angles. 

(2) Stable training: Existing GAN-based approaches use multiple loss objectives alongside adversarial losses, which are often difficult to balance, resulting in unstable training. 

In contrast, PIDM exhibits better training stability and mode coverage. 

Also, our model is less prone to hyperparameters. 

(3) Meaningful interpolation: Our proposed PIDM allows us to achieve smooth and consistent linear interpolation in the latent space. 

(4) Flexibility: The models from existing work are usually task dependent, requiring different models for various tasks (e.g., separate models for unconditional, poseconditional, pose and style-conditional generation tasks). 

In contrast, in our case, a single model can be used to perform multiple tasks. 

Furthermore, PIDM inherits the flexibility and controllability of diffusion models that enable various downstream tasks (e.g., appearance control, see Fig. 7) using our model. 

Overall Framework: Fig. 2 shows the overview of the proposed generative model. 

Given a source image $x_{s}$ and a target pose $x_{p}$, our goal is to train a conditional diffusion model $p_{θ}(y|$x_{s}, x_{p}$) where the final output image y should not only satisfy the target pose matching requirement, but should also have the same style as in $x_{s}$. 

The denoising network $\epsilon_{θ}$ in PIDM is a UNet-based design composed of a noise prediction module $H_{N}$ and a texture encoder $H_{E}$. 

The encoder $H_{E}$ encodes the texture patterns of the source image $x_{s}$. 

To obtain multi-scale features we derive output from the different layers of $H_{E}$ resulting in stacked feature representation $F_{s} = [f_{1}, f_{2}, ..., f_{m}]$. 

To transfer rich multi-scale texture patterns from the source image distribution to the noise prediction module $H_{N}$ , we propose to use cross-attention based Texture diffusion blocks (TDB) that are embedded in different layers of $H_{N}$ . 

This allows the network to fully exploit the correspondences between the source and target appearances, thus resulting in distortion-free images. 

During inference, to amplify the conditional signal of $x_{s}$ and $x_{p}$ in the sampled images, we adapt the classifier-free guidance [7] in our sampling technique to achieve disentangled guidance. 

It not only improves the overall quality of the generation, but also ensures accurate transfer of texture patterns. 

We provide detailed analysis of the proposed generative model in Sec. 3.1, the Texture diffusion blocks in Sec. 3.2 and our disentangled guidance based sampling technique in Sec. 3.3.

## 3.1. Texture-Conditioned Diffusion Model

The generative modeling scheme of PIDM is based on the Denoising diffusion probabilistic model [6] (DDPM).


The general idea of DDPM is to design a diffusion process that gradually adds noise to the data sampled from the target distribution y0 ∼ q(y0), while the backward denoising process attempts to learn the reverse mapping. 

The denoising diffusion process eventually converts an isotropic Gaussian noise yT ∼ N (0, I) into the target data distribution in T steps. 

Essentially, this scheme divides a complex distribution-modeling problem into a set of simple denoising problems. The forward diffusion path of DDPM is a Markov chain with the following conditional distribution:

![Formula 1]()

where t ∼ [1, T] and β1, β2, ..., βT is a fixed variance schedule with βt ∈ (0, 1). Using the notation αt = 1 − βt and α¯t = Qt i=1 αi , we can sample from q(yt|y0) in a closed form at an arbitrary timestep t: yt = √ α¯ty0 + √ 1 − α¯t, where  ∼ N (0, I). 

The true posterior q(yt−1|yt) can be approximated by a deep neural network to predict the mean and variance of yt−1 with the following parameterization, 

![Formula 2]()

Noise prediction module HN : Instead of directly deriving µθ following [6], we predict the noise θ(yt, t, xp, xs) using our noise prediction module HN . 

The noisy image yt is concatenated with the target pose xp and passed through HN to predict the noise. 

xp will guide the denoising process and ensure that the intermediate noise representations and the final image follow the given skeleton structure. 

To inject the desired texture patterns into the noise predictor branch, we provide the multiscale features of the texture encoder HE through Texture diffusion blocks (TDB). 

To train the denoising process, we first generate a noisy sample yt ∼ q(yt|y0) by adding Gaussian noise  to y0, then train a conditional denoising model θ(yt, t, xp, xs) to predict the added noise using a standard MSE loss:

![Formula 3]()

Nichol et al. [17] present an effective learning strategy as an improved version of DDPM with fewer steps needed and applies an additional loss term Lvib to learn the variance Σθ. 

The overall hybrid objective that we adopt is as follows:

![Formula 4]()

## 3.2. Texture Diffusion Blocks (TDB) 

To mix the style of the source image within the noise prediction branch, we employ cross-attention based TDB units that are embedded in different layers of HN . 

Let F l h be the noise features in layer l of the noise prediction branch. 

Given the multiscale texture features Fs derived from the HE as input to TDB units, the attention module essentially computes the region of interest with respect to each query position, which is important to subsequently denoise the given noisy sample in the direction of the desired texture patterns. 

The keys K and values V are derived from HE while queries Q are obtained from noise features F l h . 

The attention operation is formulated as follows:

![Formula 5]()

where φlq, φlk, φlv are layer-specific 1 × 1 convolution operators. 

Wl refers to learnable weights to generate final cross-attended features F l o . 

We adopt TDB for the feature at specific resolutions, i.e., 32 × 32, 16 × 16, and 8 × 8.

## 3.3. Disentangled Guidance based Sampling

Once the model learns the conditional distribution, inference is performed by first sampling a Gaussian noise yT ∼ N (0, I) and then sampling from pθ(yt−1|yt, xp, xs), from t = T to t = 1 in an iterative manner. 

While the generated images using the vanilla sampling technique look photorealistic, they often do not strongly correlate with the conditional source image and target pose input. 

To amplify the effect of the conditioning signal xs and xp in the sampled images, we adapt Classifier-free guidance [7] in our multiconditional sampling procedure. 

We observe that in order to sample images that not only fulfil the style requirement, but also ensure perfect alignment with the target pose input, it is important to employ disentangled guidance with respect to both style and pose. 

To enable disentangled guidance, we use the following equation:

![Formula 6]()

where uncond = θ(yt, t, ∅, ∅) is the unconditioned prediction of the model, where we replace both conditions with the all-zeros tensor ∅. 

The pose-guided prediction and the style-guided prediction are respectively represented by pose = θ(yt, t, xp, ∅) − uncond and style = θ(yt, t, ∅, xs) − uncond. 

wp and ws are guidance scale corresponding to pose and style. 

In practice, the diffusion model learns both conditioned and unconditioned distributions during training by randomly setting conditional variables xp and xs = ∅ for η% of the samples, so that θ(yt, t, ∅, ∅) approximates p(y0) more faithfully. 

## 4. Experiments

Datasets: We carry out experiments on DeepFashion In-shop Clothes Retrieval Benchmark [11] and Market1501 [27] dataset. 

DeepFashion contains 52,712 highresolution images of fashion models. 

Following the same data configuration in [30], we split this dataset into training and testing subsets with 101,966 and 8,570 pairs, respectively. Skeletons are extracted by OpenPose [1]. 

Market1501 contains 32,668 low-resolution images. 

The images vary in terms of the viewpoints, background, illumination, etc. 

For both datasets, personal identities of the training and testing sets do not overlap. 

Evaluation Metrics: We evaluate the model using three different metrics. Structure Similarity Index Measure (SSIM) [22] and Learned Perceptual Image Patch Similarity (LPIPS) [26] are used to quantify the reconstruction accuracy. 

SSIM calculates the pixel-level image similarity, while LPIPS computes the distance between the generated images and reference images at the perceptual domain.

Frechet Inception Distance ` (FID) [5] is used to measure the realism of the generated images. 

It calculates the Wasserstein-2 distance between distributions of the generated images and the ground-truth images. 

Implementation Details: Our PIDM model has been trained with T = 1000 noising steps and a linear noise schedule. 

During training, we adopt an exponential moving average (EMA) of the denoising network weights with 0.9999 decay. In all experiments, we use a batch size of 8.

Adam optimizer is used with learning rate set to 2e−5 . 

For disentangled guidance, we use η = 10. 

For sampling, the values of wp and ws are set to 2.0. 

For the DeepFashion dataset, we train our model using 256 × 176 and 512 × 352 images. 

For Market-1501, we use 128 × 64 images. 

## 4.1. Quantitative and Qualitative Comparisons 

We quantitatively compare (Tab. 1) our proposed PIDM with several state-of-the-art methods, including Def-GAN [20], PATN [30], ADGAN [14], PISE [23], GFLA [19], DPTN [24], CASD [28], CocosNet2 [29] and NTED [18]. 

The experiments are done on both 256×176 and 512×352 resolution for DeepFashion and 128 × 64 resolution for Market-1501 dataset. 

Tab. 1 shows that our model achieves the best FID score indicating that our model can generate higher-quality images compared to the previous approaches. 

Furthermore, PIDM performs favorably against other methods in terms of the reconstruction metrics SSIM and LPIPS. 

This means that our model can generate images with not only accurate structures, but also can correctly transfer the texture of the source image to the target pose. 

In Fig. 3, we present a comprehensive visual comparison of our method with other state-of-the-art frameworks on DeepFashion dataset1 . 

The results of the baselines are obtained using pre-trained models provided by the corresponding authors. 

It can be observed that PISE [23] and ADGAN [14] fail to generate sharp images and cannot keep the consistency of shape and texture. 

While GFLA [19] somewhat preserves the texture in the source image, it struggles to obtain reasonable results for the invisible regions of the source image (e.g., 4∼5-th rows in the left column).

NTED [18] and CASD [28] improve the results slightly but they are still not able to adequately preserve the source appearance in complex scenarios (e.g., 1∼3-th rows in the left and 7∼10-th rows in the right column). 

In comparison, our proposed PIDM accurately retains the appearance of the source while also producing images that are more natural and sharper. 

Moreover, even if the target pose is complex (e.g., 10-th row in the left column), our method can still generate it precisely. 

We also visually compare our method with other baselines on the Market-1501 dataset in Fig. 4. 

Even with the complex background, PIDM still performs favorably in generating photo-realistic results as well as a consistent appearance with the source image. 

## 4.2. User Study

To demonstrate the effectiveness of our model in terms of human perception, we present our user study on 100 human participants. 

The user study is conducted to measure different aspects of our model with respect to ground truth images and generated images using other methods. 

(i) To compare with ground-truth images, we randomly select 30 generated images and 30 real images from the test set. 

Participants are required to determine whether a given image is real or fake. 

Following [28, 30], we adopt two metrics: R2G and G2R. 

R2G is the percentage of the real images classified as generated images and G2R is the percentage of the generated images classified as real images. 

(ii) To compare with other methods, we randomly select 30 sets of images where each set includes source image, target pose, ground-truth and images generated by our method and the baseline. 

The participants are required to select the best image with respect to the provided source image and ground truth. 

The participants are advised to provide their response based on the ability of each competing approach towards producing accurate texture patterns and pose structure. 

We quantify this using another metric called Jab which is defined as the percentage of images considered the best among all models. 

Higher values of these three metrics mean better performance. 

The result of the study is shown in Fig. 5. 

PIDM performs favorably against all baselines for all three metrics on DeepFashion dataset. 

For instance, PIDM images were interpreted as real images 48% (G2R) out of total cases, which is nearly 18% higher than the second best model. 

Our Jab score is 56% showing that the participants favor our approach more frequently than other methods.

## 4.3. Ablation Study

We perform multiple ablation studies to validate the merits of the proposed contributions. 

Tab. 2 shows the impact of texture diffusion blocks (TDB) and distangled classifier-free (DCF) guidance on the DeepFashion dataset. 

Baseline† neither employs a separate texture encoder nor uses TDB units. 

It only comprises of a UNet-based noise prediction module where the target pose and source image are concatenated with the noisy image and passed through the module to output a denoised version of the image. 

We extend the Baseline† with additional texture encoder to extract meaningful texture representations from the source image which are concatenated with respective layers of the noise prediction module. 

This is denoted as Baseline in Tab. 2. 

While the Baseline is able to generate realistic person images, it has a limited ability to retain the appearance of the source image as shown in Fig. 6. 

To effectively model the complex interplay between appearance and pose information, we integrate texture diffusion module (TDB) into the noise prediction module. 

We refer to this as Baseline+TDB. 

On the DeepFahsion dataset, the Baseline achieves an FID score of 9.8510. 

In comparison to the Baseline, the Baseline+TDB improves the FID by a margin of 2.3377. 

While the generated images using the vanilla sampling technique look photo-realistic, they often do not strongly correlate with the conditional source image and target pose input. 

To improve the correlation, we first use vanilla classifier-free (CF) guidance during the sampling. 

While the vanilla CF guidance slightly improves the results, we observe that in order to sample images that not only fulfil the style requirement, but also ensure perfect alignment with the target pose input, it is important to employ disentangled guidance with respect to both style and pose. 

We here refer to our final proposed approach Baseline+TDB+DCF-guidance where we employ distangled classifier-free (DCF) guidance to tightly align the output image style and pose with the source image appearance and target pose. 

In comparison to the Baseline+TDB, our proposed DCF guidance improves the FID, SSIM and LPIPS by a margin of 1.1462, 0.0134 and 0.0194 respectively.

## 4.4. Appearance Control and Editing

Our proposed PIDM inherits the flexibility and controllability of diffusion models that enable appearance control by combining cloth textures extracted from style images into the reference image.

Given a source (style) image xs, a reference image y ref , and a binary mask m that marks the region of interest in the reference image, the problem is to generate an image y¯, s.t. the appearance of y¯  m is consistent with the source image xs, while the complementary area remains same.

To achieve this, we first calculate yreftin a time step t: yreft =√α¯tyref +√1 − α¯t, where  ∼ N (0, I).

During inference, starting from a Gaussian noise yT ∼ N (0, I), we predict yt iteratively using the trained diffusion model from t = T to t = 1. 

In each step t, we use the binary mask m to retain the unmasked regions of y ref by using the relation: yt = myt + (1−m)y ref t.

The results of the appearance control task are shown in Fig. 7(a). 

We observe that our model can seamlessly combine the areas of interest and generate coherent output images with realistic textures. 

Style Interpolation: In Fig. 7(b), we show our interpolation results between two style images. 

We use DDIM [21] sampling to enable smooth interpolation using our trained diffusion model. 

Specifically, we use spherical linear interpolation between noises y 1 T and y 2 T , and linear interpolation between style features F 1 s and F 2 s. 

As shown in Fig. 7(b), the texture of the clothes gradually changes from the style of the left source image to that of the right source image.

## 4.5. Application to Person Re-identification

Here, we evaluate the applicability of the images generated by our PIDM as a source of data augmentation for the downstream task of person re-identification (re-ID). 

We perform the re-ID experiment on Market-1501 dataset. 

We randomly select 20%, 40%, 60%, and 80% of total training set of real Market-1501 dataset such that at least one image per identity is selected. 

We denote the obtained set as Dtr. 

We first initialize a ResNet50 backbone network using Dtr for the re-ID task. 

We refer to this as Standard in Tab. 3. 

Then, we augment Dtr with images generated by our PIDM. 

The images are generated using randomly chosen image of the same identity in Dtr as the source image. 

The target pose is randomly selected from Dtr. 

Consequently, an augmented training set Daug is created from all these generated images and the real set Dtr. 

We finetune the ResNet50 backbone using the augmentation set Daug. 

As shown in Tab. 3, PIDM achieves consistent improvements over previous works.

## 5. Conclusion

We proposed a diffusion-based approach, PIDM, for pose-guided person image generation. 

PIDM disintegrates the complex transfer problem into a series of simpler forward-backward denoising steps. 

This helps in learning plausible source-to-target transformation trajectories that result in faithful textures and undistorted appearance details. 

We introduce texture diffusion module and disentangled classifier-free guidance to accurately model the correspondences between appearance and pose information available in source and target images. 

We show the effectiveness of PIDM on two datasets by performing extensive qualitative, quantitative and human-based evaluations. 

In addition, we show how our generated images can help in downstream tasks such as person re-identification.