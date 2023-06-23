---
layout: post 
title: "(VITON)VITON-GAN: Virtual Try-on Image Generator Trained with Adversarial Loss Translation"
categories: [1. Computer Engineering]
tags: [1.7. Paper Review]
---

### [VITON Paper List](https://maizer2.github.io/1.%20computer%20engineering/2022/08/01/paper-of-VITON.html)

### [$$\mathbf{VITON-GAN:Virtual\;Try-on\;Image\;Generator\;Trained\;with\;Adversarial\;Loss}$$](https://arxiv.org/pdf/1711.08447v4.pdf)

##### $$\mathbf{Shion\;Honda}$$

##### $$\mathbf{The\;University\;of\;Tokyo}$$

##### $$\mathbf{The\;International\;Research\;Center\;for\;Neurointelligence}$$

![Figure-1](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-08-04-(VITON)VITON-GAN-Virtual-Try-on-Image-Generator-Trained-with-Adversarial-Loss/Figure-1.png)

> Figure 1: Samples from generated images. The models in the left column virtually wear the clothes from the top row.
>> 그림 1: 생성된 이미지에서 샘플. 왼쪽 열에 있는 모델들은 사실상 맨 윗줄의 옷을 입는다.

### $\mathbf{Abstract}$

> Generating a virtual try-on image from in-shop clothing images and a model person’s snapshot is a challenging task because the human body and clothes have high flexibility in their shapes. In this paper, we develop a Virtual Try-on Generative Adversarial Network (VITON-GAN), that generates virtual try-on images using images of in-shop clothing and a model person. This method enhances the quality of the generated image when occlusion is present in a model person’s image (e.g., arms crossed in front of the clothes) by adding an adversarial mechanism in the training pipeline.
>> 매장 내 의류 이미지와 모델 인물의 스냅숏에서 가상 체험 이미지를 생성하는 것은 인체와 옷의 형태 유연성이 높기 때문에 어려운 작업이다. 본 논문에서는 매장 내 의류와 모델 인물의 이미지를 사용하여 가상 체험 이미지를 생성하는 가상 체험 생성 적대 네트워크(VITON-GAN)를 개발한다. 이 방법은 훈련 파이프라인에 적대적 메커니즘을 추가하여 모델 사람의 이미지(예: 옷 앞에서 팔짱을 낀 경우)에 폐색이 있을 때 생성된 이미지의 품질을 향상시킨다.

### $\mathbf{1\;Introduction}$

> Despite the recent growth of online apparel shopping, there is a tremendous demand by consumers for buying clothes after trying them on in real shops. If e-commerce sites can offer virtual try-on images from a snapshot of the customer, they can improve their user experience.
>> 최근 온라인 의류 쇼핑의 성장세에도 불구하고 실제 매장에서 입어본 후 옷을 구매하려는 소비자들의 수요가 엄청나다. 전자 상거래 사이트가 고객의 스냅샷에서 가상 체험 이미지를 제공할 수 있다면 사용자 경험을 개선할 수 있습니다.

> To solve this task, previous approaches combined a U-net gener- ator and thin plate spline (TPS) transform [HWW∗18] [WZL∗18]. The TPS transform keeps the patterns and letters of the clothes accurate when mapped on the human body. The latest work (CP- VTON) [WZL∗ 18] used a human parser [GLZ∗ 17] and pose esti- mator [CSWS17] in its pipeline to extract the person’s representa- tion (explanatory variable) independent of wearing clothes (objec-tive variable). As shown in Figure 3, however, we report that these methods fail when arms are crossed in front of the clothes (occlu- sion), generating blurred arms due to reconstruction loss.
>> 이 과제를 해결하기 위해 이전 접근 방식은 U-net 생성기와 얇은 판 스플라인(TPS) 변환을 결합했다[HWW1818] [WZL1818]. TPS 변환은 인체에 매핑될 때 옷의 패턴과 글자를 정확하게 유지합니다. 최신 작업(CP-VTON) [WZL 18 18]은 파이프라인에서 인간 파서 [GLZ 17 17]과 포즈 추정기 [CSWS17]를 사용하여 옷(객체-목적 변수)과 무관한 사람의 표현(설명 변수)을 추출했다. 그러나 그림 3에서 보듯이, 우리는 이러한 방법이 옷 앞에서 팔짱을 끼면 실패하여(폐색) 재구성 손실로 인해 팔이 흐려진다고 보고한다.

> For generating realistic images, generative adversarial networks (GANs) have been successfully used [BDS19] [KLA18]. Unlike other generative models using reconstruction loss such as VAE, GANs are able to generate fine, high-resolution, and realistic im- ages because adversarial loss can incorporate perceptual features that are difficult to define mathematically.
>> 현실적인 이미지를 생성하기 위해 생성적 적대 네트워크(GAN)가 성공적으로 사용되었다 [BDS19] [KLA18]. VAE와 같은 재구성 손실을 사용하는 다른 생성 모델과 달리, GAN은 적대적 손실이 수학적으로 정의하기 어려운 지각적 특징을 통합할 수 있기 때문에 미세하고 고해상도, 현실적인 이미지를 생성할 수 있다.

> In this paper, we propose an image generator that alleviates the occlusion problem, called Virtual Try-On GAN (VITON-GAN). This generator consists of two modules, the geometry matching module (GMM) and the try-on module (TOM) as was implemented in CP-VTON, except adversarial loss is additionally included in the TOM to address the occlusion problem.
>> 본 논문에서는 VITON-GAN(Virtual Try-On GAN)이라는 폐색 문제를 완화하는 이미지 생성기를 제안한다. 이 생성기는 폐색 문제를 해결하기 위해 적대적 손실이 TOM에 추가로 포함되는 것을 제외하고 CP-VTON에서 구현된 것과 같은 두 개의 모듈, 즉 기하 매칭 모듈(GMM)과 트라이온 모듈(TOM)로 구성된다.

### $\mathbf{2.\;Methods}$

> The whole training pipeline of VITON-GAN is presented in Figure 2. There are three major updates from CP-VTON. First, TOM is trained adversarially against the discriminator that uses the TOM result image, in-shop clothing image, and person representation as inputs and judges whether the result is real or fake. Second, the loss function of GMM includes the L1 distance between the generated and real images of clothes layered on the body. Finally, random horizontal flipping is used for data augmentation. The source codes and the trained model are available at https://github.com/ shionhonda/viton- gan.
>> VITON-GAN의 전체 교육 파이프라인이 그림 2에 제시되어 있다. CP-VTON의 주요 업데이트는 세 가지입니다. 첫째, TOM은 TOM 결과 이미지, 매장 내 의류 이미지 및 인물 표현을 입력으로 사용하는 판별기에 대해 적대적 훈련을 받고 결과가 진짜인지 가짜인지를 판단한다. 둘째, GMM의 손실 기능은 생성된 옷의 이미지와 실제 옷의 이미지 사이의 L1 거리를 포함한다. 마지막으로, 무작위 수평 플립은 데이터 증강에 사용된다. 소스 코드와 훈련된 모델은 https://github.com/ shionhonda/viton-gan에서 이용할 수 있다.

![Figure-2](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-08-04-(VITON)VITON-GAN-Virtual-Try-on-Image-Generator-Trained-with-Adversarial-Loss/Figure-2.png)

> Figure 2: Overview of the VITON-GAN training pipeline.
>> 그림 2: VITON-GAN 교육 파이프라인 개요

### $\mathbf{3.\;Experiments\;and\;Results}$

> To show the effect on the occlusion problem of VITON-GAN, a virtual try-on experiment was conducted using the same dataset as CP-VTON. The dataset contains 16,253 female model’s snapshots and top clothing image pairs, which were split into 13,221, 1,000, and 2,032 pairs for training, validation, and test sets, respectively. All result images presented in this paper are from the test set.
>> VITON-GAN 폐색 문제에 미치는 영향을 보여주기 위해 CP-VTON과 동일한 데이터 세트를 사용하여 가상 트라이온 실험을 수행하였다. 데이터 세트에는 16,253개의 여성 모델의 스냅샷과 상위 의류 이미지 쌍이 포함되어 있으며, 이들은 각각 훈련, 검증 및 테스트 세트를 위해 13,221개, 1,000개, 2,032개 쌍으로 분할되었다. 이 문서에 제시된 모든 결과 이미지는 테스트 세트에서 가져온 것입니다.

> As shown in Figure 3, VITON-GAN generated hands and arms more clearly than CP-VTON in occlusion cases. However, arm gen- eration failed when the model’s original clothing was half-sleeve and the tried-on clothing was long-sleeve (see Figure 4: upper row). This was because the TPS transform was not able to deal with topo- logical changes that often occurred in the case of occlusion with long-sleeve shirts. Also, although in most cases VITON-GAN gen- erated images as fine as CP-VTON (see Figure 1), it occasionally generated blurred images as shown in the lower row of Figure 4.
>> 그림 3과 같이 폐색시 VITON-GAN은 CP-VTON보다 손과 팔을 더 명확하게 발생시켰다. 그러나 모델의 원래 옷이 반팔이고 입어본 옷이 긴팔일 때 팔 생성에 실패하였다(그림 4: 윗줄 참조). TPS 변환이 긴 소매 셔츠로 폐색할 경우 종종 발생하는 위상 변화를 감당할 수 없었기 때문이다. 또한 대부분의 경우 VITON-GAN은 CP-VTON만큼 미세한 이미지를 생성했지만(그림 1 참조), 때때로 그림 4의 아래쪽 행에 표시된 것처럼 흐릿한 이미지를 생성했다.

### $\mathbf{4.\;Conclusions}$

> Here, we propose a virtual try-on image generator from 2D im- ages of a person and top clothing that alleviates the occlusion prob- lem. Future work will include improving the quality of generated parts of the human body and addressing topological changes in the clothes.
>> 여기서는 폐색 문제를 완화하는 사람과 상의의 2D 이미지에서 가상 트라이온 이미지 생성기를 제안한다. 향후 작업에는 인체의 생성된 부위의 품질을 개선하고 옷의 위상 변화를 다루는 것이 포함될 것이다.

![Figure-3](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-08-04-(VITON)VITON-GAN-Virtual-Try-on-Image-Generator-Trained-with-Adversarial-Loss/Figure-3.png)

> Figure 3: Successful cases of the proposed method.
>> 그림 3: 제안된 방법의 성공 사례

![Figure-4](https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.7.%20Literature%20Review/2022-08-04-(VITON)VITON-GAN-Virtual-Try-on-Image-Generator-Trained-with-Adversarial-Loss/Figure-4.png)

> Figure 4: Failed cases of the proposed method.
>> 그림 4: 제안된 방법의 실패 사례

### $\mathbf{Acknowledgements}$

> This work was supported by the Chair for Frontier AI Education, the University of Tokyo. Also, we would like to thank I. Sato, T. Harada, T. Mano, and C. Yokoyama for correcting this paper.
>> 이 작업은 도쿄 대학 프런티어 AI 교육 의장의 지원을 받았다. 또한, 우리는 I에게 감사하고 싶다. 사토, T.하라다, T. 마노, 그리고 C. 이 종이를 고쳐준 요코야마.

---

### $\mathbf{References}$

[BDS19] BROCK A., DONAHUE J., SIMONYAN K.: Large scale GAN training for high fidelity natural image synthesis. In International Con- ference on Learning Representations (2019). 1

[CSWS17] CAO Z., SIMON T., WEI S.-E., SHEIKH Y.: Realtime multi- person 2d pose estimation using part affinity fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2017). 1

[GLZ∗17] GONG K., LIANG X., ZHANG D., SHEN X., LIN L.: Look into person: Self-supervised structure-sensitive learning and a new benchmark for human parsing. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2017). 1

[HWW∗18] HAN X., WU Z., WU Z., YU R., DAVIS L. S.: Viton: An image-based virtual try-on network. In Proceedings of the IEEE Confer- ence on Computer Vision and Pattern Recognition (2018). 1

[KLA18] KARRAS T., LAINE S., AILA T.: A style-based genera- tor architecture for generative adversarial networks. arXiv preprint arXiv:1812.04948 (2018). 1

[WZL∗18] WANG B., ZHENG H., LIANG X., CHEN Y., LIN L., YANG M.: Toward characteristic-preserving image-based virtual try-on net- work. In Proceedings of the European Conference on Computer Vision (2018). 1