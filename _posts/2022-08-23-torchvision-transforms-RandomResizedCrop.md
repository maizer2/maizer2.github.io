---
layout: post
title: "Pytorch torchvision.transforms.RandomResizedCrop()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[Pytorch - torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html#randomresizedcrop)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```python
torchvision.transforms.RandomResizedCrop(size, 
                                         scale=(0.08, 1.0), 
                                         ratio=(0.75, 1.3333333333333333), 
                                         interpolation=<InterpolationMode.BILINEAR: 'bilinear'>
                                         )
```

### explain

> Crop a random portion of image and resize it to a given size.
>> 이미지의 랜덤한 부분으로 자른 후 주어진 사이즈로 사이즈 변환합니다.

> If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions
>> 만약 이미지가 torch Tensor 객체라면, […, H, W] 리스트 모양으로 예상할 수 있습니다. …의 의미는 임의 숫자의 선행 차원을 의미합니다. [선행 차원이란 Tensor가 가지고 있던 차원수를 의미하는 것 같습니다. 크기 변환에 의미가 있기 때문에 차원 수는 유지된다고 볼 수 있습니다.]

> A crop of the original image is made: the crop has a random area (H * W) and a random aspect ratio. This crop is finally resized to the given size. This is popularly used to train the Inception networks.
>> 입력 이미지를 자랐다면, 이미지 내의 랜덤한 위치인 (H * W)과 랜덤한 방향의 각도로 변환(잘리게)됩니다. 최종적으로 주어진 사이즈로 변환되게 됩니다. 이 과정은 전형적으로 Inception networks 훈련에 사용됩니다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> RANDOMRESIZEDCROP, PyTorch, [https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html#randomresizedcrop](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html#randomresizedcrop)