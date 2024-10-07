---
layout: post
title: "Torchvision RandomHorizontalFlip()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[Pytorch - torchvision.transforms.RandomHorizontalFlip](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#randomhorizontalflip)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```python
torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

### explain

> Horizontally flip the given image randomly with a given probability.
>> 주어진 확률(p)로 주어진 이미지를 Horizontally flip 합니다.[Horizontally flip은 수평으로 뒤집는것을 의미합니다.]

> If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions
>> 만약 이미지가 torch Tensor 객체라면, […, H, W] 리스트 모양으로 예상할 수 있습니다. …의 의미는 임의 숫자의 선행 차원을 의미합니다. [선행 차원이란 Tensor가 가지고 있던 차원수를 의미하는 것 같습니다. 크기 변환에 의미가 있기 때문에 차원 수는 유지된다고 볼 수 있습니다.]

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> RANDOMHORIZONTALFLIP, PyTorch, [https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#randomhorizontalflip](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#randomhorizontalflip)