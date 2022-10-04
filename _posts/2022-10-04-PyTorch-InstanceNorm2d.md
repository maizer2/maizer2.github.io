---
layout: post
title: "PyTorch InstanceNorm2d()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[PyTorch - torch.nn.InstanceNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```python
torch.nn.InstanceNorm2d(
    num_features: int,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False,
    device=None,
    dtype=None
) -> None:
```

```Example
m = torch.nn.InstanceNorm2d(100)
m = torch.nn.InstanceNorm2d(100, affine=True)
input = torch.randn(20, 100, 35, 45)
output = m(input)
```

### explain

* Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf).
    * [위 논문에서 사용된 Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf)을 4차원 입력 데이터에 적용시킨다.

[Instance Normalization 리뷰]()
---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> INSTANCENORM2D, PyTorch, [https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d)