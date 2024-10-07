---
layout: post
title: "PyTorch pow()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[PyTorch - torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html#torch.pow)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```python
a = torch.randn(4)
# tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
torch.pow(a, 2)
# tensor([ 0.1875,  1.5561,  0.4670,  0.0779])

exp = torch.arange(1., 5.)
# tensor([ 1.,  2.,  3.,  4.])
a = torch.arange(1., 5.)
# tensor([ 1.,  2.,  3.,  4.])
torch.pow(a, exp)
# tensor([   1.,    4.,   27.,  256.])
```

### explain

* Takes the power of each element in input with exponent and returns a tensor with the result.
    * 입력 Tensor와 exponent의 각 요소를 power 함수에 적용시키고 결과 Tensor를 반환합니다.
* exponent can be either a single float number or a Tensor with the same number of elements as input.
    * exponent는 입력과 동일한 수의 요소를 가진 단일 float 또는 Tensor 일 수 있습니다.
* When exponent is a scalar value, the operation applied is:
    * exponent가 단일 값(float)일 경우, 적용되는 연산은 다음과 같습니다:

$$out_{i}=x_{i}^{exponent}$$

* When exponent is a tensor, the operation applied is:
    * exponent가 tensor일 경우, 적요오디는 연산은 다음과 같습니다:

$$out_{i}=x_{i}^{exponent_{i}}$$

* When exponent is a tensor, the shapes of input and exponent must be broadcastable.
    * exponent가 tensor일 경우 input과 exponent는 반드시 broadcastable 해야합니다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> TORCH.POW, PyTorch, [https://pytorch.org/docs/stable/generated/torch.pow.html#torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html#torch.pow)