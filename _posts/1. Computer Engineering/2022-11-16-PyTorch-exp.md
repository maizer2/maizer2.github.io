---
layout: post
title: "PyTorch exp()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[PyTorch - torch.exp](https://pytorch.org/docs/stable/generated/torch.exp.html#torch.exp)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>



```python
torch.exp(torch.tensor([1, 2, 3], [4, 5, 6]))
# tensor([e^1,  e^2, e^3], [e^4, e^5, e^6])
```

### explain

* Returns a new tensor with the exponential of the elements of the input tensor *input*.
    * 입력 텐서인 *input*의 각 요소를 자연 상수를 밑으로 하는 exponential(지수)로서 변환합니다.

$$y_{i} = e^{x_{i}}$$

#### Parameters

* input: torch.Tensor
    * the input tensor.
        * 입력 텐서입니다.

#### Keyword Arguments

* out: Optional[torch.Tensor] = None
    * the output tensor
        * 출력할 텐서입니다.
        * 입력 텐서의 각 요소를 자연 상수를 밑으로 하는 지수로 변환되어 계산됩니다.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> TORCH.EXP, PyTorch, [https://pytorch.org/docs/stable/generated/torch.exp.html#torch.exp](https://pytorch.org/docs/stable/generated/torch.exp.html#torch.exp)