---
layout: post
title: "PyTorch bmm()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[PyTorch - torch.bmm](https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```python
input = torch.randn(10, 3, 4)
mat2 = torch.randn(10, 4, 5)
res = torch.bmm(input, mat2)
res.size()
# torch.Size([10, 3, 5])
```

### explain

* Performs a batch matrix-matrix product of matrices stored in input and mat2.
    * Input과 mat2에 저장된 행렬의 batch matrix-matrix 곱셈을 수행한다.
* input and mat2 must be 3-D tensors each containing the same number of matrices.
    * Input과 mat2는 각각 동일한 수의 행렬을 포함하는 3D 텐서여야 합니다.
* If input is a (b $\times$ n $\times$ m)(b×n×m) tensor, mat2 is a (b $\times$ m $\times$ p)(b×m×p) tensor, out will be a (b $\times$ n $\times$ p)(b×n×p) tensor.
    * 만약 입력 텐서가 (b $\times$ n $\times$ m)(b×n×m) 이고, mat2 텐서가 (b $\times$ m $\times$ p)(b $\times$ m $\times$ p) 이면, out 텐서는 (b $\times$ n $\times$ p)(b $\times$ n $\times$ p) 이다.

$$out_{i}=x_{i}^{exponent}$$

* Note

    * This function does not [broadcast](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics). For broadcasting matrix products, see [torch.matmul()](https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul).
        * 해당 함수는 [broadcast](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics)를 지원하지 않습니다. broadcasting matrix가 지원되는 함수는, [torch.matmul()](https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul)을 참조하세요.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> TORCH.BMM, PyTorch, [https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm](https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm)