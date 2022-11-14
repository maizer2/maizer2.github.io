---
layout: post
title: "PyTorch div()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[PyTorch - torch.div](https://pytorch.org/docs/stable/generated/torch.div.html#torch.div)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```python
x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
torch.div(x, 0.5)
# tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])

a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
...               [ 0.1815, -1.0111,  0.9805, -1.5923],
...               [ 0.1062,  1.4581,  0.7759, -1.2344],
...               [-0.1830, -0.0313,  1.1908, -1.4757]])
b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
torch.div(a, b)
# tensor([[-0.4620, -6.6051,  0.5676,  1.2639],
#         [ 0.2260, -3.4509, -1.2086,  6.8990],
#         [ 0.1322,  4.9764, -0.9564,  5.3484],
#         [-0.2278, -0.1068, -1.4678,  6.3938]])

torch.div(a, b, rounding_mode='trunc')
# tensor([[-0., -6.,  0.,  1.],
#         [ 0., -3., -1.,  6.],
#         [ 0.,  4., -0.,  5.],
#         [-0., -0., -1.,  6.]])

torch.div(a, b, rounding_mode='floor')
# tensor([[-1., -7.,  0.,  1.],
#         [ 0., -4., -2.,  6.],
#         [ 0.,  4., -1.,  5.],
#         [-1., -1., -2.,  6.]])
```

### explain

* Divides each element of the input *input* by the corresponding element of *other*.
    * 입력 텐서인 *input*의 각 요소를 *other* 텐서의 각 요소로 나눕니다.

$$\text{out}_i = \frac{\text{input}_i}{\text{other}_i}$$

* Note

    * By default, this performs a “true” division like Python 3. See the rounding_mode argument for floor division.
        * 기본적으로, 이 나누기는 python3와 같은 "true" 나누기를 수행합니다. floor division은 rounding_mode argument를 참고하세요.

#### Parameters

* input: torch.Tensor
    * the dividend
        * 나누는 대상
* other: Union[torch.Tensor, int, float]
    * the divisor
        * 나누는 것

#### Keyword Arguments

* rounding_mode: Optional[str] = None
    * Type of rounding applied to the result:
        * Rounding의 유형에 따라 적용되는 결과:

        * None - default behavior. Performs no rounding and, if both input and other are integer types, promotes the inputs to the default scalar type. Equivalent to true division in Python (the / operator) and NumPy’s np.true_divide.
            * None - 기본 동작. 반올림하지 않고, 만약 input과 other이 정수형일 경우, input을 scalar 형식으로 변환합니다. 파이썬의 나눗셈 연산자인 "/"과 동일하고, Numpy의 np.true_divide와 동일합니다.
        * "trunc" - rounds the results of the division towards zero. Equivalent to C-style integer division.
            * "trunc" - 나눗셈의 결과를 0으로 반올림 합니다. C언어의 정수 나눗셈과 동일합니다.
        * "floor" - rounds the results of the division down. Equivalent to floor division in Python (the // operator) and NumPy’s np.floor_divide.
            * "floor" - 반올림 결과를 반올림합니다. 파이썬의 "//" 연산자와 동일하고, Numpy의 np.floor_divide와 동일합니다.

* out: Optional[torch.Tensor] = None
    * the output tensor
        * 출력할 텐서입니다.
---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> TORCH.DIV, PyTorch, [https://pytorch.org/docs/stable/generated/torch.div.html#torch.div](https://pytorch.org/docs/stable/generated/torch.div.html#torch.div)