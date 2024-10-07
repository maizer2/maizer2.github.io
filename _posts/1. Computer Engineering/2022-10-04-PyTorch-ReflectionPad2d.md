---
layout: post
title: "PyTorch ReflectionPad2d()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[PyTorch - torch.nn.ReflectionPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```python
torch.nn.ReflectionPad2d(
    padding: Tuple[int, int, int, int]  # Tuple[left, right, top, bottom]
)
```

```Example
m = torch.nn.ReflectionPad2d(2)
input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
print(input)
print(m(input))

'''
tensor([[[[0., 1., 2.],
          [3., 4., 5.],
          [6., 7., 8.]]]])
tensor([[[[8., 7., 6., 7., 8., 7., 6.],
          [5., 4., 3., 4., 5., 4., 3.],
          [2., 1., 0., 1., 2., 1., 0.],
          [5., 4., 3., 4., 5., 4., 3.],
          [8., 7., 6., 7., 8., 7., 6.],
          [5., 4., 3., 4., 5., 4., 3.],
          [2., 1., 0., 1., 2., 1., 0.]]]])
'''
```

### explain

* Pads the input tensor using the reflection of the input boundary.
    * 입력 데이터의 반전 값을 padding으로 추가

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> REFLECTIONPAD2D, PyTorch, [https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d)