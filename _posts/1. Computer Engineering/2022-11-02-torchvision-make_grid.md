---
layout: post
title: "Torchvision make_grid()"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

[Pytorch - torchvision.utils.make_grid](https://pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html)<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```python
torchvision.utils.make_grid(
    tensor: Union[Tensor, List[Tensor]], 
    nrow: int = 8, 
    padding: int = 2, 
    normalize: bool = False, 
    value_range: Optional[Tuple[int, int]] = None, 
    scale_each: bool = False, 
    pad_value: float = 0.0, 
    **kwargs
) -> Tensor
```

* tensor (Tensor or list) – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
    * (Batch_size x Channels x Hight x Width) 모양의 4차원 미니베치 tensor 혹은 모두 같은 크기의 이미지 리스트

* nrow (int, optional) – Number of images displayed in each row of the grid. The final grid size is (B / nrow, nrow). Default: 8.
    * grid의 각 행에 표시되는 이미지의 개수. 마지막 grid 크기는 ( B / nrow, nrow)이다. 기본 값으로 8.

* padding (int, optional) – amount of padding. Default: 2.
    * padding의 양. 기본 값으로 2.

* normalize (bool, optional) – If True, shift the image to the range (0, 1), by the min and max values specified by value_range. Default: False.
    * 만약 True일 경우, (0, 1) 범위로 이미지를 이동한다, 아래 value_range에서 지정돤 최소, 최대값을 기준으로 한다. 기본 값으로: False.

* value_range (tuple, optional) – tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. By default, min and max are computed from the tensor.
    * tuple(min, max)에서 최소, 최대값은 숫자 이고, 이 값들은 이미지 정규화에 사용됩니다. 기본적으로 최소, 최대값은 tensor로부터 계산됩니다.

* scale_each (bool, optional) – If True, scale each image in the batch of images separately rather than the (min, max) over all images. Default: False.
    * True일 경우, 모든 이미지에 대한 (최소, 최대) 대신 이미지 배치에서 각 이미지의 크기를 개별적으로 조정합니다. 기본 값으로: False.

* pad_value (float, optional) – Value for the padded pixels. Default: 0.
    * padding 픽셀을 위한 값. 기본 값으로: 0.

---

### Example

[Example gallery > Visualization utilities](https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py)

```
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

dog1_int = read_image(str(Path('assets') / 'dog1.jpg'))
dog2_int = read_image(str(Path('assets') / 'dog2.jpg'))
dog_list = [dog1_int, dog2_int]

grid = make_grid(dog_list)
show(grid)
```

![dog1_int, dog2_int](https://pytorch.org/vision/stable/_images/sphx_glr_plot_visualization_utils_001.png)

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> MAKE_GRID, torchvision, [https://pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html](https://pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html)