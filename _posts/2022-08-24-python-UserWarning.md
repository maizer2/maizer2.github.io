---
layout: post
title: "warnings.warn("The default behavior for interpolate/upsample with float scale_factor changed")"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

### [PASTA-GAN-plusplus - Issues](https://github.com/xiezhy6/PASTA-GAN-plusplus/issues/5#issue-1349366828)

```
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:3103: UserWarning: The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details.
warnings.warn("The default behavior for interpolate/upsample with float scale_factor changed "
```

### solution

Turn off the UserWarring message

torch_utils/misc.py

add this code in torch_utils/misc.py

```python
warnings.filterwarnings(action='ignore')
```

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> [Python] 경고 메시지(warnings) 숨기기, Yonggeun Shin, write June/26/2020, visit Agust/24/2022, [https://yganalyst.github.io/etc/memo_15/](https://yganalyst.github.io/etc/memo_15/)