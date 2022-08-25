---
layout: post
title: "ImportError: No module named 'upfirdn2d_plugin'"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

### [PASTA-GAN-plusplus - Issues](https://github.com/xiezhy6/PASTA-GAN-plusplus/issues/4#issue-1349327246)

### Error message

```
warnings.warn('Failed to build CUDA kernels for upfirdn2d. Falling back to slow reference implementation. Details:\n\n' + traceback.format_exc())
Setting up PyTorch plugin "upfirdn2d_plugin"... Failed!
/workspace/PASTA-GAN-plusplus/torch_utils/ops/upfirdn2d.py:34: UserWarning: Failed to build CUDA kernels for upfirdn2d. Falling back to slow reference implementation. Details:

Traceback (most recent call last):
  File "/workspace/PASTA-GAN-plusplus/torch_utils/ops/upfirdn2d.py", line 32, in _init
    _plugin = custom_ops.get_plugin('upfirdn2d_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
  File "/workspace/PASTA-GAN-plusplus/torch_utils/custom_ops.py", line 110, in get_plugin
    torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)
  File "/opt/conda/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 986, in load
    return _jit_compile(
  File "/opt/conda/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1213, in _jit_compile
    return _import_module_from_library(name, build_directory, is_python_module)
  File "/opt/conda/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1560, in _import_module_from_library
    file, path, description = imp.find_module(module_name, [path])
  File "/opt/conda/lib/python3.8/imp.py", line 296, in find_module
    raise ImportError(_ERR_MSG.format(name), name=name)
ImportError: No module named 'upfirdn2d_plugin'
```

### torch_utils/ops/upfirdn2d.py

line 31 ~ 35

```python
try:
   _plugin = custom_ops.get_plugin('upfirdn2d_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
except:
   warnings.warn('Failed to build CUDA kernels for upfirdn2d. Falling back to slow reference implementation. Details:\n\n' + traceback.format_exc())
```

### Check StyleGAN2 Issue

[StyleGAN2 issue](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/39#issuecomment-781515268)

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> [Python] 경고 메시지(warnings) 숨기기, Yonggeun Shin, write June/26/2020, visit Agust/24/2022, [https://yganalyst.github.io/etc/memo_15/](https://yganalyst.github.io/etc/memo_15/)