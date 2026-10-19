---
layout: post
categories: [1. Computer Engineering, 1. Artificial Intelligence, 3. Frameworks, 1. PyTorch]
title: "pytorch 1.7.1 install error"
tags: [Pillow, pip install]
---

### [python-pillow - Issues](https://github.com/python-pillow/Pillow/issues/4242)

### Error message

```
...

The headers or library files could not be found for zlib,
a required dependency when compiling Pillow from source.
```

### Solution

```
pip3 -U --force-reinstall pip
```