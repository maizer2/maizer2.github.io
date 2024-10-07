---
layout: post
title: "pytorch 1.7.1 install error"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
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