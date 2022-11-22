---
layout: post
title: "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

### Problem

DDP를 사용하여 훈련 중 "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"와 같은 런타임 에러가 발생했다. 

이전 싱글 GPU로 훈련을 진행할 때도 같은 에러가 나온적이 있어 대략적으로 파이토치의 autograd의 그레프 문제라는 것을 알고 있었고 clone과 detach를 해주는 등 예상가는 변수에 조치를 취해줬지만 해결이되지 않았다. 

이에대한 파이토치 그룹 커뮤니티가 활발하게 활성화 돼 있는데, 이를 참고하여 해결하였다.

> Runtime errors such as "RuntimeError: one of the variables needed for gradient computing has been modified by an in-place operation" occurred during training using DDP.
>
> When training with a single GPU before, the same error came out, so I knew that it was roughly a graph problem of pytorch's autograd, and I took action on the expected variables such as clone and detach, but it was not solved.
>
> The pytorch group community is actively active in this regard, and it was solved by referring to it.

### Solution from Offical documentation and community

* [RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: ... #23](https://github.com/NVlabs/FUNIT/issues/23#issuecomment-997770973)


### Solution

나의 경우는 ReLU와 같은 **inplace=True** 때문에 발생하는 문제는 아니고, DDP의 버퍼문제라고 말해주고 있다.

정확히 어떤 구조인지 밝혀지지는 않았지만 **DistributedDataParallel(...,broadcast_buffers=False,... )**매개변수를 추가해주면 문제가 해결된다.

> In my case, it is not caused by **inplace=True** such as ReLU, but it is said that it is a buffer problem of DDP.
>
> Although it is not known exactly what structure it is, adding **DistributedDataParallel(...,broadcast_buffers=False,...)** parameters solves the problem.