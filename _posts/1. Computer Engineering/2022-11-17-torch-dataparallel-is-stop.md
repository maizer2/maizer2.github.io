---
layout: post
title: "nn.DataParallel gets stuck"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.2.2. Deep Learning, a.a. Pytorch]
---

### Problem

Multi GPU를 사용하기 위해 파이토치 nn.DataParallel()을 사용하였지만, 명시한 GPU가 100% 풀 로드하고 있음에도 Optimizer.step()에서 다음으로 넘어가지 않는 현상이 발생하였다. 최근까지 문제를 해결하지 못했지만, pytorch discuss에 비슷한 사례와 해결책을 찾을 수 있었다.

> Although i used Pytorch nn.DataParallel() for use Multi GPU, Even though the specified GPU is fully loaded at 100%, a phenomenon occurred where Optimizer.step() did not proceed to the next step. It didn't solve the problem until recently, but i was able to find similar examples and solutions on pytorch discuss.

### Solution from Offical documentation and community

* [nn.DataParallel gets stuck](https://discuss.pytorch.org/t/nn-dataparallel-gets-stuck/125427/1)

* [IOMMU Advisory for Multi-GPU Environments](https://docs.amd.com/bundle/IOMMU-Advisory-for-Multi-GPU-Environments/page/IOMMU_Advisory_for_Multi-GPU_Environments.html)

### Solution

AMD 공식문서에 따르면, AMD CPU의 IOMMU 기능이 멀티 GPU 사용을 실패하게 한다.
    
* centOS의 경우 [IOMMU Advisory for Multi-GPU Environments](https://docs.amd.com/bundle/IOMMU-Advisory-for-Multi-GPU-Environments/page/IOMMU_Advisory_for_Multi-GPU_Environments.html) 을 참고한다.

* Ubuntu의 경우 [nn.DataParallel gets stuck](https://discuss.pytorch.org/t/nn-dataparallel-gets-stuck/125427/1)에서 제안된 다음 글을 따른다. **"I modified this in my BIOS. Will be different depending on your system/MoBo maybe. But in my case I have an ASRock TRX40 Creator, hit delete or F11 at startup and went to Advanced → AMD CBS → NBIO → there should be an Option for IOMMU, which I set to disabled."**

> According to AMD offical documentation, AMD CPU's IOMMU function will fail to use multiple GPUs.
> 
> * Case of centOS, refer to [IOMMU Advisory for Multi-GPU Environments](https://docs.amd.com/bundle/IOMMU-Advisory-for-Multi-GPU-Environments/page/IOMMU_Advisory_for_Multi-GPU_Environments.html)
> 
> * For Ubuntu, Follow the article suggested by [nn.DataParallel gets stuck](https://discuss.pytorch.org/t/nn-dataparallel-gets-stuck/125427/1). **"I modified this in my BIOS. Will be different depending on your system/MoBo maybe. But in my case I have an ASRock TRX40 Creator, hit delete or F11 at startup and went to Advanced → AMD CBS → NBIO → there should be an Option for IOMMU, which I set to disabled."**