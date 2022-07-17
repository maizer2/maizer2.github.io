---
layout: post
title: "CuDNN 설치 후 pytorch GPU사용 안됨"
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker]
---

### 사건의 발생

어제 CuDNN 정상 설치 확인 후 docker container의 pytorch에서 gpu사용이 안되는 것을 확인했다.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Using cpu device
```

미쳐버리겠지만 빨리 해결해야한다.

### 문제 파악

CuDNN 설치 후 gpu사용이 안되는 것이라 추측을 가지고 있기에 다음을 확인해본다.

1. Nvidia driver 정상 확인
2. CUDA ToolKit 정상 확인
3. CuDNN 정상 확인
4. Docker 문제 확인
    * Container 내, 외부 연동되는지 확인

맘같아선 다 지우고 다시깔고싶은데 일단 해본다.

### Docker 확인

우선 4번 문제를 확인하기 위해 container를 하나더 만들어서 --gpus all을 해보도록 한다.

...

새로운 container에 설치 중 이상한걸 찾았는데

호스트 서버에 11.7 cudaToolkit을 설치했는데 container에는 11.6이 설치된것이다.

일단 설치 중이니 정상적으로 설치되는지 확인 해보겠다.

...

```
Python 3.8.13 (default, Mar 28 2022, 11:38:47)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.cuda.is_available())
False
```

안되는거 확인했으나 컨테이너 문제인것 같다는 의심이 안지워져서 Nvidia condatiner를 사용하여 설치해보도록 한다.

[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

...

설치 하였지만 여전히 안되는 것을 확인했다.

### CuDNN 확인

각종 블로거들이 다음 커멘드를 입력하면 어떤 출력을 한다는데 나는 안된다..

```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

아무 출력도 없는데... 뭘까

```cudnn.h

#include <cuda_runtime.h>
#include <stdint.h>

#include "cudnn_version.h"
#include "cudnn_ops_infer.h"
#include "cudnn_ops_train.h"
#include "cudnn_adv_infer.h"
#include "cudnn_adv_train.h"
#include "cudnn_cnn_infer.h"
#include "cudnn_cnn_train.h"

#include "cudnn_backend.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
}
#endif

#endif /* CUDNN_H_ */
```

파일을 열었더니 전부 주석처리 돼 있는데... 맞는건가?

머리가 터질거 같다.. 그냥 초기화 하고싶다.