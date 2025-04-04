---
layout: post
title: "Windows Wsl2 docker 내부에 pytorch, cuda 설치하기"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker, a.a. Pytorch, 1.7. Network]
---

이글은 Tistory, 박핑구님의 블로그<sup><a href="https://pinggoopark.tistory.com/m/117">Link</a></sup>를 참고하였습니다.

|OS|CPU|GPU|ubuntu|docker|
|---|---|---|---|---|
|Windows 10 Pro|AMD Ryzen 5 3600 6-Core Processor|Nvidia Geforce RTX 2070 Super|18.04|20.10.13, build a224086|

### GPU Driver 설치

[https://www.nvidia.co.kr/Download/index.aspx?lang=kr](https://www.nvidia.co.kr/Download/index.aspx?lang=kr)

본인의 GPU에 맞는 드라이버를 설치한다.<br/>

### Docker Desktop 설치

[https://www.docker.com/get-started/](https://www.docker.com/get-started/)

windows 용으로 설치한다.

### Docker Desktop Setting

Wsl2을 사용할 수 있도록 설정한다.

Docker Desktop -> Setting -> Resources -> WSL INTEGRATION

체크 항목
* Enable integration with my default WSL distro
* Ubuntu 

Apply & Restart<br/>

### wsl2 Ubuntu GPU Driver 설치 확인

Ubuntu로 들어와 **nvidia-smi** 명령어를 실행한다.

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.60.02    Driver Version: 512.15       CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:26:00.0  On |                  N/A |
|  0%   44C    P5    21W / 215W |    859MiB /  8192MiB |     22%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

다음과 같은 표가 나오면 정상적으로 GPU Driver가 설치 된 것이다.

2022-04-04 기준 CUDA Version이 11.6이 아닌경우 드라이버 확인이 필요하다.<br/>

### WSL2 Ubuntu에 CUDA 설치

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

### Docker GPU Test

```코드실행
sudo docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

```정상출력
Run "nbody -benchmark [-numbodies=<numBodies>]" to measure performance.
        -fullscreen       (run n-body simulation in fullscreen mode)
        -fp64             (use double precision floating point values for simulation)
        -hostmem          (stores simulation data in host memory)
        -benchmark        (run benchmark to measure performance)
        -numbodies=<N>    (number of bodies (>= 1) to run in simulation)
        -device=<d>       (where d=0,1,2.... for the CUDA device to use)
        -numdevices=<i>   (where i=(number of CUDA devices > 0) to use for simulation)
        -compare          (compares simulation results running once on the default GPU and once on the CPU)
        -cpu              (run n-body simulation on the CPU)
        -tipsy=<file.bin> (load a tipsy model file for simulation)

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

> Windowed mode
> Simulation data stored in video memory
> Single precision floating point simulation
> 1 Devices used for simulation
GPU Device 0: "Turing" with compute capability 7.5

> Compute 7.5 CUDA device: [NVIDIA GeForce RTX 2070 SUPER]
40960 bodies, total time for 10 iterations: 70.371 ms
= 238.410 billion interactions per second
= 4768.196 single-precision GFLOP/s at 20 flops per interaction
```

**에러코드 발생시**

에러를 잘 읽어보면 해결방법이 나온다.  
나같은 경우는 CUDA Version이 맞지 않아 드라이버를 업데이트 하였다.

### Docker Ubuntu 18.04 Container 설치<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

```ubuntu
# 포트포워딩 개념이 부족한 사람은 [오늘도 야근, 도커(Docker) : 포트 포워딩 설정(포트 맵핑)하기](https://tttsss77.tistory.com/155)를 참고하세요.
sudo docker run -i -t --gpus all -p 외부포트:내부포트 ubuntu:18.04    #외부, 내부포트는 현재 사용하지 않는 포트를 지정해 열어주면 된다, 8888은 해킹당하기 쉬우니까 외부 포트 개방할 때 조심
```

ubuntu 18.04 image 설치 및 컨테이너 내부로 접속된다.

*추후 container 이름 설정 및 포트 설정을 통해 외부 접속이 가능하게 할 수 있다.*

### Container GPU 확인

```ubuntu
nvidia-smi
```

### Anaconda 설치

[https://repo.anaconda.com/archive/](https://repo.anaconda.com/archive/)

위 공식 아카이브에서 Linux 설치

모르겠으면 아래 복붙 (2022-04-05기준 콘다 최신버전)

```ubuntu
cd /tmp
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
sh Anaconda3-5.3.1-Linux-x86_64.sh
rm Anaconda3-5.3.1-Linux-x86_64.sh
source ~/.bashrc
```

이후 쉘에 (base)가 보이면 완료

### conda 설정

conda env는 개인 취향에 맞게 설정하면 됩니다. 

우선 제 기준으로 설정하겠습니다.

```ubuntu
conda create --name pytorch python=3.6
conda activate pytorch
```

이후 (base) -> (pytorch) 가 됨을 확인할 수 있습니다.

### Pytorch 설치

[https://pytorch.kr/get-started/locally/](https://pytorch.kr/get-started/locally/)

위 링크에 들어가 사양에 맞게 선택 후 명령어를 실행하면 됩니다.

Stable(1.11.0) -> Windows -> Conda -> Python -> CUDA 11.3

***주의***  
conda env를 확인하고 설치하십시오

```ubuntu
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

---
##### 참고문헌

<a href="#footnote_1_2" name="footnote_1_1">1.</a> GPU, Docker docs, [https://docs.docker.com/config/containers/resource_constraints/#gpu](https://docs.docker.com/config/containers/resource_constraints/#gpu)