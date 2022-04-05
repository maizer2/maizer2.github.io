---
layout: post
title: "Docker Jupyter notebook 외부접속"
categories: "Docker"
tags: [Ubuntu]
---

윈도우 Wsl2 내부에 Docker 설치<sup><a href="https://maizer2.github.io/docker/2022/04/04/Windows-Wsl2-docker-내부에-pytorch,-cuda-설치하기.html">Link</a></sup>

이번에는 Docker Anaconda에 Jupyter를 설치하고 브라우저를 통해 접속하는 방법을 공부해보았다.

### jupyter 설치

```ubuntu
conda activate pytorch
conda install jupyter
```

### jupyter 설치확인

```ubuntu
jupyter --version
```

버전이 출력되면 정상설치 된 것이다.

### docker 내부 Ip확인

```ubuntu
apt-get install net-tools
ifconfig
```

```ubuntu
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.17.0.2  netmask 255.255.0.0  broadcast 172.17.255.255
        ether 02:42:ac:11:00:02  txqueuelen 0  (Ethernet)
        RX packets 24665  bytes 36109866 (36.1 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 8094  bytes 2490631 (2.4 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 0  bytes 0 (0.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```
위와 같이 나오면 **net-tools**가 정상설치 된 것이다.

eth0의 ip, inet의 172.17.0.2(다를 수 있다.) IP를 기억한다.

### jupyter notebook 실행 및 포트 개방

```ubuntu
jupyter notebook --allow-root --ip 172.17.0.2
```

```ubuntu
[I 14:51:32.471 NotebookApp] Serving notebooks from local directory: /root/1. git/1. python/First-step-on-the-Pytorch
[I 14:51:32.471 NotebookApp] Jupyter Notebook 6.4.3 is running at:
[I 14:51:32.471 NotebookApp] http://172.17.0.2:8888/?token=2ade87864e31eebbe3b0ad103227d7550e015ea4acb34e52
[I 14:51:32.471 NotebookApp]  or http://127.0.0.1:8888/?token=2ade87864e31eebbe3b0ad103227d7550e015ea4acb34e52
[I 14:51:32.471 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 14:51:32.474 NotebookApp] No web browser found: could not locate runnable browser.
[C 14:51:32.474 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-1437-open.html
    Or copy and paste one of these URLs:
        http://172.17.0.2:8888/?token=2ade87864e31eebbe3b0ad103227d7550e015ea4acb34e52
     or http://127.0.0.1:8888/?token=2ade87864e31eebbe3b0ad103227d7550e015ea4acb34e52
```

위와같이 나오면 Jupyter notebook이 정상 실행된 것이다.

이제 wsl2 쉘이 아닌 윈도우 브라우저에 위 URL을 입력해보자

접속되면 성공한 것이다.