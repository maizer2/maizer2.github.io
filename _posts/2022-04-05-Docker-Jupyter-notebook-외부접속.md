---
layout: post
title: "Docker Jupyter notebook 외부접속"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, 1.2. Artificial Intelligence, 1.4. OS, 1.4.1. Linux, 1.5. Container, 1.5.1. Docker, a.a. Pytorch, 1.8. Network]
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

eth0의 ip 172.17.0.2 IP를 기억한다.

위 ip는 docker 내부의 아이피이다.

### Jupyter_notebook config 생성

config 파일은 jupyter의 설정 파일이다.

내부 포트 및 패스워드 설정시 필요한데, 이 파일은 명령어를 통해 생성할 수 있다.

```ubuntu
jupyter notebook --generate-config -y
>>>> Overwrite /root/.jupyter/jupyter_notebook_config.py with default config? y
>>>> Writing default config to: /root/.jupyter/jupyter_notebook_config.py
```

아래 위치에 config 파일이 생성되었다. 수정하기전에 우선 jupyter notebook에 접속할 때 입력할 비밀번호를 생성하겠다.

```ubuntu
ipython
```

```ipython
In [1] : from notebook.auth import passwd

In [2] : passwd()
>>>>> Enter password:
>>>>> Verify password:


Out[2] : '입력한 비밀번호 값 해쉬값으로 출력, 해쉬값 저장해야됨 복사해두세요'

In [3] : quit()
```

### Jupyter_notebook config 수정

```ubuntu
apt-get install vi

vi /root/.jupyter/jupyter_notebook_config.py
```

위 코드를 실행하면 내부 설정파일이 뜨는데 다음 설정을 수정해주자

혹시 Vim 사용이 익숙하지 않으신 분들에게 설명

vim은 단축키를 통해 입력 및 이동을 할 수 있습니다.

a를 누르면 입력 모드가 시작되고 esc를 누르면 입력 모드가 종료되며 마지막 행 모드가 됩니다.

마지막 행 모드, 입력 모드에서 / {찾으려는 단어} 엔터, 를 사용하여 단어를 찾을 수 있습니다.

 vi editer에서 저장 및 나가는 방법 및 단축키는

|모드|명령어|설명|
|---|---|---|
|명령어 모드|a|커서에서 한칸 띄워서 입력모드|
|명령어 모드|i|커서자리에서 입력모드|
|명령어 모드|h|왼쪽으로 커서 이동|
|명령어 모드|j|아래로 커서 이동|
|명령어 모드|k|위로 커서 이동|
|명령어 모드|l|오른쪽으로 커서 이동|
|명령어 모드|e|다음 단어로 커서 이동|
|명령어 모드|b|이전 단어로 커서 이동|
|명령어 모드|$, Shift + 4|오른쪽 끝으로 이동|
|명령어 모드|^, Shift + 6|왼쪽 끝으로 이동|
|명령어 모드|v|홀드함, 커서부터 홀드시켜 단어를 다수 선택 가능|
|명령어 모드|s|커서 문자 삭제하고 입력모드|
|명령어 모드|x|커서 문자 복사 후 삭제하고 입력모드|
|명령어 모드|y|복사|
|명령어 모드|p|붙여넣기|
|마지막 행 모드|:q|저장하지 않고 나가기, 수정사항이 있으면 나가지지 않음, ! 옵션을 줘서 강제 나가기 가능, 저장안됨|
|마지막 행 모드|:wq|저장하고 나가기|


```ubuntu
c.NotebookApp.ip='localhost'
c.NotebookApp.open_browser=False
c.NotebookApp.password='위에 저장했던 해쉬값을 넣어줌'
c.NotebookApp.password_required=True
c.NotebookApp.port=8888     #도커 컨테이너 생성시 개방했던 내부 포트를 입력해준다, 
c.NotebookApp.iopub_data_rate_limit=1.0e10  
c.NotebookApp.terminado_settings={'shell_command': ['/bin/bash']}  # terminal을 bash로 실행
```

### jupyter notebook 실행 및 포트 개방

```ubuntu
jupyter notebook --allow-root --ip 0.0.0.0
```


```ubuntu
(pytorch) root@ae7f12647952:~/1. dev# jupyter notebook --ip 0.0.0.0 --allow-root

[I 12:15:14.802 NotebookApp] Serving notebooks from local directory: 
[I 12:15:14.803 NotebookApp] Jupyter Notebook 6.4.3 is running at:
[I 12:15:14.803 NotebookApp] http://ae7f12647952:내부포트/
[I 12:15:14.803 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

### 외부에서 서버 도커 접속하기

외부 -> 호스트서버(우분투) -> 도커

호스트서버 Ip : 호스트서버 port

외부컴퓨터 웹 브라우저 검색창에 호스트 IP:Port 를 입력하면 주피터로 접속된다.

만약 접속이 안됐다면 네트워크 공부를 열심히 하자, 공유기, 모뎀등으로 네트워크 구조가 복잡하게 돼 있을 수 있으니 포트 구성을 열심히 해보자..

이거 한번 잘하면 나중에 무조껀 쓸곳있을거다.