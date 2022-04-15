---
layout: post
title: "(Nvidia-Jetson) 2. Can't Flash jetson os on nx module"
categories: "연구_인공지능"
tags: [Nvidia jetson]
---

### 주저리

내나이 26살, 늘 개고생하고 느끼는 점 : 하라는대로 하면 해결된다.

### 구매처에서 나눴던 얘기

안녕하세요 기술지원 담당자 입니다.

리눅스 OS 가 설치된 PC에서 NVIDIA SDK Manager 를 이용해서 진행을 하셔야 한다고 답변 드렸었는데 

진행을 해보셨을까요? 

일반 윈도우 환경에서 가상 머신으로 OS를 구동 하시지 마시고 멀티부팅으로 OS를 설치하신 후 진행 해보시길 바랍니다.

가상 머신으로 구동 시 usb가 제대로 인식이 안되거나, 인식이 되어도 끊어지는 현상이 발생 합니다. 

아래 링크를 열어서 보시길 바랍니다.

https://wiki.seeedstudio.com/Jetson-Mate/



>Software Set Up. If you are using modules with micro-SD card from Developer Kit, we suggest you install and configure the system by following this guide for Jetson Nano, this guide for Jetson Nano 2GB and this guide for Jetson Xavier NX. If you are using modules with eMMC storage, please use NVIDIA’s official SDK Manager and follow the steps below

eMMC가 내장된 모듈을 사용한다면 NVIDIA’s official SDK Manager 를 사용해서 진행 하라는 내용이 있습니다.

NVIDIA SDK Manager 는 리눅스 환경에서만 실행이 가능합니다.

리눅스 OS 가 설치된 PC에서 아래 스탭을 진행 하시면 됩니다.

중간에 하드웨어 선택하는 단계에서는 NX 모듈을 선택 하시면 됩니다.

최종 설치 단계 까지 이르게 되면 시간이 많이 오래 걸립니다.  이점 참고 바랍니다.

디바이스마트에서는 전문적 기술지원은 불가능합니다. Seeed studio 나 Jetson 포럼에 문의 하셔서 기술 지원을 받으셔야 합니다.

이점 양해 바랍니다.

감사합니다.

<\b>

### 정답은 리눅스에서 Jetson을 설치하는것이였다.

저도 압니다. 리눅스에서 해야하는거, 근데 왜 안했냐구요?

일단 회사 컴퓨터에 리눅스 설치할 여유 컴퓨터가 없었고 그렇다고 윈도우 밀고 설치하고싶지도 않았습니다.

그래서 Virtual Box를 통해 설치했던건데, 대부분 후기 보면 다 성공했다고 올라왔었다구요 ㅠㅠ

그리고 이전에 Jetson Xavier AGX Dev Kit 설치할 때도 VB써서 설치했는데.. ?

결국 64GB USB에 리눅스 OS 설치해서 깔았더니 잘 깔렸다고한다. 제길

Nx Module OS 설치하고 Nano에도 초기화 하고 OS 재설치 하였다.

### 하지만

eMMC 16GB로는 할 수 있는게 없다..

외장하드 하나 달아야할거 같은데 ... 다시 시작 ...