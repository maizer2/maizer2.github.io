---
layout: post
title: "(Nvidia-Jetson) 1. Jetson mate on nx module"
categories: "연구_인공지능"
tags: [Nvidia jetson]
---

### 주저리

인공지능 예비 석박사들은 Jetson 하나 잘쓰면 든든하게 연구할 수 있는데, 국내에는 Jetson에 대한 설명이 너무 부족하다..

나같은 하알못, 하드웨어 알지 못하는 석사생은 눈물을 머금고 맨땅에 해딩할 뿐이다.<br/>

### Jetson mate 구성

|Master|WORKER1|WORKER2|WORKER3|
|------|------|------|------|
|Jetson Nano|Jetson Xavier Nx Module|x|x|

### 현재 발생하는 문제점

공부하며 알아낸 사실은 Jetson mate보드는 jetson 모듈들을 클러스터 시스템으로 구성시켜주는 보드이다.

>클러스터 시스템이란 N개 이상의 컴퓨터로 하나의 시스템처럼 동작하는 컴퓨터들의 집합을 뜻한다. 단일 컴퓨터보다 더 좋은 성능을 나타내어 저렴한 컴퓨터 시스템을 다수 구성하여 성능을 끌어올린다.

나는 이 보드를 하나의 컴퓨터 보드로 생각하고 구매를 하였다. 따라서 Nano 모듈하나를 Master에 장착하고 Nx 모듈을 통해 다중 GPU를 구성한다 생각하였다.

하지만 일반 GPU와 다르게 Jetson 모듈은 GPU가 아닐뿐더러 하나의 컴퓨터로봐야한다. cpu, gpu, ram이 한 보드에 들어있다.

따라서 Master 슬롯에 있는 Nano는 잘 실행되지만 내 생각대로였다면 Nano에서 GPU가 2개 인식 됐어야 한다.<sup>이는 논리적 오류입니다.</sup>

### 파생되는 궁금점

Jetson Xavier Nx Module은 Nano Develope Kit의 모듈과 달리 SD카드 슬롯이 없다.

Nx Develope Kit의 모듈에도 SD카드 슬롯이 있지만 단일 Module은 SD카드 슬롯이 없다.

그래서 나는 처음에 단일 GPU라고 착각을 했다.

개인이 SD카드 슬롯을 설치하기란 쉽지 않다고 생각되고 USB를 통해 OS를 연결하게 하면 똑같이 사용할 수 있지 않을까?

다른 사용자들이 USB를 통해 부팅에 성공한 사례를 확인하였고 다음시간에 설치해보기로 한다. 이번장은 공부했던 내용을 정리하도록 한다.

### Jetson mate란?

[Seeed](https://www.seeedstudio.com/about_seeed) 회사에서 Nvidia Jetson 하드웨어를 활용하기 위해 제작한 보드이다.

![Jetson Mate Board](https://files.seeedstudio.com/wiki/Jetson-Mate/banner-2.png)

> Jetson Mate is a carrier board which can install up to 4 Nvidia Jetson Nano/NX SoMs. There is an on board 5-port gigabit switch enabling the 4 SoMs to communicate with each other. All the 3 peripheral SoMs can be powered on or off separately. With a 65W 2-Port PD charger, for Jetson Nano SoMs or a 90W 2-Port PD charger for Jetson NX SoMs, and one ethernet cable, developers can easily build their own Jetson Cluster.<sup><a href="footnote_1_1" name="footnote_1_2">[1]</a></sup>
>> Jetson Mate는 최대 4개의 Nvidia Jetson Nano/NX SoM을 설치할 수 있는 캐리어 보드입니다. 4개의 SoM이 서로 통신할 수 있는 온보드 5포트 기가비트 스위치가 있습니다. 3개의 주변 장치 SoM은 모두 개별적으로 전원을 켜거나 끌 수 있습니다. 65W 2포트 PD 충전기, Jetson Nano SoMs용 또는 Jetson NX SoM용 90W 2포트 PD 충전기, 그리고 하나의 이더넷 케이블로 개발자는 자신의 Jetson 클러스터를 쉽게 구축할 수 있습니다.

단일 gpu를 사용하는 jetson Nano/NX Develope Kit과 달리 NX Module<sup><a href="footnote_2_1" name="footnote_2_2">[1]</a></sup>과 같이 모듈 형태의 GPU를 4개까지 연결하여 서버로 구성할 수 있는 보드이다.

### Jetson mate의 Fan 제어

mate는 크게 뚜껑 Fan, 각 모듈별 Fan 총 5개의 Fan이 존재한다.

가장 먼저 뚜껑에 해당되는 Fan을 제어하는 방법이다.

```ubuntu
sudo sh -c 'echo 100 > /sys/devices/pwm-fan/target_pwm'
```
0 ~ 255 사이의 값을 넣어주면 크기에 따라 회전력이 증가한다.



---
##### 참고문헌

<a href="footnote_1_2" name="footnote_1_1">1.</a> Jetson Mate, Seeed studio, [https://wiki.seeedstudio.com/Jetson-Mate/](https://wiki.seeedstudio.com/Jetson-Mate/)

<a href="footnote_2_2" name="footnote_2_1">2.</a> 모든 JETSON XAVIER NX 제품 둘러보기, nvidia,  [https://www.nvidia.com/ko-kr/autonomous-machines/jetson-store/](https://www.nvidia.com/ko-kr/autonomous-machines/jetson-store/)