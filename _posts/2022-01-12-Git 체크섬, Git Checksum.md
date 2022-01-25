---
layout: post
title: "Git 체크섬, Git Checksum"
categories: "용어"
tags: [보안, Git]
---

DVCS(분산 버전 관리 시스템), Git을 공부하던 중 눈에 익숙한 용어를 읽고 넘어갔다.

Checksum(체크섬)이라는 용어는 참 쉽게 볼 수 있지만 정확한 정의를 모르는 것 같아 따로 정리해두려고 한다.

### Git에서 Checksum이란?

SHA-1 해시를 사용하여 40자, 16진수 문자열로 만들어진 해쉬값이다.

Git은 파일의 내용이나 디렉터리 구조를 이용해 체크섬을 구한다.

```Checksum
24b9da6552252987aa493b52f8696cd6d3b00373
```

Git은 해시로 식별한다. 따라서 Git은 파일을 해당 파일의 해시로 저장한다.


### CRC(cyclic redundancy check), 순환 중복 검사

네트워크 용어로서 통신 오류 검증 방법이다. TCP/IP, MAC 등의 계층에서 오류검증을 목적으로 사용한다.

정해진 다항식이 결정되어 있고, 이것에 따라 송신 쪽에서 계산하여 헤더에 붙여 보내면 수신 쪽에서 다시 계산하고 보내진 체크섬과 비교한다.

경우에 따라 하드웨어 또는 소프트웨어 방법으로 계산한다. 상황에 따라 개발자가 설계 결정하고 구현한다. 

보통 TCP/IP등은 소프트웨어적인 방법이 대부분이고, 밑의 계층으로 갈수록 하드웨어에 의존하는 경향이 있다.


---

##### 참고문헌

1. "1.3 시작하기 - Git 기초" Git-scm. https://git-scm.com/book/ko/v2/%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0-Git-%EA%B8%B0%EC%B4%88

2. "체크섬" ko.wikipedia. https://ko.wikipedia.org/wiki/%EC%B2%B4%ED%81%AC%EC%84%AC
