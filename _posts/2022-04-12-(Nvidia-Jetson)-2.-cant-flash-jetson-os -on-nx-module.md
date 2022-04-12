---
layout: post
title: "(Nvidia-Jetson) 2. Can't Flash jetson os on nx module"
categories: "연구_인공지능"
tags: [Nvidia jetson]
---

### 주저리

jetson은 지옥이다.<br/>

### 현재 발생하는 문제점

안된다. flash가 안먹힌다. xavier agx dev kit에서 고생고생하면서 설치 했었는데 nx module은 os 설치부터가 안된다.

보드 자체에 불이 안들어오는거 보니 모듈 문제인가? 전원은 들어가는거 같은데 전혀 반응이 없어서 머리아프다.

시간도 없는데 시험 끝나고 만져야겠다.

### 얻은점

eMMC<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>는 Embedded Multi Media Card로서 MMC인 Flash Memory인데 Flash Memory는 비휘발성 반도체 저장장치로서 전기적으로 자유롭게 재기록이 가능하다는 장점이 있다.

Embedded<sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup>란 기계나 기타 제어가 필요한 시스템에 대해, 제어를 위한 특정 기능을 수행하는 컴퓨터 시스템으로 장치 내에 존재하는 전자 시스템이다.

Jetson 모듈 중 Xavier에 해당하는 제품들은 eMMC가 내장되어 있다. 내장 저장공간이 있어 sd카드가 없어도 된다. 하지만 Nx Module은 20gb로서 저용량이라 OS 설치 이외에 설치는 버거운 용량이다.

---

##### 참고문헌

<a href="#footnote_1_2" name="footnote_1_1">1.</a> eMMC, 나무위키, [https://namu.wiki/w/eMMC](https://namu.wiki/w/eMMC)

<a href="#footnote_2_2" name="footnote_2_1">2.</a> 임베디드 시스템, 위키백과, [https://ko.wikipedia.org/wiki/%EC%9E%84%EB%B2%A0%EB%94%94%EB%93%9C_%EC%8B%9C%EC%8A%A4%ED%85%9C](https://ko.wikipedia.org/wiki/%EC%9E%84%EB%B2%A0%EB%94%94%EB%93%9C_%EC%8B%9C%EC%8A%A4%ED%85%9C)