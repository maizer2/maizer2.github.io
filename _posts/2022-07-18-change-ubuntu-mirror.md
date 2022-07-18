---
layout: post
title: "Ubuntu mirror server change to kakao"
categories: [1. Computer Engineering]
tags: [1.4. OS, 1.4.1. Linux]
---

```ubuntu-server
sudo vi /etc/apt/sources.list

# Change every context to next line

# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.
deb http://mirror.kakao.com/ubuntu focal main restricted
# deb-src http://mirror.kakao.com/ubuntu focal main restricted

## Major bug fix updates produced after the final release of the
## distribution.
deb http://mirror.kakao.com/ubuntu focal-updates main restricted
# deb-src http://mirror.kakao.com/ubuntu focal-updates main restricted

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team. Also, please note that software in universe WILL NOT receive any
## review or updates from the Ubuntu security team.
deb http://mirror.kakao.com/ubuntu focal universe
# deb-src http://mirror.kakao.com/ubuntu focal universe
deb http://mirror.kakao.com/ubuntu focal-updates universe
# deb-src http://mirror.kakao.com/ubuntu focal-updates universe

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team, and may not be under a free licence. Please satisfy yourself as to
## your rights to use the software. Also, please note that software in
## multiverse WILL NOT receive any review or updates from the Ubuntu
## security team.
deb http://mirror.kakao.com/ubuntu focal multiverse
# deb-src http://mirror.kakao.com/ubuntu focal multiverse
deb http://mirror.kakao.com/ubuntu focal-updates multiverse
# deb-src http://mirror.kakao.com/ubuntu focal-updates multiverse

## N.B. software from this repository may not have been tested as
```
---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 까먹으면 적어두자, whiteglass, Write:2021.03.25, attach:2022.07.18 방문, [https://whiteglass.tistory.com/12](https://whiteglass.tistory.com/12)