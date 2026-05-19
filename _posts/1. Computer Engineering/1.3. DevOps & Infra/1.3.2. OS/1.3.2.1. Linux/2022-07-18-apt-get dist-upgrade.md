---
layout: post
categories: [1. Computer Engineering, 1.3. DevOps & Infra, 1.3.2. OS, 1.3.2.1. Linux]
title: "Ubuntu apt-get dist-upgrade"
tags: [apt-get, Dependency]
---

### apt-get update

Package **list** update, not upgrade Package.

### apt-get upgrade

Package list upgrade. not check dependency.

### apt-get dist-upgrade<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

Upgrade the package list to **suit the dependency**.

---

##### Reference

<a href="#footnote_1_2" name="footnote_1_1">1.</a> sudo apt-get update upgrade dist-upgrade 차이, wonjinho81, Write:2018.10.17, attach:2022.07.18 방문, [https://blog.naver.com/wonjinho81/221379267906](https://blog.naver.com/wonjinho81/221379267906)