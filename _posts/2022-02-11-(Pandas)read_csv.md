---
layout: post
title: "Pandas read_csv"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python, Pandas]
---

### read_csv 객체[[1](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)]

쉼표로 구분된 값(csv) 파일을 [DataFrame](https://maizer2.github.io/용어_python/2022/02/11/(Pandas)DataFrame.html) 으로 읽어온다.

또한 선택적으로 파일을 청크로 반복하거나 분할하는 것을 지원한다.

#### Parameters

##### filepath_or_buffer

문자열, path 객체, file과 같은 객체

모든 유효한 문자열 주소는 가능하다. URL(http, ftp, gs, 로컬 file 등등)

경로 객체를 전달하는 경우, 모든 [os.PathLike](https://maizer2.github.io/용어_python/2022/02/11/Python-os.html) 를 허용한다.

---

##### 참고문언

1. pandas API reference, ver 1.4.0, [https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)