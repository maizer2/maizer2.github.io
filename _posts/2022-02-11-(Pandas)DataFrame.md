---
layout: post
title: "Pandas DataFrame"
categories: "용어_Python"
tags:  [Python, Pandas]
---

### DataFrame 객체[[1](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)]

**기본 pandas 데이터 구조.**

* 2차원, 크기 변경 가능, potentially heterogeneous tabular data (테이블 형식 데이터).
  * potentially heterogeneous → 여러 다른 종류들(heterogeneous)로 이루어질 수 있는(가능성있는, potentially)
* 데이터 구조에는 레이블이 지정된 축(행 및 열)도 포함된다..
* 산술 연산은 행 레이블과 열 레이블 모두에 정렬된다.
* Series 객체를 위한 dict-like 컨테이너로 생각할 수 있다.<br/><br/>

  
#### 매개변수

pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)

---

### DataFrame Method

---

#### DataFrame.head()

처음부터 n개 까지의 행을 반환한다.


##### 매개변수

DataFrame.head(n)

n : int, default 5

##### 반환 값

처음부터 n개 까지의 행을 반환.

---

#### DataFrame.info()

DataFrame의 간략한 요약을 출력한다.

---

##### 참고문언

1. pandas API reference, ver 1.4.0, [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)