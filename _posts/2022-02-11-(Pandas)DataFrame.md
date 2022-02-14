---
layout: post
title: "Pandas DataFrame"
categories: "용어_Python"
tags:  [Python, Pandas]
---

### DataFrame 객체 <sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

**기본 2차원 pandas 데이터 구조.**

* 2차원, 크기 변경 가능, potentially heterogeneous tabular data (테이블 형식 데이터).
  * potentially heterogeneous → 여러 다른 종류들(heterogeneous)로 이루어질 수 있는(가능성있는, potentially)
* 데이터 구조에는 레이블이 지정된 축(행 및 열)도 포함된다..
* 산술 연산은 행 레이블과 열 레이블 모두에 정렬된다.
* Series 객체를 위한 dict-like 컨테이너로 생각할 수 있다.<br/><br/>

  
#### 매개변수

pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)

---

### DataFrame Method


#### DataFrame.head() <sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup>

처음부터 n개 까지의 행을 반환한다.


##### 매개변수

DataFrame.head(n)

n : int, default 5

##### 반환 값

처음부터 n개 까지의 행을 반환.

#### DataFrame.info() <sup><a href="#footnote_3_1" name="footnote_3_2">[3]</a></sup>

DataFrame의 간략한 요약을 출력한다.

#### DataFrame.value_counts() <sup><a href="#footnote_4_1" name="footnote_4_2">[4]</a></sup>

DataFrame의 고유한 행 수를 포함하는 Series를 반환합니다.  

Nan 값을 제외한(매게변수 설정으로 Nan값 포함 가능) 카테고리의 개수를 출력한다.

#### DataFrame.describe() <sup><a href="#footnote_5_1" name="footnote_5_2">[5]</a></sup>

숫자형 특성의 요약정보를 보여준다.

Series의 인덱스는 count, mean, std, min, max, select_dtypes, 25%, 50%, 75% 를 표시한다.

---

##### 참고문언

<a href="#footnote_1_2" name="footnote_1_1">1.</a> DataFrame, pandas API reference, ver 1.4.0, [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

<a href="#footnote_2_2" name="footnote_2_1">2.</a> DataFrame.head(), pandas API reference, ver 1.4.0, [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html)

<a href="#footnote_3_2" name="footnote_3_1">3.</a> DataFrame.info(), pandas API reference, ver 1.4.0, [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html)

<a href="#footnote_4_2" name="footnote_4_1">4.</a> DataFrame.value_counts(), pandas API reference, ver 1.4.0 [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html)

<a href="#footnote_5_2" name="footnote_5_1">5.</a> DataFrame.describe(), pandas API reference, ver 1.4.0 [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)