---
layout: post
title: "Pandas Series"
categories: "용어_Python"
tags:  [Python, Pandas]
---

### Series 객체[[1](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)]

**기본 1차원 pandas 데이터 구조.**

축 레이블이 있는 1차원 ndarray 객체이며 

  
#### 매개변수

pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)

---

### DataFrame Method

---

#### DataFrame.head() [[2](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html)]

처음부터 n개 까지의 행을 반환한다.


##### 매개변수

DataFrame.head(n)

n : int, default 5

##### 반환 값

처음부터 n개 까지의 행을 반환.

---

#### DataFrame.info() [[3](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html)]

DataFrame의 간략한 요약을 출력한다.

---

#### DataFrame.value_counts() [[4](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html)]

DataFrame의 고유한 행 수를 포함하는 Series를 반환합니다.  

Nan 값을 제외한(매게변수 설정으로 Nan값 포함 가능) 카테고리의 개수를 출력한다.

#### DataFrame.describe() [[5](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)]

숫자형 특성의 요약정보를 보여준다.

Series의 인덱스는 count, mean, std, min, max, select_dtypes, 25%, 50%, 75% 를 표시한다.

---

##### 참고문언

1. DataFrame, pandas API reference, ver 1.4.0, [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

2. DataFrame.head(), pandas API reference, ver 1.4.0, [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html)

3. DataFrame.info(), pandas API reference, ver 1.4.0, [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html)

4. DataFrame.value_counts(), pandas API reference, ver 1.4.0 [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html)

5. DataFrame.describe(), pandas API reference, ver 1.4.0 [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)