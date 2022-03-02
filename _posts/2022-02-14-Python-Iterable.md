---
layout: post
title: "Python Iterable"
categories: "용어_Python"
tags:  [Python]
---

### Python 공문에서 Iterable 이란?<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>
> An object capable of returning its members one at a time. Examples of iterables include all sequence types (such as list, str, and tuple) and some non-sequence types like dict, file objects, and objects of any classes you define with an __iter__() method or with a __getitem__() method that implements Sequence semantics.  
Iterables can be used in a for loop and in many other places where a sequence is needed (zip(), map(), …). When an iterable object is passed as an argument to the built-in function iter(), it returns an iterator for the object. This iterator is good for one pass over the set of values. When using iterables, it is usually not necessary to call iter() or deal with iterator objects yourself. The for statement does that automatically for you, creating a temporary unnamed variable to hold the iterator for the duration of the loop. See also iterator, sequence, and generator.
>> 멤버들을 한 번에 하나씩 돌려줄 수 있는 객체. 이터러블의 예로는 모든 (list, str, tuple 같은) 시퀀스 형들, dict 같은 몇몇 비 시퀀스 형들, 파일 객체들, __iter__() 나 시퀀스 개념을 구현하는 __getitem__() 메서드를 써서 정의한 모든 클래스의 객체들이 있습니다.  
이터러블은 for 루프에 사용될 수 있고, 시퀀스를 필요로 하는 다른 많은 곳 (zip(), map(), …) 에 사용될 수 있습니다. 이터러블 객체가 내장 함수 iter() 에 인자로 전달되면, 그 객체의 이터레이터를 돌려줍니다. 이 이터레이터는 값들의 집합을 한 번 거치는 동안 유효합니다. 이터러블을 사용할 때, 보통은 iter() 를 호출하거나, 이터레이터 객체를 직접 다룰 필요는 없습니다. for 문은 이것들을 여러분을 대신해서 자동으로 해주는데, 루프를 도는 동안 이터레이터를 잡아둘 이름 없는 변수를 만듭니다. 이터레이터, 시퀀스, 제너레이터 도 보세요.

### 요약

파이썬을 사용하다보면 for문에 인자로 내장 컨테이너 혹은 collection을 전달하게 되면, 그 객체의 이터레이터를 반환하며 멤버들을 한 번에 하나씩 돌려주게 됩니다.

---

##### 참고 문헌

<a href="#footnote_1_2" name="footnote_1_1">1.</a>  Python glossary, Python, ver 3.10.2, [https://docs.python.org/ko/3/glossary.html#term-iterable](https://docs.python.org/ko/3/glossary.html#term-iterable)