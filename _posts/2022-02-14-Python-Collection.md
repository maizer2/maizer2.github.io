---
layout: post
title: Python Collection
categories: "용어_Python"
tags: [Python]
---

### Python Collection 이란?<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

파이썬 내장 컨테이너인 dict, list, set 및 tuple에 대한 대안을 제공하는 특수 컨테이너 데이터 형을 구현한다.

|객체|설명|
|------|---|
|namedtuple()|이름 붙은 필드를 갖는 튜플 서브 클래스를 만들기 위한 팩토리 함수|
|deque|양쪽 끝에서 빠르게 추가와 삭제를 할 수 있는 리스트류 컨테이너|
|ChainMap|여러 매핑의 단일 뷰를 만드는 딕셔너리류 클래스|
|Counter|해시 가능한 객체를 세는 데 사용하는 딕셔너리 서브 클래스|
|OrderedDict|항목이 추가된 순서를 기억하는 딕셔너리 서브 클래스|
|defaultdict|누락된 값을 제공하기 위해 팩토리 함수를 호출하는 딕셔너리 서브 클래스|
|UserDict|더 쉬운 딕셔너리 서브 클래싱을 위해 딕셔너리 객체를 감싸는 래퍼|
|UserList|더 쉬운 리스트 서브 클래싱을 위해 리스트 객체를 감싸는 래퍼|
|UserString|더 쉬운 문자열 서브 클래싱을 위해 문자열 객체를 감싸는 래퍼|


---

##### 참고 문헌

<a href="#footnote_1_2" name="footnote_1_1">1.</a> collections, Python, ver 3.10.2, [https://docs.python.org/ko/3/library/collections.html?highlight=collection#module-collections](https://docs.python.org/ko/3/library/collections.html?highlight=collection#module-collections)
