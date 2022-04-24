---
layout: post
title: "Python Dictionary"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python]
---

### [Dictionary(dict), Map](http://pythonstudy.xyz/python/article/14-컬렉션--Dictionary)

Dictionary는 "키(Key) - 값(Value)" 쌍을 요소로 갖는 컬렉션,  해시테이블(Hash Table) 구조이다.

파이썬에서 Dictionary는 "dict" 클래스로 구현되어 있다. 

Dictionary의 키(key)는 그 값을 변경할 수 없는 Immutable 타입이어야 하며, Dictionary 값(value)은 Immutable과 Mutable 모두 가능하다. 예를 들어, Dictionary의 키(key)로 문자열이나 Tuple은 사용될 수 있는 반면, 리스트는 키로 사용될 수 없다.



#### items() 메서드

dict의 items()는 Dictonary의 키-값 쌍 Tuple 들로 구성된 dict_items 객체를 리턴한다.


---

##### 참고문헌

1. "컬렉션:Dictionary" 예제로 배우는 파이썬 프로그래밍. http://pythonstudy.xyz/python/article/14-컬렉션--Dictionary
