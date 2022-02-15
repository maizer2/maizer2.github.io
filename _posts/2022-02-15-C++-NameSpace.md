---
layout: post
title: Python NameSpace(네임스페이스)
categories: "용어_Python"
tags: [Python, C++]
---

### NameSpace란?

#### wikipedia<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

이름공간 또는 네임스페이스(영어: namespace)는 개체를 구분할 수 있는 범위를 나타내는 말로 일반적으로 하나의 이름 공간에서는 하나의 이름이 단 하나의 개체만을 가리키게 된다.

* 파일 시스템은 파일에 이름을 할당하는 이름공간이다.
* 일부 프로그래밍 언어들은 이름공간 안에 변수와 함수를 조직한다. 컴퓨터 프로그래밍 언어인 C에서는 전역 지역 공간과 지역 이름 공간라는 이름 공간에 대한 개념이 있는데, 각각의 이름 공간에서는 같은 변수나 함수 이름을 사용할 수 없지만, 영역이 다르면 변수나 함수명이 같을 수도 있다. C++와 Java 프로그래밍 언어에서는 이름 공간을 명시적으로 지정하여 사용할 수 있다.
* 컴퓨터 네트워크와 분산 시스템은 이름을 컴퓨터, 프린터, 웹사이트, (원격) 파일 등의 자원에 할당한다.

```C++
#include <iostream>
using std::cout;
using std::endl;

namespace Box1{
   int boxSide = 4;
}

namespace Box2{
   int boxSide = 12;
}

int main () {
  cout << Box1::boxSide << endl;  //output 4
  cout << Box2::boxSide << endl;  //output 12
  return 0;
}
```

---

### Python 에서 네임스페이스<sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup>

네임스페이스는 전역, 지역, 빌트-인 3가지로 분류된다.

![namespace by python](https://hcnoh.github.io/assets/img/2019-01-30-python-namespace/02.jpg)

<div style="text-align: center; font-weight: bold; font-style: italic"> 출처 : A diagram of different namespaces in Python<sup><a href="#footnote_3_1" name="footnote_3_2">[3]</a></sup></div>

> * 빌트-인 네임스페이스: 기본 내장 함수 및 기본 예외들의 이름들이 소속된다. 파이썬으로 작성된 모든 코드 범위가 포함된다.
> * 전역 네임스페이스: 모듈별로 존재하며, 모듈 전체에서 통용될 수 있는 이름들이 소속된다.
> * 지역 네임스페이스: 함수 및 메서드 별로 존재하며, 함수 내의 지역 변수들의 이름들이 소속된다.

#### Python namespace의 특징

> * 네임스페이스는 딕셔너리 형태로 구현된다.
> * 모든 이름 자체는 문자열로 되어있고 각각은 해당 네임스페이스의 범위에서 실제 객체를 가리킨다.
> * 이름과 실제 객체 사이의 매핑은 가변적(Mutable)이므로 런타임동안 새로운 이름이 추가될 수 있다.
> * 다만, 빌트인 네임스페이스는 함부로 추가하거나 삭제할 수 없다.

---

##### 참고 문헌

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 이름공간, wikipedia, [https://ko.wikipedia.org/wiki/이름공간](https://ko.wikipedia.org/wiki/이름공간)

<a href="#footnote_2_2" name="footnote_2_1">2.</a> [Python] 네임스페이스 개념 정리, Hyungcheol Noh, [https://hcnoh.github.io/2019-01-30-python-namespace](https://hcnoh.github.io/2019-01-30-python-namespace)

<a href="#footnote_3_2" name="footnote_3_1">3.</a> Python Namespace and Scope, Programiz, [https://www.programiz.com/python-programming/namespace](https://www.programiz.com/python-programming/namespace)

