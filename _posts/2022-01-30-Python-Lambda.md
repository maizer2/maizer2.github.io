---
layout: post
title: "Python Lambda(람다)"
categories: [1. Computer Engineering]
tags: [1.1. Programming, 1.1.1. Python]
---

### 람다(lambda)

#### 람다 정의

"lambda" 는 런타임에 생성해서 사용할 수 있는 익명 함수 입니다. 이것은 함수형 프로그래밍 언어에서 lambda와 정확히 똑같은 것은 아니지만, 파이썬에 잘 통합되어 있으며 filter(), map(), reduce()와  같은 전형적인 기능 개념과 함께 사용되는 매우 강력한 개념입니다. 

lambda는 쓰고 버리는 일시적인 함수 입니다. 함수가 생성된 곳에서만 필요합니다. 즉, 간단한 기능을 일반적인 함수와 같이 정의해두고 쓰는 것이 아니고 필요한 곳에서 즉시 사용하고 버릴 수 있습니다. [[2](https://offbyone.tistory.com/73)]

#### 람다 형식

```Python
lambda 매개변수 : 표현식
```

##### 람다 예제 [[3](https://wikidocs.net/64)]

일반 함수 작성

```Python
>>> def hap(x, y):
...   return x + y
...
>>> hap(10, 20)
30
```

람다 함수 적용

```Python
>>> (lambda x,y: x + y)(10, 20)
30
```


---

##### 참고문헌

1. "파이썬 문법 5 - 람다(lambda) 함수" OffByOne. 쉬고 싶은 개발자. https://offbyone.tistory.com/73

2. "3.5 람다(lambda)" 왕초보를 위한 Python: 쉽게 풀어 쓴 기초 문법과 실습. https://wikidocs.net/64
