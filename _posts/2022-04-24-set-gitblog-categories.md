---
layout: post
title: "Set gitblog categories"
categories: "Git"
tags: [GitHub, Gitblog]
---

깃블로그를 처음 시작할 때 머리아팠던게 카테고리 설정이다.

점점 쓰는 글이 많아지다보니 카테고리랑 태그가 엄청나게 늘어났다.

처음에는 자포자기 심정으로 따로 구성없이 했는데 이게 점점 이상해지는 것 같다.

오늘 한번 작정하고 수정해보도록 하자.

### 현재 카테고리 상황

BookReview
Docker
Front_End
Git
Linux
OpenCV
끄적임
논문리뷰
문법
연구_인공지능
용어_C++
용어_Python
용어_보안
용어_수학
용어_인공지능
의학
회사에서

---

사실 카테고리를 보면 통합할 수 있는게 많이 보인다.

태그로 넣어도 충분할 것 같은 것도 보이고

### 태그의 문제점

태그가 너무 많아지니까 스크롤 해도 맨 아래 태그를 보기 힘들어졌다.

이건 블로그 세팅 문제라... 나중에 고치기로 하자

### 카테고리 통합하기

Container
    - Docker
    - Kubernetes
    - ...

OS
    - Windows
    - Linux
    - ...

Git
    - Github
    - GitBlog
    - ...

끄적임 -> etc.
    - 회사에서
    - ...

### 애매한거

공부
    - 컴퓨터공학
        - BookReview
        - Front_End
        - OpenCV
        - 논문리뷰
        - 문법
        - 연구_인공지능
        - 용어_C++
        - 용어_Python
        - 용어_보안
        - 용어 _인공지능
    - 영어
    - 수학
    - 의학

### 태그는 어떻게 할 것 인가?

태그 앞에 숫자를 붙여서 중요한 태그를 상위에 위치시킬 수는 없을까?

### 카테고리 순서정렬

현재 내 블로그에서 카테고리와 테그는 첫번째 문자 순서대로 오름차순이다.

1, 11, 2, 3, 4 .. 이런식

![아스키 코트표](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile5.uf.tistory.com%2Fimage%2F216CE84C52694FF02054D4)

위 아스키코드표를 따라서 카테고리르 정렬하도록 한다.

### 최종 카테고리

상위 카테고리, 하위 태그

1. 컴퓨터 공학
    1. Programming
        1. python
        2. C++
        3. Java
        4. ...
    2. Artificial Intelligence
        1. Machine Learning
        2. Deep Learning
            1. ANN
            2. CNN
            3. RNN
            4. GAN
            5. ...
    3. Git
        1. GitHub
        2. GitBlog
    4. OS
        1. Linux
        2. Windows
    5. Container
        1. Docker
        2. Kubernetes
    6. Literature review
2. 수학
3. 영어
4. 의학
5. etc.
    1. 회사에서
    2. 삶을 살아가는 태도