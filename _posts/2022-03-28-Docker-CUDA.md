---
layout: post
title: "Docker CUDA 설정"
categories: "Docker"
tags: [AI, Machine Learning, Deep Learning, CUDA, Ubuntu]
---

이 글은 모두의 근삼이<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>블로그를 참고하여 작성하였습니다.

### Docker GPU 할당

컨테이너 내부에 GPU를 할당하기 위해서는 nvidia-container-runtime 패키지를 설치해줘야 한다.

### Docker CUDA 설정



---
##### 참고문헌


<a href="#footnote_1_2" name="footnote_1_1">1.</a> docker 컨테이너에서 GPU 사용, 모두의 근삼이, 2020. 05. 14 작성, 2022. 03. 28 방문, [https://ykarma1996.tistory.com/92](https://ykarma1996.tistory.com/92)

<a href="#footnote_2_2" name="footnote_2_1">2.</a> GPU, Docker docs, [https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu) 
