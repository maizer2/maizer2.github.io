---
layout: post
title: "AutoEncoder"
categories: [1. Computer Engineering]
tags: [1.2. Artificial Intelligence, 1.2.2.4. VAE, a.b. UnSupervised Learning]
---

### AutoEncoder란?<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

입력 데이터의 특성을 추출하여 입력 데이터와 비슷한 데이터를 재현(복원)한다.

<center><img alt="AutoEncoder" src="https://blog.keras.io/img/ae/autoencoder_schema.jpg"></center>

<center>AutoEncoder 추상화<sup><a href="#footnote_2_1" name="footnote_2_2">[2]</a></sup></center><br/>

### AutoEncoder 특징

1. UnSupervised Learning
    * 입력값 $x$만 들어있기 때문
2. Manifold Learning
    * Feature(특성)추출시 차원축소가 일어나기 떄문
3. Generative Model
    * Feature를 통해 데이터를 생성한다.
4. ML, Maximum Likelihood density estimation, [최대가능도방법](https://ko.wikipedia.org/wiki/최대가능도_방법)
    * 최대가능도방법을 사용하여 학습한다.

### Manifold Learning  
#### 왜 Manifold Learning을 해야할까?

1. Data 압축
2. Data 시각화
3. Curse of Dimensionality (차원의 저주), 데이터 차원이 커질수록 필요한 샘플 데이터가 기하급수적으로 커지는 것을 해소한다.
    * 고차원 데이터를 저차원으로 접어서 표현한다.

<center><img alt="Manifold" src="https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.2.%20Artificial%20Intelligence/2022-03-29-AutoEncoder/Manifold.png"></center>

<center>Manifold 시각화<sup><a href="#footnote_3_1" name="footnote_3_2">[3]</a></sup></center><br/>

$d$차원의 Manifold, $M$은 $m$차원의 공간에 속해 있으며, $f: R^{d} \to R^{m}$을 만족하는 명시적인 매핑 함수 $f$가 존재한다 $(d<m)$

* 데이터의 위치와 방향에 따라 달라보이는 값이 Manifold 학습을 통해 같은 데이터임을 알 수 있음. 
* 두 샘플간의 거리가 먼 경우 Manifold 를 통해 거리 축소가 가능하다.

### Manifold Learning Algorithm  
#### t-SNE, t-Stochastic Neighbor Embedding

고차원 공간을 가지고 있는 데이터를 저차원 공간으로 매핑하는 것

Reconstruction, 재배치를 통해 비슷한 특성끼리 군집하게 만들어준다.

Perplexity란, Sample의 가까운 점(Sampel)은 Label에 상관없이 같은 군집이라고 가정하는 척도

* Perplexity가 작으면 Sample이 멀기 때문에 군집 공간의 크기가 커진다.
* Perplexity가 크면 Sample이 가까워서 군집 공간의 크기가 작아진다.

#### UMAP, Uniform Manifold Approximation and Projection

퍼지이론을 기초로 표현한다.

유클리드 공간에서 가까운 Sample들을 군집화 한다.

거리에 따라 촌수 관계를 통해 1촌, 2촌, ..., n촌이 되며 공간을 왜곡하지 않고 군집을 정리할 수 있다.

UMAP은 t-SNE에 비해 공간 차지가 적고 성능이 좋다.


### AutoEncoder의 이론적 설명

<center><img width="800" alt="AutoEncoder" src="https://raw.githubusercontent.com/maizer2/gitblog_img/main/1.%20Computer%20Engineering/1.2.%20Artificial%20Intelligence/2022-03-29-AutoEncoder/AutoEncoder.PNG"></center>

<center>AutoEncoder $E(x)\to z \to D(\hat{x})$</center><br/>

### Boltzmann Machine<sup><a href="#footnote_4_1" name="footnote_4_2">[4]</a></sup>

통계학의 볼츠만 통계에 유래하였다. [홉필드 네트워크](https://en.wikipedia.org/wiki/Hopfield_network)와 같이 완전 연결된 인공 신경망이지만, 확률적 뉴런을 기반으로 계단 함수를 사용하여 결정론적으로 출력값을 만들지 않고 어느 정도 확률을 가지면 1을 출력하고 아니면 0을 출력한다.

볼츠만 머신의 뉴런은 **가시 유닛**과 **은닉 유닛** 두개의 그룹으로 나눠져 있고 모든 뉴런은 동일한 확률적 방식으로 작동한다.

**가시 유닛**만이 **입력**과 **출력**을 한다.

확률로 인해 안정화되지 못하고 여러 설정이 계속 전환된다. 하지만 충분히 긴 시간동안 실행되면 **열평형**을 통해 특정 설정이 관측되게 된다.

볼츠만 머신을 훈련시키기 위한 효율적이 방법은 없다. 하지만 **제한된 볼츠만 머신**은 매우 효율적인 훈련 알고리즘이 개발되었다.

### Restricted Boltzmann Machine, RBM, 제한된 볼츠만 머신<sup><a href="#footnote_5_1" name="footnote_5_2">[5]</a></sup>

<center><img alt="RBM" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Restricted_Boltzmann_machine.svg/330px-Restricted_Boltzmann_machine.svg.png"></center>

<center>제한된 볼츠만 머신</center><br/>

**가시 유닛**과 **히든 유닛**사이에만 연결층이 있다. 기존의 <a href="#footnote_4_2" name="footnote_4_1">볼츠만 머신</a>은 가시 유닛끼리, 히든 유닛끼리 연결되었었다.

*미구엘 카레이라 페르피닝*과 *제프리 힌튼*이 **CD, Contrastive Divergence** 라는 효율적인 훈련 알고리즘을 제안하였다.

최대 장점은 네트워크가 **열평형**에 도달할 때까지 기다릴 필요가 없다. 매우 효율적이다.

### Deep Belief Network, DBN, 심층 신뢰 신경만

여러 개의 RBM 층을 쌓아올려 만들었다.

하위층은 입력 데이터 $x$에서 저수준 Feature, $z$를 학습, 상위층은 고수준의 $z$를 학습한다.

---
##### 참고문헌

<a href="#footnote_1_2" name="footnote_1_1">1.</a> 확률적 그래픽 모델 12강 Autoencoder, KMOOC 이준열 겸임교수 (성균관대학교 스마트팩토리융합학과), [http://www.kmooc.kr/courses/course-v1:AIIA+AIIA01+2021_T3_AIIA01/courseware/4948761df9e44d3a91f360233310bd39/d7627d7a7be143749d382c8ddd7455a0/1?activate_block_id=block-v1%3AAIIA%2BAIIA01%2B2021_T3_AIIA01%2Btype%40vertical%2Bblock%407bc6cfa7c50c4ccc8b25a36688f0775c](http://www.kmooc.kr/courses/course-v1:AIIA+AIIA01+2021_T3_AIIA01/courseware/4948761df9e44d3a91f360233310bd39/d7627d7a7be143749d382c8ddd7455a0/1?activate_block_id=block-v1%3AAIIA%2BAIIA01%2B2021_T3_AIIA01%2Btype%40vertical%2Bblock%407bc6cfa7c50c4ccc8b25a36688f0775c)

<a href="#footnote_2_2" name="footnote_2_1">2.</a> Building Autoencoders in Keras, The Keras Blog, [https://blog.keras.io/building-autoencoders-in-keras.html](https://blog.keras.io/building-autoencoders-in-keras.html)

<a href="#footnote_3_2" name="footnote_3_1">3.</a> [인공지능 이론] Manifold Learning, roytravel.tistory, [https://roytravel.tistory.com/105](https://roytravel.tistory.com/105)

<a href="#footnote_4_2" name="footnote_4_1">4.</a> 볼츠만 머신, "핸즈온핸즈온 머신러닝(2판): 사이킷런, 케라스, 텐서플로 2를 활용한 머신러닝, 딥러닝 완벽 실무", 오렐리아제롱 작성, 박해선 옮김, 901p

<a href="#footnote_5_2" name="footnote_5_1">5.</a> 제한된 볼츠만 머신, "핸즈온핸즈온 머신러닝(2판): 사이킷런, 케라스, 텐서플로 2를 활용한 머신러닝, 딥러닝 완벽 실무", 오렐리아제롱 작성, 박해선 옮김, 903p
