---
layout: post
title: "Entropy of Machine Learning"
categories: [1. Computer Engineering, 2. Mathematics]
tags: [1.2. Artificial Intelligence, 1.2.1. Machine Learning, 2.2. Pobability and Statistics]
---

### **Concept of Entropy**

Entropy는 노드에 서로 다른 데이터가 얼마나 섞여 있는지를 의미하는 impurity(불순도)를 측정한다.

Imputrity가 낮을수록 데이터가 섞여 있지 않다는 것을 의미한다.

<br/>

### **Expressiion of Entropy**

$$Entropy(d) = - \sum p(x)logP(x)$$

$$ \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; = -\sum_{i=1}^{k}p(i|d)log_{2}(p(i|d))$$