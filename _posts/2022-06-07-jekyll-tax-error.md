---
layout: post
title: "TEX error to Jekyll"
categories: [1. Computer Engineering]
tags: [1.3. Git, 1.3.2. GitBlog]
---

분명히 MathJex 설정을 잘못한거 같은데 수정할 시간은 없고...

일단 적용안되는 문법이 있어서 기술해둠

### Error

|Tex|Explanation|
|---|-----------|
|\mathbb{}|Git에서는 사용됨, Jekyll에선 안됨|
|$\mid{}$|\mid{}로 사용|
|(a)_{b} .. $$|한 문장에 괄호_{} 문법을 하고 $$ 문법을 추가하면 에러생김|
|\underset{\parallel}{x}|underset 아래에 $\parallel$사용 안됨|
|\underbrace{}{}|\underbrace{}{}다음에 수식오면 인식 안됨|
|∀|\forall{} 로 사용|
|*|\*사용하지말고 \ast{}로 사용|
|1_{}|숫자_{} 수식 적용안됨|
|a^{}_{}|a_{}^{}와 같이 아래, 위 순서를 지켜야 에러 안남|
|$\vert|\vert로 사용해야됨 아니면 테이블로 인식함|