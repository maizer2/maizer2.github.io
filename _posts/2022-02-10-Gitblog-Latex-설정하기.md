---
layout: post
title: Gitblog Latex 설정하기
categories: [1. Computer Engineering]
tags: [1.3. Git, 1.3.2 GitBlog, Latex]
---

이 글은 Jekyll 테마 [Yat](https://github.com/jeffreytse/jekyll-theme-yat) 을 기준으로 작성하였습니다.

### 요약

블로그 폴더/_includes/views/article.html

article 태그 아래 해당 코드를 넣어준다.

```html
<script type="text/x-mathjax-config">
	MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

<script type="text/javascript" async
	src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```

[maizer2.github.io/.../article.html](https://github.com/maizer2/maizer2.github.io/blob/main/_includes/views/article.html) 에서 확인하실 수 있습니다.

---

### 서론

평소 수학에는 관심도 없었지만 석사와 ETRI 인턴을 진행하기 위해 필수적으로 수학에 한발 나아가게 되었다.  
하다보니 너무 재밌는 수학... GitBlog에도 쓰고 싶은데... 어떻게 해야하지?

[Yat](https://github.com/jeffreytse/jekyll-theme-yat/blob/master/_posts/2016-01-01-another-test-markdown.md) github에는 별다른 설정 없이도 LaTeX를 쓰는 것을 확인 할 수 있었는데... 내 블로그에 하려고 하니까 안된다...

그래서 찾다 찾다 관련 글을 써주신 블로그를 찾게 되었다. [Link for 근본없는 개발자](https://helloworldpark.github.io/jekyll/update/2016/12/18/Github-and-Latex.html)

---

### 왜 안돼?

위 코드를 넣었는데 적용이 안돼서 내렸다 올렸다 딴곳에 넣어보고 아주 난리를 쳤다.

진짜 열받는게 매번 푸쉬해서 블로그 상태를 확인해야되니까 너무 번거롭고 귀찮았다.. 하지만 루비를 사용해서 서버를 열려고 하니 bundle exec jukyll serve 이 코드가 안먹힌다..

에러가 뜨는데 구글에 쳐도 안나온다... 3시간 찾아봤는데 시간이 너무 아까워서 일단 보류...

---

### 캐쉬, 쿠키 초기화가 답

결국 캐쉬랑 쿠키 초기화 후 적용된것을 확인 하였다... ㅎㅎ

잘 적용이 되는 거 같아 좋았지만... 벌써 시간이 퇴근시간이네... 

---

##### 참고문헌

1. Github로 블로그 만들기 + LaTeX 적용하기, 근본없는 개발자, 2016-12-18 작성, 2022-02-10 방문, "https://helloworldpark.github.io/jekyll/update/2016/12/18/Github-and-Latex.html"