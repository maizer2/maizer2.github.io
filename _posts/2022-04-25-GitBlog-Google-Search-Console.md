---
layout: post
title: GitBlog Google에 노출하기
categories: [1. Computer Engineering]
tags: [1.3. Git, 1.3.2. GitBlog]
---

### Google Search Console

[https://search.google.com/search-console](https://search.google.com/search-console)에서 블로그를 google에 노출할 수 있다.

속성 추가를 통해 블로그 url등록이 가능하다.

URL 접두어에 블로그 주소를 넣어주고 계속

html파일을 추가하라는 안내에 따라 블로그 폴더에 넣어준다.

### GitBlog 설정

GitBlog, naverblog등 다양한 사이트에 sitemap이라는 xml파일을 가지고 있다.

sitemap은 웹 사이트, 블로그의 지도라고 생각할 수 있다.

모든 페이지를, 페이지의 url을 한페이지에 모두 모아뒀다.

이를 직접 xml파일을 폴더에 추가해주는 방법과 Jekyll 플러그인을 통해 추가해주는 방법이 있다.

jekyll 서버를 열줄 안다면 플러그인을 추가할 수 있고, 그게 아니면 xml파일을 통해 자동으로 웹 페이지를 크롤링하도록 할 수 있다.

### 플러그인 추가

[https://maizer2.github.io/1.%20computer%20engineering/2022/04/24/open-jekyll-server.html](https://maizer2.github.io/1.%20computer%20engineering/2022/04/24/open-jekyll-server.html)

jekyll 서버 여는 방법을 정리해 뒀다.

[https://github.com/jekyll/jekyll-sitemap](https://github.com/jekyll/jekyll-sitemap)

sitemap 플러그인을 확인할 수 있다.

---

Usage를 따라하면 쉽게 적용할 수 있다.

#### Usage

Gemfile에 다음을 추가해준다. 추가 후 bundle 실행

```Gemfile
# Add gem 'jekyll-sitemap' to your site's Gemfile and run bundle

gem 'jekyll-sitemap
``` 
_config.yml에 플러그인을 등록해준다.

```_config.yml
# Add the following to your site's _config.yml:

url: "블로그 주소 추가" # the base hostname & protocol for your site
plugins:
  - jekyll-sitemap
```

### sitemap 접속해보기

github_username.github.io/sitemap.xml 에 접속해본다.

접속했을 때 xml파일이 뜨면(뭔가 글이 쭉 뜬다) 성공한거고

404 페이지가 뜨면 플러그인 설치가 실패한거다.

플러그인 설치를 다시 해보고 안되면 다음을 따른다

### sitemap.xml 파일 직접 추가해주기 

[https://www.ascentkorea.com/what-is-robots-txt-sitemap-xml/](https://www.ascentkorea.com/what-is-robots-txt-sitemap-xml/)

위 블로그를 참고해봐도 좋다.

```sitemap.xml
  <? xml version = "1.0"encoding = "UTF-8"?>
<urlset
      xmlns = "http://www.sitemaps.org/schemas/sitemap/0.9"
      xmlns : xsi = "http://www.w3.org/2001/XMLSchema-instance"
      xsi : schemaLocation = "http://www.sitemaps.org/schemas/sitemap/0.9
      http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd ">
<URL>
  <loc> https://www.ascentkorea.com/search-engine-optimization/ </ loc>
  <priority> 1.00 </ priority>
  <changefreq> weekly </ changefreq>
</ url>
<URL>
  <loc> https://www.ascentkorea.com/search-data-research-content-marketing/ </ loc>
  <priority> 0.80 </ priority>
  <changefreq> weekly </ changefreq>
</ url>
<URL>
  <loc> https://www.ascentkorea.com/japan-marketing/ </ loc>
  <priority> 0.80 </ priority>
  <changefreq> weekly </ changefreq>
</ url>
<URL>
  <loc> https://www.ascentkorea.com/ascent-korea-official-blog-listeningmind/ </ loc>
  <priority> 0.80 </ priority>
  <changefreq> weekly </ changefreq>
</ url>
</ urlset>
```

sitemap.xml 파일을 블로그 상위, root에 위치시킨다.

[https://github.com/maizer2/maizer2.github.io](https://github.com/maizer2/maizer2.github.io) 내 블로그를 참고해봐도 좋다.

### google search console에 sitemap 등록하기

콘솔창 좌측 Sitemaps를 클릭하면 사이트맵 url을 추가할 수 있다.

sitemap.xml 제출

제출을 통해 사이트맵에 등록되는데 생각보다 설치 시간이 오래 걸리고, 한달이 지나도 안되는 경우도 있다.

### 콘솔 왼쪽에서 URL 검사

URL 검사 탬을 통해 .../sitemap.xml 페이지를 등록할 수 있다.

등록이 오래걸릴 수 있지만 우리가 할 수 있는 최선이다.

그래도 여기까지 해봤는데 안되면 커뮤니티에 문의하는 방법이 있다.

사실 나도 아직 sitemap 등록을 못했다.