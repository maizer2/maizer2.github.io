---
layout: post
title: "(GitBlog) jekyll 서버 열기"
categories: [1. Computer Engineering]
tags: [1.3. Git, 1.3.2. GitBlog]
---

### Jekyll<sup><a href="#footnote_1_1" name="footnote_1_2">[1]</a></sup>

Jekyll는 비전문가도 쉽게 웹 사이트를 만들어주는 생성기이다.

보편적으로 GitHub를 통해 무료로 호스팅을 하는 블로그, GitBlog를 만들 때 사용된다.

하지만 GitBlog를 생성하는 건 쉽지만 Jekyll를 통해 로컬서버를 여는건 생각보다 까다롭다.

따라서 필자가 Jekyll 로컬 서버를 열면서 겪은 점을 서술하고 실제로 서버도 열어보도록 한다.

### [https://jekyllrb-ko.github.io/](https://jekyllrb-ko.github.io/)

위 사이트는 한국어로 번역된 Jekyll 블로그이다.

이또한 Github를 통해 호스팅하고 있고 Github를 통해 사이트 구조를 쉽게 알 수 있다.

처음 GitBlog를 하는 개발자라면 이렇게 큰 블로그를 통해 공부할 수도 있다.

### Jekyll 블로그 생성하기

[https://jekyllrb-ko.github.io/docs/](https://jekyllrb-ko.github.io/docs/) 공식 Documentation에 들어간다.

Jekyll는 Ruby를 통해 설치할 수 있다.

빠른시작 - 절차 - 1. 완전한 [루비 개발환경](https://jekyllrb-ko.github.io/docs/installation/)을 설치한다.

루비 개발환경을 선택하여 사용자 컴퓨터에 맞는 OS에 따라 설치한다. 최신 버전을 설치해주면 된다.

#### Windows

설치 - 가이드 - 윈도우즈 - 윈도우즈에서의 Jekyll - Jekyll 설치하기 - RunyInstaller 를 통한 설치

[RubyInstaller 다운로드 페이지](https://rubyinstaller.org/downloads/)에서 왼쪽 위 최신버전(본인에 맞는 메모리, x86 or x64)에 맞춰 설치한다.

[https://zelkun.tistory.com/entry/install-jekyll-on-Windows](https://zelkun.tistory.com/entry/install-jekyll-on-Windows) 위 블로그의 사진을 참고하면 크게 어렵지 않게 설치할 수 있다.

설치가 끝나면 Bash창이 뜬다.

> Which components shall be installed? if unsure press ENTER [1,3]

사람들은 대부분 1번을 선택해서 설치한다. 나는 그냥 3번 했다.

설치가 끝나면 Enter를 눌러라고 나오는 그렇게되면 설치가 끝나게된다.

#### ruby version check

cmd에서 ruby 정상설치 확인을 위해 version을 확인한다.

```CMD
ruby --version
```

#### Jekyll, Bundler 설치

[https://jekyllrb-ko.github.io/docs/installation/windows/](https://jekyllrb-ko.github.io/docs/installation/windows/)에서 설치 순서를 따를 수 있다.

gem을 사용하여 jekyll와 bundler를 설치한다.

```CMD
gem install jekyll bundler
```

#### Jekyll version check

```CMD
jekyll --version
```

#### 빈 블로그 폴더 생성하기

Jekyll를 통해 블로그 폴더를 생성해준다.

블로그 폴더를 먼저 생성해주고 나중에 설정을 통해 플러그인등을 설치할 수 있다.

```CMD
jekyll new blog_folder_name
```

blog_folder_name은 폴더 이름이다. 

현재 위치 폴더에서 지정해준 블로그 이름의 폴더가 생성된다.

생성 후 폴더에 들어가면 자동생성된 블로그 폴더들이 있을 것이다.

### 로컬 서버로 블로그 열기

```CMD
Bundler

...

jekyll serve
```

Bundler를 실행하여 폴더 내부에 있는 Gemfile(플러그인)을 설치해준다.

처음 설치할 때 자동으로 설치 돼 있을거지만 한번 더 실행해준다.

이후 Jekyll serve를 통해 서버를 열어준다.

위 커멘드를 두개를 합쳐서 사용하는게 아래 커멘드지래

```CMD
bundler exec jekyll serve
```

나는 에러가 나왔다.

```CMD
...\Ruby31-x64\bin\ruby.exe: invalid option -D  (-h will show valid options) (RuntimeError)
```

루비 설치폴더에서 뭔가 수정해야할 것 같은데아무리 해결방법을 찾아봐도 안나와서 포기했다.

그냥 따로 실행해주면 에러 없이 실행 가능하다

### 서버 실행시 에러

[https://velog.io/@minji-o-j/jekyll-오류-해결](https://velog.io/@minji-o-j/jekyll-오류-해결)

위 블로그를 통해 오류 해결이 가능하다.

커멘드창에서 발생하는 오류 내역을 잘 읽고 해결하자

나는 cannot load such file -- webrick 에러가 발생하여

gem install webrick 를 통해 설치하였더니 해결 됐다.

### 서버 실행

서버가 켜지고 https://localhost:4000 를 통해 접속이 가능하다.

블로그를 들어가면 기본 Theme이 설치 돼 있는데 나는 Yat Theme을 사용하여 블로그를 만들었다.

### Yat Theme 설치

[https://github.com/jeffreytse/jekyll-theme-yat/](https://github.com/jeffreytse/jekyll-theme-yat/)

위 Repo에서 Yat 바닐라 테마를 확인 할 수 있다.

하지만 나는 Yat 테마를 조금 수정한 테마를 설치하도록 하겠다.

[https://github.com/vanhiupun/Vanhiupun.github.io](https://github.com/vanhiupun/Vanhiupun.github.io)

위 테마는 Yat 바탕으로 만들어진 테마인데 조금더 간단하고 화면을 넓게 사용할 수 있다.

git clone을 통해 (혹은 zip으로 내려받아서 풀어주기) 다운로드 해주고 내부 파일을 GitBlog 폴더에 넣어준다.

기존에 GitBlog를 설치해본 경험이 있다면 적절하게 _config 파일을 수정해주고

그게 아니라면 전부 본인의 Gitblog 폴더에 넣어주면된다.

중국인이 수정한 테마다보니 중국어가 써져있어서 불편할 것이다.

[https://github.com/maizer2](https://github.com/maizer2) 그냥 내 블로그 클론해서 써도 된다.

나는 한국에 맞게 DISQUS를 수정해놔서 좀더 편할거다.

블로그 설정이 끝났다면 위에 bundler, jekyll serve를 통해 서버를 열어주고 내부 모습을 확인해준다.