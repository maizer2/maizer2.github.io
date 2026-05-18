# maizer2.github.io

Jekyll로 만든 개인 블로그 소스입니다.

## Structure

- `_posts/`: 공개 글
- `_posts_temporary/`: 보관 중인 임시 글. 빌드에서는 제외합니다.
- `_layouts/`, `_includes/`, `_sass/`: 블로그 화면과 스타일
- `assets/`: CSS, JavaScript, 이미지

## Local Development

```bash
bundle install
bundle exec jekyll serve --future
```

기본 주소는 `http://localhost:4000`입니다.
