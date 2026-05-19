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

## Post Front Matter

### `categories` — 직접 지정 (자동 X)

Jekyll은 `_posts/A/B/C/foo.md` 구조에서 카테고리를 자동으로 만들어주지 않습니다
(자동 도출은 `A/B/_posts/foo.md` 구조일 때만 동작). 그래서 front matter에 명시해야 합니다:

```yaml
categories: [1. Computer Engineering, 1.1. Artificial Intelligence, 1.1.4. Paper Reviews]
```

매번 손으로 쓰는 대신, 파일을 원하는 폴더에 넣은 뒤 아래 스크립트를 돌리면
**경로 그대로 `categories` 필드를 채워줍니다**:

```bash
python _scripts/populate_categories_from_path.py
```

즉 워크플로우는 **"파일을 leaf 폴더에 넣기 → 스크립트 실행"**.
모든 포스트는 최하위(leaf) 디렉토리에 있어야 합니다.

### `tags` — 직접 지정

태그는 자동화 없이 본인이 front matter에 쓰는 값 그대로 사용됩니다:

```yaml
tags: [JB]
```

포스트 헤더의 "키워드 칩"과 사이드바 "Related Posts" 유사도 점수가 이 태그 기반으로 계산됩니다.
