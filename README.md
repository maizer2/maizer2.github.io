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
categories: [1. Computer Engineering, 1. Artificial Intelligence, 2. Deep Learning]
```

매번 손으로 쓰는 대신, 파일을 원하는 폴더에 넣은 뒤 아래 스크립트를 돌리면
**경로 그대로 `categories` 필드를 채워줍니다**:

```bash
python _scripts/populate_categories_from_path.py
```

즉 워크플로우는 **"파일을 leaf 폴더에 넣기 → 스크립트 실행"**.
모든 포스트는 최하위(leaf) 디렉토리에 있어야 합니다.

### 폴더 이름 규칙 — 로컬 번호

폴더 이름은 `<번호>. <이름>` 형식이며, **번호는 부모 폴더 안에서만 유효한 로컬 번호**입니다
(상위 번호를 이어 쓰지 않음). `0`은 해당 단계의 General(총론) 폴더에 씁니다.

```text
_posts/
└── 1. Computer Engineering/
    └── 1. Artificial Intelligence/
        ├── 0. General/
        ├── 1. Machine Learning/
        └── 2. Deep Learning/
            └── 3. Trustworthy AI/
```

카테고리 트리(`assets/js/category-tree.js`)는 `categories` 배열의 **순서**(= 경로 순서)로
계층을 만들고, 포스트 헤더의 메인 카테고리는 배열의 **마지막 항목**(leaf)입니다.
예전 전역 번호(`1.1.2. Deep Learning`) 폴더가 남아 있으면
`python _scripts/localize_folder_numbers.py`로 로컬 번호로 바꿀 수 있습니다.

### `tags` — 직접 지정

태그는 자동화 없이 본인이 front matter에 쓰는 값 그대로 사용됩니다:

```yaml
tags: [JB]
```

포스트 헤더의 "키워드 칩"과 사이드바 "Related Posts" 유사도 점수가 이 태그 기반으로 계산됩니다.

## 배포 (GitHub Pages)

`main` 브랜치에 push하면 **`.github/workflows/pages.yml`** GitHub Action이 Jekyll을 빌드해
GitHub Pages로 자동 배포합니다. 별도 호스팅 서비스는 필요하지 않습니다.

글 작성 워크플로우:

1. `_posts/`의 원하는 leaf 폴더에 마크다운 파일 추가
2. `python _scripts/populate_categories_from_path.py`로 `categories:` 필드 채우기
   (또는 push 시 **`.github/workflows/sync-categories.yml`** Action이 자동으로 채움)
3. commit & push → GitHub Pages 자동 빌드 & 배포
