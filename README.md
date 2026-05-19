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

## 관리자 페이지 (Decap CMS) 설정

[/admin/](https://maizer2.github.io/admin/)에서 웹 UI로 글을 작성/수정할 수 있습니다.
이 기능은 **Netlify 호스팅**과 **Netlify Identity** 인증이 필요합니다.

### 1) Netlify에 사이트 연결

1. https://app.netlify.com 에서 GitHub 계정으로 가입/로그인
2. **Add new site → Import an existing project → GitHub → `maizer2.github.io` 선택**
3. Build settings는 `netlify.toml`이 자동 인식 — 그대로 두고 **Deploy**
4. 배포가 끝나면 `xxx.netlify.app` 주소가 생성됨 (커스텀 도메인은 나중에 연결 가능)

### 2) Netlify Identity 활성화

Netlify 사이트 대시보드에서:

1. **Site configuration → Identity → Enable Identity**
2. **Registration preferences → Invite only** (외부인이 마음대로 가입 못 하도록)
3. **Services → Git Gateway → Enable Git Gateway** (CMS가 repo에 commit할 수 있게 함)

### 3) 본인을 관리자로 초대

1. Identity 탭 → **Invite users** → 본인 이메일 입력
2. 받은 메일의 "Accept the invite" 링크 클릭
3. 자동으로 `/admin/`으로 리다이렉트되며 비밀번호 설정 후 로그인

### 4) 사용

- `/admin/`에서 글 작성/수정/삭제
- 저장 시 자동으로 repo에 commit → Netlify가 자동 빌드 & 배포 (~1분)
- `categories:` 필드는 **`.github/workflows/sync-categories.yml`** GitHub Action이 자동으로 채워줍니다
  (push 후 새 commit이 생기고 Netlify가 한 번 더 재빌드됨 — 무료 플랜에서 문제 없음)

### 로컬에서 CMS 미리보기

`admin/config.yml`에 `local_backend: true`를 추가한 뒤:

```bash
npx decap-server   # 별도 터미널에서
bundle exec jekyll serve --future
```

`http://localhost:4000/admin/`에서 GitHub 인증 없이 로컬 파일을 직접 편집할 수 있습니다 (배포 시에는 이 옵션 제거).
