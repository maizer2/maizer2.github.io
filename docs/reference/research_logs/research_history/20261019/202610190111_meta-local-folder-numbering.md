# 포스트 폴더 번호를 전역(dotted) 번호에서 로컬 번호로 전환

## 메타 정보
- 작성 시각: 2026-10-19 01:11 (KST)
- 관련 대화 주제: `_posts/` 하위 폴더 이름이 상위 번호를 이어 써서 길어지는 문제 해결
- 관련 파일
  - `_scripts/localize_folder_numbers.py` (신규, 폴더 이름 변환)
  - `_scripts/populate_categories_from_path.py` (기존, `categories:` 재생성에 사용)
  - `assets/js/category-tree.js` (트리 파싱·키 생성 로직 수정)
  - `_includes/views/post-header.html` (메인 카테고리 선택 로직 수정)
  - `README.md` (폴더 이름 규칙 문서화)

## 배경 / 맥락
기존 폴더 이름은 `1. Computer Engineering/1.1. Artificial Intelligence/1.1.2. Deep Learning/1.1.2.3. Trustworthy AI/1.1.2.3.1. AI Safety/1.1.2.2.3. Jailbreak` 처럼 상위 번호를 모두 이어 쓰는 **전역 번호** 방식이었다.
사용자 지적: 깊어질수록 이름이 매우 길어지고 구분이 어렵다. 하위 폴더에서도 `0`부터 다시 번호를 시작하는 **로컬 번호**를 원함.

조사 결과 전역 번호는 이미 일관성이 깨져 있었다. 예: `1.1.2.3.1. AI Safety` 아래 자식들이 `1.1.2.2.1. Benchmark`, `1.1.2.2.2. Safety Training`, `1.1.2.2.3. Jailbreak` 로 부모 접두사와 불일치. 전역 번호는 폴더를 옮기거나 상위 번호가 바뀔 때마다 하위 전체를 갱신해야 해서 유지가 어렵다는 증거.

## 논의 내용 (번호 체계에 의존하던 지점 조사)
1. **`_scripts/populate_categories_from_path.py`**: 경로의 디렉토리 이름을 그대로 `categories: [...]` 에 넣음. 번호 형식과 무관하므로 수정 불필요. 단, 폴더 이름 변경 후 재실행 필요.
2. **`assets/js/category-tree.js`**
   - `parseEntry`가 `"1.1.2. Deep Learning"` 의 점 구분 접두사를 분해해 `prefix=['1','1','2']` 를 만들고, 이것으로 (a) 노드 키(`"1.1.2"`), (b) 로컬 표시명(`"2. Deep Learning"`), (c) `categories` 배열을 깊이순으로 정렬(`categoryDepth`)하는 데 사용.
   - 로컬 번호에서는 (c)가 작동하지 않음(모두 깊이 1). 하지만 `categories` 배열은 populate 스크립트가 경로 순서(바깥→안쪽)로 채우고 Jekyll이 YAML 순서를 보존하므로 정렬 자체가 불필요.
   - 노드 키가 전역 접두사에서 나왔으므로 로컬 번호에서는 `"1"` 이 여러 깊이에 중복 등장 → `highlightPath`, `defaultExpandedKeys`, `li.dataset.key` 가 충돌할 위험. 키를 **부모 키 + '.' + 로컬 번호** 로 합성하면 기존과 동일한 `"1.1.2"` 형태의 고유 키가 유지됨.
   - 사용되지 않던 `ensurePath`, `keyFromPrefixSlice` 는 제거.
3. **`_includes/views/post-header.html`**: `page.categories | sort | last` 로 leaf(메인 카테고리)를 찾았음. 전역 번호에서는 문자열 정렬 시 접두사가 긴 항목이 뒤로 가서 우연히 동작했지만, 로컬 번호에서는 `"0. General"` 같은 항목이 앞으로 오는 등 깨짐. 배열 마지막 항목(`| last`)을 쓰면 경로 순서상 leaf가 됨.
4. **`_layouts/post.html`** prev/next: `page.categories | join: "||"` 로 전체 경로 문자열 비교 → 로컬 번호여도 경로 전체가 비교되므로 그대로 동작.
5. **`site.categories` 해시**(`tree-data.html` entries, `get_categories.html`, `_layouts/categories.html`): 로컬 번호에서는 `"0. General"` 처럼 서로 다른 경로의 폴더가 같은 이름으로 합쳐짐. 그러나 `pages/categories.html` 은 `layout: tree` 를 쓰고, 트리 JS는 `entries` 를 읽지 않으며, `layout: categories` 를 쓰는 페이지가 없음 → 실사용 영향 없음(레거시 템플릿).
6. **본문 내 폴더 경로 참조 / 이미지 경로**: `assets/images` 는 폴더 번호 구조를 따르지 않고, 포스트 본문에 전역 번호 경로를 참조하는 곳 없음(grep 확인).
7. **형제 충돌 검사**: 전체 폴더에서 마지막 번호만 남겼을 때 같은 부모 아래 번호가 겹치는 경우 없음(스크립트 내 COLLISION 검사로 확인).

## 검토한 대안
- **A. 로컬 번호(마지막 세그먼트만 유지)** — 채택. 이름이 짧아지고, 폴더 이동 시 하위 갱신 불필요. 기존 불일치 번호(`1.1.2.2.x` under `1.1.2.3.1`)도 자연히 해소.
- **B. 전역 번호 유지 + 불일치만 수정** — 길이 문제가 그대로 남고, 사용자의 요구와 불일치.
- **C. 번호를 완전히 제거하고 이름만 사용** — 정렬 순서를 폴더 이름으로 제어할 수 없어짐(현재 `0. General` 을 맨 앞에 두는 관례 유지 불가).
- **D. 로컬 번호로 바꾸면서 빈 번호(gap)도 재부여** (예: Deep Learning 아래 `1, 3` → `1, 2`; DevOps 아래 `1,2,3,5` → `1,2,3,4`) — 사용자가 의도적으로 비워둔 번호일 수 있어 **하지 않음**. 요청 시 별도 작업.

## 결정사항
1. `_posts/` 하위 모든 폴더 이름을 `<로컬 번호>. <이름>` 으로 변경한다(72개 폴더). 변환 규칙: 전역 접두사의 **마지막 세그먼트**만 유지.
2. 번호 gap 과 기존 순서는 그대로 둔다(재부여 안 함).
3. 변환은 `_scripts/localize_folder_numbers.py` 로 수행하며, 멱등(이미 로컬이면 변화 없음)·bottom-up·형제 충돌 시 중단한다.
4. 변환 후 `_scripts/populate_categories_from_path.py` 로 364개 포스트의 `categories:` 를 재생성한다.
5. `category-tree.js`: 깊이 정렬 제거, 배열 순서로 계층 구성, 노드 키 = 부모 키 + '.' + 로컬 번호, 표시명 = 항목 그대로. 같은 부모 아래 번호가 같고 이름이 다른 경우 키에 `:<이름>` 을 붙여 분리. 레거시 전역 접두사가 들어와도 마지막 세그먼트로 해석.
6. `post-header.html`: 메인 카테고리 = `page.categories | last`.
7. `README.md` 에 폴더 이름 규칙(로컬 번호, `0` = General) 섹션 추가.
8. 연구 기록용 `docs/` 폴더는 Jekyll 빌드에서 제외한다(`_config.yml` exclude).

## 결정 근거
- (1,2) 사용자 요구는 "하위 폴더에서도 0부터". 마지막 세그먼트 유지는 정보 손실 없이 가장 작은 변경이며, 이미 깨진 전역 번호와의 불일치도 해소된다. gap 재부여는 사용자 의도를 추정해야 하므로 보류.
- (3,4) 기존 운영 방식(폴더 = 카테고리, 스크립트/Action 으로 front matter 동기화)을 그대로 유지한다.
- (5) 트리의 유일 키 형태(`1.1.2`)를 유지해 `defaultExpandedKeys: ['1.1.2']` 등 기존 호출부가 그대로 동작하게 함. node 시뮬레이션으로 364개 포스트 전체 트리 계층·카운트·natural sort(`10.` 이 `9.` 뒤)·기본 펼침을 확인.
- (6) 정렬에 의존하던 우연한 동작을 명시적 규칙(배열 마지막 = leaf)으로 대체.

## 검증
- `localize_folder_numbers.py --dry-run` 재실행 → 0건(멱등 확인).
- PyYAML 로 364개 포스트 front matter 파싱 → `categories` 가 모두 문자열이며 경로와 일치.
- git 이 364개 파일을 전부 rename(R) 으로 인식.
- 로컬에 bundler 가 없어 Jekyll 빌드는 미실행. push 시 `pages.yml` Action 빌드로 최종 확인 필요.

## 미해결 사항 (Open Questions)
- 번호 gap(`2. Deep Learning` 아래 `2` 없음, `3. DevOps & Infra` 아래 `4` 없음) 재부여 여부.
- `6. etc/1. Work & Career` 아래 `1. ETRI` 와 `3. ETRI` 가 이름이 같음(기존부터). 구분 이름이 필요하면 별도 결정.
- 레거시 템플릿(`_layouts/categories.html`, `sidebar/category-list.html`, `tree-data.html` 의 `entries`)은 `site.categories` 이름 충돌(`0. General`)에 노출되지만 현재 미사용. 정리 여부.

## 변경 이력
- 2026-10-19 최초 작성.
