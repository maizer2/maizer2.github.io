"""
Add per-book tags to each post under _posts/5. BookReview/.

For each post, parse the book name from the filename's leading "(...)" prefix
and inject a "5.X. <Book Name>" tag into the front matter, so the Categories
tree displays posts grouped by book under "5. BookReview".

Book numbers are assigned by first-publication date (oldest = 5.1.).
"""
import os
import re
import sys
from collections import OrderedDict

BOOKREVIEW_DIR = '_posts/5. BookReview'

# slug (as it appears in filename) -> display name (with spaces, ordered by first appearance date)
BOOKS = OrderedDict([
    ('프로그래머를-위한-선형대수',                   '프로그래머를 위한 선형대수'),
    ('OpenCV-4로-배우는-컴퓨터-비전과-머신-러닝',     'OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝'),
    ('Hands-On-Machine-Learning-2',                'Hands-On Machine Learning 2'),
    ('실전-예제로-배우는-GAN',                      '실전 예제로 배우는 GAN'),
    ('한-걸음씩-알아가는-선형대수학',                 '한 걸음씩 알아가는 선형대수학'),
    ('선형대수와-통계학으로-배우는-머신러닝-with-파이썬', '선형대수와 통계학으로 배우는 머신러닝 with 파이썬'),
    ('파이토치-첫걸음',                            '파이토치 첫걸음'),
    ('혼공머신',                                  '혼공머신'),
    ('미술관에-GAN-딥러닝-실전-프로젝트',             '미술관에 GAN 딥러닝 실전 프로젝트'),
])

SLUG_TO_TAG = {
    slug: f'5.{idx}. {display}'
    for idx, (slug, display) in enumerate(BOOKS.items(), start=1)
}


def add_book_tag(content: str, book_tag: str) -> str | None:
    """Insert book_tag into the tags: line. Return new content, or None if unchanged."""
    if not content.startswith('---'):
        return None
    parts = content.split('---', 2)
    if len(parts) < 3:
        return None
    _, fm, body = parts

    lines = fm.split('\n')
    out_lines = []
    found_tags = False
    changed = False

    for line in lines:
        if line.lstrip().startswith('tags:') and not found_tags:
            found_tags = True
            if book_tag in line:
                out_lines.append(line)
                continue
            # Match: <indent>tags: [<items>]
            m = re.match(r'^(\s*tags:\s*\[)(.*?)(\]\s*)$', line)
            if m:
                prefix, items, suffix = m.group(1), m.group(2), m.group(3)
                items = items.rstrip()
                items = f'{items}, {book_tag}' if items else book_tag
                out_lines.append(prefix + items + suffix)
                changed = True
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    if not found_tags:
        # No tags: line — insert one right after categories: (or at top of FM)
        inserted = False
        out_lines2 = []
        for line in out_lines:
            out_lines2.append(line)
            if not inserted and line.lstrip().startswith('categories:'):
                # Preserve leading indent
                indent = line[:len(line) - len(line.lstrip())]
                out_lines2.append(f'{indent}tags: [{book_tag}]')
                inserted = True
                changed = True
        if not inserted:
            # Fallback: prepend to fm
            out_lines2.insert(1, f'tags: [{book_tag}]')
            changed = True
        out_lines = out_lines2

    if not changed:
        return None
    return '---' + '\n'.join(out_lines) + '---' + body


def main() -> int:
    dry = '--dry-run' in sys.argv
    files = sorted(os.listdir(BOOKREVIEW_DIR))
    changed_count = 0
    unmatched = []

    for fname in files:
        if not fname.endswith('.md'):
            continue
        m = re.match(r'^\d{4}-\d{2}-\d{2}-\(([^)]+)\)', fname)
        if not m:
            unmatched.append(fname)
            continue
        slug = m.group(1)
        book_tag = SLUG_TO_TAG.get(slug)
        if not book_tag:
            unmatched.append(f'{fname}  (slug: {slug})')
            continue

        path = os.path.join(BOOKREVIEW_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = add_book_tag(content, book_tag)
        if new_content is None:
            continue

        changed_count += 1
        if dry:
            print(f'[DRY] would tag {fname} -> {book_tag}')
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'tagged {fname} -> {book_tag}')

    print(f'\nTotal {"would change" if dry else "changed"}: {changed_count} file(s)')
    if unmatched:
        print(f'\nUnmatched ({len(unmatched)}):')
        for u in unmatched:
            print(f'  {u}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
