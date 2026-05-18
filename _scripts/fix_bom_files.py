"""
Fix the 3 BOM-tainted Hands-On files that were skipped by the swap and
book-tagging scripts.

Steps per file:
  1. Strip UTF-8 BOM.
  2. Apply the 1.1.x <-> 1.2.x swap (they were missed earlier).
  3. Fix the rare "tag:" (singular) typo to "tags:".
  4. Add the per-book tag.
"""
import os
import sys

sys.path.insert(0, '_scripts')
from swap_tags import swap_tag_line
from add_book_tags import SLUG_TO_TAG, add_book_tag

BOM = '﻿'

TARGETS = [
    '_posts/5. BookReview/2022-01-13-(Hands-On-Machine-Learning-2)-0.-서론.md',
    '_posts/5. BookReview/2022-01-13-(Hands-On-Machine-Learning-2)-1.-한눈에-보는-머신러닝.md',
    '_posts/5. BookReview/2022-02-07-(Hands-On-Machine-Learning-2)-2.-머신러닝-프로젝트-처음부터-끝까지.md',
]

BOOK_TAG = SLUG_TO_TAG['Hands-On-Machine-Learning-2']


def process(path: str) -> None:
    with open(path, 'r', encoding='utf-8-sig') as f:  # 'utf-8-sig' strips BOM on read
        content = f.read()

    if content.startswith(BOM):
        content = content[len(BOM):]

    # Step: fix "tag:" -> "tags:" typo in front matter
    parts = content.split('---', 2)
    if len(parts) == 3:
        _, fm, body = parts
        new_lines = []
        for line in fm.split('\n'):
            stripped = line.lstrip()
            if stripped.startswith('tag:') and not stripped.startswith('tags:'):
                indent_len = len(line) - len(stripped)
                line = line[:indent_len] + 'tags:' + stripped[len('tag:'):]
            new_lines.append(line)
        content = '---' + '\n'.join(new_lines) + '---' + body

    # Step: apply 1.1 <-> 1.2 swap on tag/category lines only
    parts = content.split('---', 2)
    if len(parts) == 3:
        _, fm, body = parts
        new_lines = []
        for line in fm.split('\n'):
            stripped = line.lstrip()
            if stripped.startswith('tags:') or stripped.startswith('categories:'):
                new_lines.append(swap_tag_line(line))
            else:
                new_lines.append(line)
        content = '---' + '\n'.join(new_lines) + '---' + body

    # Step: add per-book tag
    new_content = add_book_tag(content, BOOK_TAG)
    if new_content is not None:
        content = new_content

    # Write back WITHOUT BOM
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)

    print(f'fixed: {path}')


def main() -> int:
    for path in TARGETS:
        if not os.path.exists(path):
            print(f'NOT FOUND: {path}')
            continue
        process(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
