"""
Populate `categories:` in each post's front matter from its directory path.

Jekyll's auto-derive doesn't add categories for directories NESTED INSIDE
_posts/ (it only does so when _posts itself is nested). So we set them
explicitly here.

For _posts/A/B/C/foo.md  ->  categories: [A, B, C]
"""
import os
import re
import sys

POSTS_ROOT = '_posts'


def get_categories_from_path(path: str) -> list[str]:
    """Return the list of directory names between _posts/ and the file."""
    rel = os.path.relpath(path, POSTS_ROOT)
    parts = rel.replace('\\', '/').split('/')
    return parts[:-1]  # drop the filename


def process_file(path: str, dry_run: bool = False) -> bool:
    cats = get_categories_from_path(path)
    if not cats:
        return False

    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    if not content.startswith('---'):
        return False
    parts = content.split('---', 2)
    if len(parts) < 3:
        return False
    _, fm, body = parts

    cats_yaml = '[' + ', '.join(cats) + ']'
    new_line = f'categories: {cats_yaml}'

    new_fm_lines = []
    replaced = False
    for line in fm.split('\n'):
        if line.lstrip().startswith('categories:'):
            indent = line[:len(line) - len(line.lstrip())]
            new_fm_lines.append(indent + new_line)
            replaced = True
        else:
            new_fm_lines.append(line)
    if not replaced:
        # Insert after `layout:` (typical placement)
        inserted = False
        out = []
        for line in new_fm_lines:
            out.append(line)
            if not inserted and line.lstrip().startswith('layout:'):
                indent = line[:len(line) - len(line.lstrip())]
                out.append(indent + new_line)
                inserted = True
        if not inserted:
            out.insert(1, new_line)
        new_fm_lines = out

    new_content = '---' + '\n'.join(new_fm_lines) + '---' + body
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    return True


def main():
    dry = '--dry-run' in sys.argv
    changed = 0
    for root, dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            if process_file(os.path.join(root, f), dry_run=dry):
                changed += 1
    print(f'{"Would update" if dry else "Updated"} categories in {changed} files')


if __name__ == '__main__':
    main()
