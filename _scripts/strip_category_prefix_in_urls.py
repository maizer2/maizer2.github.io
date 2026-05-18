"""
Strip the old category path prefix from internal links inside post bodies.

Old:  https://maizer2.github.io/<category-path>/YYYY/MM/DD/slug.html
New:  https://maizer2.github.io/YYYY/MM/DD/slug.html

Matches one OR MORE path segments before /YYYY/MM/DD/ and removes them.
Idempotent: already-clean URLs are left alone.
"""
import os
import re
import sys

POSTS_ROOT = '_posts'

# (domain)(/seg1/seg2/.../segN)(/YYYY/MM/DD/slug.html)
# The middle group is one-or-more path segments (= the old category prefix).
URL_RE = re.compile(
    r'(https?://maizer2\.github\.io)'
    r'(?:/[^/\s)"\']+)+?'                     # one or more category segments (non-greedy)
    r'(/\d{4}/\d{2}/\d{2}/[^\s)"\']+\.html)'  # date + slug + .html
)


def main():
    dry = '--dry-run' in sys.argv
    total_subs = 0
    files_changed = 0
    for root, dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as fp:
                content = fp.read()
            new_content, n = URL_RE.subn(r'\1\2', content)
            if n > 0:
                total_subs += n
                files_changed += 1
                if dry:
                    print(f'[DRY] {n} fix(es): {path}')
                else:
                    with open(path, 'w', encoding='utf-8') as fp:
                        fp.write(new_content)
                    print(f'fixed {n}: {path}')
    print(f'\nTotal: {total_subs} URL(s) {"would change" if dry else "changed"} in {files_changed} file(s)')


if __name__ == '__main__':
    main()
