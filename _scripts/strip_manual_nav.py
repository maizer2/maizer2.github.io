"""
Remove the manual prev/next navigation lines from post bodies, e.g.:

  ## [←  이전 글로](URL) 　 [다음 글로 →](URL)
  ## [다음 글로 →](URL)
  ## [←  이전 글로](URL)

This UI is now provided by the post layout's category-scoped nav, so the
hand-written line is redundant. Also strips a trailing <br/> placed
immediately after the nav line (often used as a separator for it).

Idempotent.
"""
import os
import re
import sys

POSTS_ROOT = '_posts'

NAV_LINE_RE = re.compile(r'.*(?:이전 글로|다음 글로).*')


def process(path: str, dry: bool) -> int:
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.read().splitlines(keepends=True)

    out = []
    removed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if NAV_LINE_RE.match(line.strip()) and (
            line.lstrip().startswith('##') or line.lstrip().startswith('[')
        ):
            removed += 1
            i += 1
            # Eat a single immediately-following <br/> (with optional blank between)
            j = i
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines) and lines[j].strip() in ('<br/>', '<br>', '<br />'):
                # Skip the blanks too, but keep one blank line as separator
                i = j + 1
            continue
        out.append(line)
        i += 1

    if removed and not dry:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(out)
    return removed


def main():
    dry = '--dry-run' in sys.argv
    total = 0
    files = 0
    for root, _dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            n = process(os.path.join(root, f), dry)
            if n:
                files += 1
                total += n
    print(f'{"Would remove" if dry else "Removed"} {total} nav line(s) from {files} file(s).')


if __name__ == '__main__':
    main()
