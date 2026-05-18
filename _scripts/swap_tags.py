"""
Swap 1.1.x <-> 1.2.x prefixes in all post tags/categories.

Examples:
  "1.1. Programming"          -> "1.2. Programming"
  "1.2. Artificial Intelligence" -> "1.1. Artificial Intelligence"
  "1.1.1. Python"             -> "1.2.1. Python"
  "1.2.2.1. Computer Vision"  -> "1.1.2.1. Computer Vision"

Only modifies tags: and categories: lines inside YAML front matter.
"""
import re
import glob
import sys

TMP = '@@SWAPMARK@@'

# Match the leading "1.1." or "1.2." of a tag entry, anchored on the
# preceding "[" or "," boundary (with optional whitespace). The lookbehind
# is a fixed single char; whitespace is captured in group 1 so it round-trips.
RE_11 = re.compile(r'(?<=[\[,])(\s*)1\.1\.')
RE_12 = re.compile(r'(?<=[\[,])(\s*)1\.2\.')
RE_TMP = re.compile(r'(?<=[\[,])(\s*)1\.' + TMP + r'\.')


def swap_tag_line(line: str) -> str:
    # Three-pass swap using a temp marker to avoid the obvious 1.1 -> 1.2 -> 1.1 trap.
    line = RE_11.sub(lambda m: f'{m.group(1)}1.{TMP}.', line)
    line = RE_12.sub(lambda m: f'{m.group(1)}1.1.', line)
    line = RE_TMP.sub(lambda m: f'{m.group(1)}1.2.', line)
    return line


def process_file(path: str, dry_run: bool = False) -> bool:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.startswith('---'):
        return False

    # Split into front matter + body. content.split('---', 2) -> ['', fm, body]
    parts = content.split('---', 2)
    if len(parts) < 3:
        return False

    _, fm, body = parts
    new_fm_lines = []
    changed = False
    for line in fm.split('\n'):
        stripped = line.lstrip()
        if stripped.startswith('tags:') or stripped.startswith('categories:'):
            new_line = swap_tag_line(line)
            if new_line != line:
                changed = True
                if dry_run:
                    print(f'  - {line.rstrip()}')
                    print(f'  + {new_line.rstrip()}')
            new_fm_lines.append(new_line)
        else:
            new_fm_lines.append(line)

    if not changed:
        return False

    new_content = '---' + '\n'.join(new_fm_lines) + '---' + body
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    return True


def main() -> int:
    dry = '--dry-run' in sys.argv
    posts = glob.glob('_posts/**/*.md', recursive=True)
    changed_count = 0
    for post in sorted(posts):
        if process_file(post, dry_run=dry):
            changed_count += 1
            print(f'{"[DRY] " if dry else ""}changed: {post}')
    print(f'\nTotal {"would change" if dry else "changed"}: {changed_count} file(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
