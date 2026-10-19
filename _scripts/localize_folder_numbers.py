"""
Convert category folder names from GLOBAL dotted numbering to LOCAL numbering.

Before:  _posts/1. Computer Engineering/1.1. Artificial Intelligence/1.1.2. Deep Learning/
After:   _posts/1. Computer Engineering/1. Artificial Intelligence/2. Deep Learning/

Only the LAST numeric segment of each folder name is kept; the parent prefix
is dropped. The hierarchy itself is unchanged, so `categories:` (derived from
the path by populate_categories_from_path.py) and the category tree keep the
same shape — just with shorter names.

Idempotent: folders that are already local ("2. Deep Learning") are untouched.
Renames bottom-up (deepest first) so parent paths stay valid during the walk.

Usage:
    python _scripts/localize_folder_numbers.py [--dry-run]
"""
import os
import re
import sys

POSTS_ROOT = '_posts'
NAME_RE = re.compile(r'^(\d+(?:\.\d+)*)\. (.+)$')


def local_name(dirname: str) -> str:
    m = NAME_RE.match(dirname)
    if not m:
        return dirname
    prefix, title = m.group(1), m.group(2)
    return f'{prefix.split(".")[-1]}. {title}'


def plan_renames() -> list[tuple[str, str]]:
    renames = []
    for root, dirs, _ in os.walk(POSTS_ROOT, topdown=False):
        seen = {}
        for d in dirs:
            new = local_name(d)
            if new in seen and seen[new] != d:
                sys.exit(f'COLLISION in {root}: "{seen[new]}" and "{d}" both -> "{new}"')
            seen[new] = d
            if new != d:
                renames.append((os.path.join(root, d), os.path.join(root, new)))
    return renames


def main():
    dry = '--dry-run' in sys.argv
    renames = plan_renames()
    for src, dst in renames:
        print(f'{"[DRY] " if dry else ""}{src}\n    -> {dst}')
        if not dry:
            if os.path.exists(dst):
                sys.exit(f'Target already exists: {dst}')
            os.rename(src, dst)
    print(f'\n{len(renames)} folder(s) {"would be" if dry else ""} renamed')


if __name__ == '__main__':
    main()
