"""
Strip category-mirror tags (e.g. "1.1. Artificial Intelligence") from every
post's `tags:` front-matter list. Tags now should be content keywords only;
the category hierarchy lives in `categories:` (populated from the directory
path by populate_categories_from_path.py).

A "category-mirror tag" is any tag whose value starts with a numeric prefix
like "1.", "1.1.", "1.1.2.", etc. Non-numeric tags (e.g. "JB", "GAN") are
kept as-is. If every tag gets stripped, the `tags:` line is set to `[]`.

Idempotent: re-running on an already-cleaned file is a no-op.
"""
import os
import re
import sys

POSTS_ROOT = '_posts'

# Matches "1.", "1.1.", "1.1.2.", ... followed by a space and a name.
NUMERIC_TAG_RE = re.compile(r'^\d+(?:\.\d+)*\.\s+.+$')


def split_tags(value: str) -> list[str]:
    """Parse a YAML inline list like '[a, b, "c, d"]' into trimmed strings."""
    inner = value.strip()
    if inner.startswith('[') and inner.endswith(']'):
        inner = inner[1:-1]
    # Naive split on commas — none of the existing tags contain commas.
    parts = [p.strip().strip('"').strip("'") for p in inner.split(',')]
    return [p for p in parts if p]


def format_tags(tags: list[str]) -> str:
    return '[' + ', '.join(tags) + ']'


def process_file(path: str, dry: bool) -> tuple[bool, list[str], list[str]]:
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    if not content.startswith('---'):
        return False, [], []
    parts = content.split('---', 2)
    if len(parts) < 3:
        return False, [], []
    _, fm, body = parts

    new_lines = []
    changed = False
    removed_all: list[str] = []
    kept_all: list[str] = []

    for line in fm.split('\n'):
        stripped = line.lstrip()
        if stripped.startswith('tags:'):
            indent = line[:len(line) - len(stripped)]
            value = stripped[len('tags:'):].strip()
            # Only handle inline-list form (which is what every post uses).
            if not (value.startswith('[') and value.endswith(']')):
                new_lines.append(line)
                continue
            tags = split_tags(value)
            kept = [t for t in tags if not NUMERIC_TAG_RE.match(t)]
            removed = [t for t in tags if NUMERIC_TAG_RE.match(t)]
            if removed:
                changed = True
                removed_all.extend(removed)
                kept_all.extend(kept)
                new_lines.append(f'{indent}tags: {format_tags(kept)}')
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if not changed:
        return False, [], []

    new_content = '---' + '\n'.join(new_lines) + '---' + body
    if not dry:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    return True, removed_all, kept_all


def main():
    dry = '--dry-run' in sys.argv
    files_changed = 0
    tags_removed = 0
    sample = []

    for root, _dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            path = os.path.join(root, f)
            changed, removed, kept = process_file(path, dry)
            if changed:
                files_changed += 1
                tags_removed += len(removed)
                if len(sample) < 5:
                    sample.append((path, removed, kept))

    print(f'{"Would update" if dry else "Updated"} {files_changed} file(s); '
          f'{tags_removed} category-mirror tag(s) stripped.')
    if sample:
        print('\nSample:')
        for path, removed, kept in sample:
            print(f'  {path}')
            print(f'    removed: {removed}')
            print(f'    kept:    {kept}')


if __name__ == '__main__':
    main()
