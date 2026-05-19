"""
Apply a {path: [tag1, tag2, ...]} mapping (from tags_mapping.json) to the
`tags:` line of each post's front matter.

Idempotent: re-running with the same mapping is a no-op (writes the same line).
Posts without an entry in the mapping are left untouched.
"""
import json
import os
import sys

MAPPING = os.path.join('_scripts', 'tags_mapping.json')


def format_tags(tags):
    return '[' + ', '.join(tags) + ']'


def apply_to_file(path: str, tags: list[str], dry: bool) -> bool:
    if not os.path.exists(path):
        print(f'  MISSING: {path}')
        return False
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    if not content.startswith('---'):
        return False
    parts = content.split('---', 2)
    if len(parts) < 3:
        return False
    _, fm, body = parts

    new_lines = []
    replaced = False
    target = f'tags: {format_tags(tags)}'
    for line in fm.split('\n'):
        s = line.lstrip()
        if s.startswith('tags:'):
            indent = line[:len(line) - len(s)]
            new_lines.append(f'{indent}{target}')
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        # Insert after categories: if present, else after layout:
        out = []
        inserted = False
        for line in new_lines:
            out.append(line)
            s = line.lstrip()
            if not inserted and (s.startswith('categories:') or s.startswith('layout:')):
                indent = line[:len(line) - len(s)]
                out.append(f'{indent}{target}')
                inserted = True
        new_lines = out

    new_content = '---' + '\n'.join(new_lines) + '---' + body
    if not dry:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    return True


def main():
    dry = '--dry-run' in sys.argv
    with open(MAPPING, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    n = 0
    for path, tags in mapping.items():
        if apply_to_file(path, tags, dry):
            n += 1
    print(f'{"Would update" if dry else "Updated"} {n} of {len(mapping)} files.')


if __name__ == '__main__':
    main()
