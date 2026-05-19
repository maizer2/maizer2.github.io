"""
Strip the leading "(...)" category marker from post filenames.

Examples:
  2022-05-18-(GAN)DCGAN-translation.md -> 2022-05-18-DCGAN-translation.md
  2023-02-01-(diffusion)stable-diffusion.md -> 2023-02-01-stable-diffusion.md

Also updates tags_mapping.json so the mapping keys stay valid.
URLs in post bodies use the slug after the date — the slug previously
contained "(GAN)" etc. — so we also rewrite any internal links of the
form /YYYY/MM/DD/(prefix)slug.html to drop the (prefix).

Idempotent: re-running on already-stripped files is a no-op.
"""
import json
import os
import re
import shutil
import sys

POSTS_ROOT = '_posts'
MAPPING_PATH = os.path.join('_scripts', 'tags_mapping.json')

# Match a YYYY-MM-DD- prefix followed by "(...)" then optional separator.
FILENAME_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}-)\([^)]+\)-?')

# Match URLs in post bodies that contain "/YYYY/MM/DD/(prefix)slug..."
URL_PREFIX_RE = re.compile(
    r'(/\d{4}/\d{2}/\d{2}/)\([^)]+\)-?'
)


def new_name(old: str) -> str:
    return FILENAME_RE.sub(r'\1', old)


def main():
    dry = '--dry-run' in sys.argv
    renames = []  # (old_path, new_path)

    for root, _dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            new_f = new_name(f)
            if new_f == f:
                continue
            old_path = os.path.join(root, f)
            new_path = os.path.join(root, new_f)
            if os.path.exists(new_path):
                print(f'  SKIP (target exists): {new_path}')
                continue
            renames.append((old_path, new_path))

    print(f'Planned renames: {len(renames)}')

    if dry:
        for old, new in renames[:20]:
            print(f'  {os.path.basename(old)}\n    -> {os.path.basename(new)}')
        if len(renames) > 20:
            print(f'  ... +{len(renames) - 20} more')
    else:
        for old, new in renames:
            shutil.move(old, new)

    # Rewrite tags_mapping.json keys
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, 'r', encoding='utf-8') as fp:
            mapping = json.load(fp)
        new_mapping = {}
        for k, v in mapping.items():
            # Apply same strip to the filename portion of the key
            d, base = os.path.split(k)
            new_mapping[os.path.join(d, new_name(base)).replace('\\', '/')] = v
        if not dry:
            with open(MAPPING_PATH, 'w', encoding='utf-8') as fp:
                json.dump(new_mapping, fp, ensure_ascii=False, indent=1)
            print(f'updated mapping: {MAPPING_PATH} ({len(new_mapping)} entries)')

    # Rewrite internal URL references inside post bodies
    url_subs = 0
    files_touched = 0
    for root, _dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8-sig') as fp:
                content = fp.read()
            new_content, n = URL_PREFIX_RE.subn(r'\1', content)
            if n > 0:
                url_subs += n
                files_touched += 1
                if not dry:
                    with open(path, 'w', encoding='utf-8') as fp:
                        fp.write(new_content)
    print(f'URL rewrites: {url_subs} in {files_touched} file(s)')


if __name__ == '__main__':
    main()
