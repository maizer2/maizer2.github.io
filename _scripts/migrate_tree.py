"""
Migrate every post's tags to the new proposed category tree.

What this does:
  1. Renames tags per TAG_MAP (e.g., 'a.a. Pytorch' -> '1.1.3.1. PyTorch').
  2. Ensures parent tags exist on each post so the tree has named intermediate
     nodes (e.g., adding '1.1.3. Frameworks' when '1.1.3.1. PyTorch' is present).
  3. Adds file-pattern-based subcategory tags for Computer Vision posts:
     - filename "..."      -> 1.1.2.1.1. GAN
     - filename "..."    -> 1.1.2.1.2. VITON
     - filename matches diffusion/DPM/DDPM/SBGM/NFD -> 1.1.2.1.3. Diffusion
  4. Strips UTF-8 BOM and fixes 'tag:' (singular) typos in YAML front matter.
  5. Dedupes and writes tags back, preserving any tag not in the map.

Safe to re-run (idempotent — already-migrated tags pass through unchanged).
"""
import os
import re
import sys
import glob

# ---- Tag rename map -------------------------------------------------------

TAG_MAP = {
    # Paper review (1.0. trick) -> proper home
    '1.0. Paper Review': '1.1.4. Paper Reviews',
    '2.0. Paper Review': '2.3. Paper Reviews',

    # 1.3-1.7 dispersed infra topics -> consolidated under 1.3. DevOps & Infra
    '1.3. Git':           '1.3.1. Git',
    '1.3.1. GitHub':      '1.3.1.1. GitHub',
    '1.3.2. GitBlog':     '1.3.1.2. GitBlog',
    '1.4. OS':            '1.3.2. OS',
    '1.4.1. Linux':       '1.3.2.1. Linux',
    '1.5. Container':     '1.3.3. Container',
    '1.5.1. Docker':      '1.3.3.1. Docker',
    '1.6. Security':      '1.3.5. Security',
    '1.7. Network':       '1.3.4. Network',

    # Frameworks (a.a.) -> AI > Frameworks (or Programming > Python for Pandas)
    'a.a. Pytorch':    '1.1.3.1. PyTorch',
    'a.a. TensorFlow': '1.1.3.2. TensorFlow',
    'a.a. OpenCV':     '1.1.3.3. OpenCV',
    'a.a. Pandas':     '1.2.1.1. Pandas',

    # ML concepts (a.b.) -> AI > ML > <concept>
    'a.b. Supervised Learning':      '1.1.1.1. Supervised Learning',
    'a.b. UnSupervised Learning':    '1.1.1.2. Unsupervised Learning',
    'a.b. SemiSupervised Learning':  '1.1.1.3. Semi-Supervised Learning',
    'a.b. Reinforcement Learning':   '1.1.1.4. Reinforcement Learning',
    'a.b. Regression Problem':       '1.1.1.5. Regression',
    'a.b. Binary Decision':          '1.1.1.6. Classification',

    # Other a.* tools
    'a.c. Latex': '1.2.3. Latex',
    'a.f. Hash':  '1.3.5.1. Hash',

    # Workplaces (f.x.) -> 6.1.x under Work / Career
    'f.a. ETRI': '6.1.1. ETRI',
    'f.b. KIDA': '6.1.2. KIDA',

    # Math renaming (drop "Mathematical " prefix repetition, fix capitalization)
    '2.1. Pure mathematics':            '2.1. Pure Mathematics',
    '2.1.1. Mathematical analysis':     '2.1.1. Analysis',
    '2.2.1. Mathematical Optimization': '2.2.1. Optimization',
    '2.2.2. Mathematical Statistics':   '2.2.2. Statistics',

    # Medicine: lift 4.1. Brain into a proper 4.1. Anatomy subtree
    '4.1. Brain':                       '4.1.1. Brain',
    'd.a. Endocrine':                   '4.1.2. Endocrine',
    'd.a. Autonomic nervous systems':   '4.1.3. Autonomic Nervous Systems',

    # Etc (6.x): translate to English + fold 6.3 into 6.2
    '6.1. 회사에서':         '6.1. Work / Career',
    '6.2 삶을 살아가는 태도':  '6.2. Life Reflections',   # original is missing the dot
    '6.2. 삶을 살아가는 태도': '6.2. Life Reflections',   # in case typo got fixed
    '6.3. 끄적임':           '6.2. Life Reflections',
    '6.4. 석사':             '6.3. Studies',
}

# ---- Parent tags to auto-add ---------------------------------------------

# When the KEY tag is present on a post, ensure all VALUES are also present.
# This guarantees the tree shows named intermediate nodes.
PARENT_TAGS = {
    # 1.1. Artificial Intelligence subtree
    '1.1.4. Paper Reviews':              ['1.1. Artificial Intelligence'],
    '1.1.3.1. PyTorch':                  ['1.1.3. Frameworks', '1.1. Artificial Intelligence'],
    '1.1.3.2. TensorFlow':               ['1.1.3. Frameworks', '1.1. Artificial Intelligence'],
    '1.1.3.3. OpenCV':                   ['1.1.3. Frameworks', '1.1. Artificial Intelligence'],
    '1.1.3. Frameworks':                 ['1.1. Artificial Intelligence'],
    '1.1.1.1. Supervised Learning':      ['1.1.1. Machine Learning', '1.1. Artificial Intelligence'],
    '1.1.1.2. Unsupervised Learning':    ['1.1.1. Machine Learning', '1.1. Artificial Intelligence'],
    '1.1.1.3. Semi-Supervised Learning': ['1.1.1. Machine Learning', '1.1. Artificial Intelligence'],
    '1.1.1.4. Reinforcement Learning':   ['1.1.1. Machine Learning', '1.1. Artificial Intelligence'],
    '1.1.1.5. Regression':               ['1.1.1. Machine Learning', '1.1. Artificial Intelligence'],
    '1.1.1.6. Classification':           ['1.1.1. Machine Learning', '1.1. Artificial Intelligence'],
    '1.1.2.1.1. GAN':                    ['1.1.2.1. Computer Vision', '1.1.2. Deep Learning', '1.1. Artificial Intelligence'],
    '1.1.2.1.2. VITON':                  ['1.1.2.1. Computer Vision', '1.1.2. Deep Learning', '1.1. Artificial Intelligence'],
    '1.1.2.1.3. Diffusion':              ['1.1.2.1. Computer Vision', '1.1.2. Deep Learning', '1.1. Artificial Intelligence'],

    # 1.2. Programming subtree
    '1.2.1.1. Pandas':                   ['1.2.1. Python', '1.2. Programming'],
    '1.2.3. Latex':                      ['1.2. Programming'],

    # 1.3. DevOps & Infra subtree (was: scattered 1.3-1.7)
    '1.3.1. Git':                        ['1.3. DevOps & Infra'],
    '1.3.1.1. GitHub':                   ['1.3.1. Git', '1.3. DevOps & Infra'],
    '1.3.1.2. GitBlog':                  ['1.3.1. Git', '1.3. DevOps & Infra'],
    '1.3.2. OS':                         ['1.3. DevOps & Infra'],
    '1.3.2.1. Linux':                    ['1.3.2. OS', '1.3. DevOps & Infra'],
    '1.3.3. Container':                  ['1.3. DevOps & Infra'],
    '1.3.3.1. Docker':                   ['1.3.3. Container', '1.3. DevOps & Infra'],
    '1.3.4. Network':                    ['1.3. DevOps & Infra'],
    '1.3.5. Security':                   ['1.3. DevOps & Infra'],
    '1.3.5.1. Hash':                     ['1.3.5. Security', '1.3. DevOps & Infra'],

    # 2. Mathematics subtree
    '2.3. Paper Reviews':                [],

    # 4. Medicine subtree
    '4.1.1. Brain':                      ['4.1. Anatomy'],
    '4.1.2. Endocrine':                  ['4.1. Anatomy'],
    '4.1.3. Autonomic Nervous Systems':  ['4.1. Anatomy'],

    # 6. Etc subtree
    '6.1.1. ETRI':                       ['6.1. Work / Career'],
    '6.1.2. KIDA':                       ['6.1. Work / Career'],
}

# ---- File-pattern subcategorization (Computer Vision papers) -------------

CV_SUBCAT_RULES = [
    # (regex on filename, new sub-category tag)
    (re.compile(r'\(GAN\)', re.IGNORECASE),                                       '1.1.2.1.1. GAN'),
    (re.compile(r'\(VITON\)', re.IGNORECASE),                                     '1.1.2.1.2. VITON'),
    (re.compile(r'\(diffusion\)|\(DPM\)|DDPM|SBGM|\bNFD\b|blog-trans\)diffusion', re.IGNORECASE), '1.1.2.1.3. Diffusion'),
]


def detect_cv_subcat_tag(filename: str) -> str | None:
    for rx, tag in CV_SUBCAT_RULES:
        if rx.search(filename):
            return tag
    return None


# ---- Tag list transformer ------------------------------------------------

def natural_key(tag: str):
    """Sort key: split prefix into uniform (kind, value) pairs for stable comparison."""
    m = re.match(r'^([^\s]+?)\.\s+(.+)$', tag.strip())
    if not m:
        return ((2, tag),)  # malformed tags sink to the bottom
    prefix, name = m.group(1), m.group(2)
    parts = []
    for p in prefix.split('.'):
        try:
            parts.append((0, int(p), ''))   # numeric segments first, integer-ordered
        except ValueError:
            parts.append((1, 0, p))         # alpha segments after, lexically ordered
    parts.append((2, 0, name))
    return tuple(parts)


def transform_tags(tags: list[str], filename: str) -> list[str]:
    new_set = set()
    for t in tags:
        t = t.strip()
        if not t:
            continue
        new_set.add(TAG_MAP.get(t, t))

    # File-pattern subcategorization (if post is about CV)
    cv_tag = detect_cv_subcat_tag(filename)
    if cv_tag:
        new_set.add(cv_tag)

    # Add parent tags transitively (fixed-point loop, but bounded by depth)
    for _ in range(10):
        added = False
        for tag in list(new_set):
            for parent in PARENT_TAGS.get(tag, []):
                if parent not in new_set:
                    new_set.add(parent)
                    added = True
        if not added:
            break

    return sorted(new_set, key=natural_key)


# ---- Per-file processing -------------------------------------------------

BOM = '﻿'


def process_file(path: str, dry_run: bool = False) -> bool:
    with open(path, 'r', encoding='utf-8-sig') as f:  # strips BOM on read
        content = f.read()

    # Defensive: also strip any residual BOM
    if content.startswith(BOM):
        content = content[len(BOM):]

    if not content.startswith('---'):
        return False
    parts = content.split('---', 2)
    if len(parts) < 3:
        return False
    _, fm, body = parts

    fname = os.path.basename(path)
    new_lines = []
    changed = False
    found_tags = False

    for line in fm.split('\n'):
        stripped = line.lstrip()

        # Fix 'tag:' (singular) typo
        if stripped.startswith('tag:') and not stripped.startswith('tags:'):
            indent_len = len(line) - len(stripped)
            line = line[:indent_len] + 'tags:' + stripped[len('tag:'):]
            stripped = line.lstrip()
            changed = True

        if stripped.startswith('tags:'):
            found_tags = True
            m = re.match(r'^(\s*tags:\s*\[)(.*?)(\]\s*)$', line)
            if m:
                prefix, items_str, suffix = m.group(1), m.group(2), m.group(3)
                items = [t.strip() for t in items_str.split(',') if t.strip()]
                new_items = transform_tags(items, fname)
                new_line = prefix + ', '.join(new_items) + suffix
                if new_line != line:
                    changed = True
                    if dry_run:
                        print(f'  - {line.rstrip()}')
                        print(f'  + {new_line.rstrip()}')
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # If the post has no tags: line but the filename pattern suggests a CV subcat,
    # add a tags line so it shows up in the new subtree.
    if not found_tags:
        cv_tag = detect_cv_subcat_tag(fname)
        if cv_tag:
            tags_list = transform_tags([], fname)
            out_lines2 = []
            inserted = False
            for line in new_lines:
                out_lines2.append(line)
                if not inserted and line.lstrip().startswith('categories:'):
                    indent = line[:len(line) - len(line.lstrip())]
                    out_lines2.append(f'{indent}tags: [{", ".join(tags_list)}]')
                    inserted = True
            new_lines = out_lines2
            changed = True

    if not changed:
        # Still rewrite to drop the BOM if we read one. Otherwise skip.
        with open(path, 'rb') as f:
            raw_first = f.read(3)
        if raw_first == b'\xef\xbb\xbf':
            if not dry_run:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
            return True
        return False

    new_content = '---' + '\n'.join(new_lines) + '---' + body
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    return True


# ---- Main ----------------------------------------------------------------

def main() -> int:
    dry = '--dry-run' in sys.argv
    posts = sorted(glob.glob('_posts/**/*.md', recursive=True))
    changed = 0
    for p in posts:
        if process_file(p, dry_run=dry):
            changed += 1
            if dry:
                print(f'[DRY] would change: {p}')
            else:
                print(f'changed: {p}')
    print(f'\nTotal {"would change" if dry else "changed"}: {changed} / {len(posts)} file(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
