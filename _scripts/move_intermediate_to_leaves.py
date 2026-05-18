"""
Move every post that's currently in an INTERMEDIATE directory (one that also
contains subdirectories) into a LEAF directory.

Routing:
  1. Filename matches a known specific topic -> targeted leaf
     (GAN-related -> 1.1.2.1.1. GAN, etc.)
  2. Otherwise -> a "<prefix>.0. General" leaf created under the current
     intermediate dir.

After moves, run populate_categories_from_path.py to refresh front-matter
`categories:` to match the new paths.

URLs are now date-based, so moving doesn't break internal links.
"""
import os
import re
import shutil
import sys

POSTS_ROOT = '_posts'

# (filename regex, target leaf path relative to POSTS_ROOT)
SPECIFIC_ROUTES = [
    # CV intermediate -> specific sub-leaves
    (re.compile(r'GAN-glossary|GAN.*workflow|Gan.*objective|load-map-for-GAN|paper-of-GAN', re.I),
     '1. Computer Engineering/1.1. Artificial Intelligence/1.1.2. Deep Learning/1.1.2.1. Computer Vision/1.1.2.1.1. GAN'),
    (re.compile(r'Markov-chain|mathematical-theory-for-Diffusion', re.I),
     '1. Computer Engineering/1.1. Artificial Intelligence/1.1.2. Deep Learning/1.1.2.1. Computer Vision/1.1.2.1.3. Diffusion'),
    (re.compile(r'DeepFashion-Database', re.I),
     '1. Computer Engineering/1.1. Artificial Intelligence/1.1.2. Deep Learning/1.1.2.1. Computer Vision/1.1.2.1.2. VITON'),

    # Python intermediate -> actually about Linux / Docker
    (re.compile(r'libgl1-mesa-glx|libgthread', re.I),
     '1. Computer Engineering/1.3. DevOps & Infra/1.3.2. OS/1.3.2.1. Linux'),
    (re.compile(r'docker-cp-command', re.I),
     '1. Computer Engineering/1.3. DevOps & Infra/1.3.3. Container/1.3.3.1. Docker'),
]


def parse_prefix(folder_basename):
    """'1.1. Artificial Intelligence' -> '1.1'"""
    m = re.match(r'^([\d]+(?:\.\d+)*)\.\s+', folder_basename)
    return m.group(1) if m else None


def general_leaf_path(intermediate_dir):
    """Build the path for a '<prefix>.0. General' leaf under an intermediate dir."""
    basename = os.path.basename(intermediate_dir.rstrip('/'))
    prefix = parse_prefix(basename)
    if prefix is None:
        return None
    leaf_prefix = f'{prefix}.0'
    return os.path.join(intermediate_dir, f'{leaf_prefix}. General')


def find_intermediate_posts():
    posts = []
    for root, dirs, fs in os.walk(POSTS_ROOT):
        if dirs and any(f.endswith('.md') for f in fs):
            for f in fs:
                if f.endswith('.md'):
                    posts.append((root, f))
    return posts


def route_target(filename, intermediate_dir):
    # Specific routes first
    for rx, target_rel in SPECIFIC_ROUTES:
        if rx.search(filename):
            return os.path.join(POSTS_ROOT, target_rel)
    # Default: <prefix>.0. General leaf under the intermediate dir
    return general_leaf_path(intermediate_dir)


def main():
    dry = '--dry-run' in sys.argv

    intermediate = find_intermediate_posts()
    print(f'Intermediate posts: {len(intermediate)}')

    moves = []
    for intermediate_dir, fname in intermediate:
        target_dir = route_target(fname, intermediate_dir)
        if not target_dir:
            print(f'  SKIP (no target): {intermediate_dir}/{fname}')
            continue
        src = os.path.join(intermediate_dir, fname)
        dst = os.path.join(target_dir, fname)
        if os.path.normpath(src) == os.path.normpath(dst):
            continue
        moves.append((src, dst))

    print(f'\nPlanned moves: {len(moves)}')
    if dry:
        for src, dst in moves[:25]:
            print(f'  {src}')
            print(f'    -> {dst}')
        if len(moves) > 25:
            print(f'  ... and {len(moves) - 25} more')
        return

    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f'moved: {os.path.basename(src)}  ->  {os.path.relpath(dst, POSTS_ROOT)}')

    print(f'\nDone. {len(moves)} files moved.')


if __name__ == '__main__':
    main()
