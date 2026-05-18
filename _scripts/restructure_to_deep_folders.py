"""
Restructure _posts/ from a flat-by-top-category layout to a DEEP folder
hierarchy that mirrors the category tree.

After this script:
  - Each file lives at _posts/<L1>/<L2>/.../<leaf>/file.md
    e.g. _posts/1. Computer Engineering/1.1. Artificial Intelligence/
         1.1.2. Deep Learning/1.1.2.1. Computer Vision/
         1.1.2.1.1. GAN/(GAN)Foo.md
  - Front-matter `categories:` is removed (Jekyll auto-derives from the path)
  - The deepest tag (computed via heuristic) determines the file's leaf folder

A separate script handles permalink config + internal-link rewriting, since
that part is decoupled.

Run via WSL (`6. etc.` trailing-dot dir requires it on Windows).
"""
import os
import re
import shutil
import sys

POSTS_ROOT = '_posts'

# ---- Folder name sanitizer ------------------------------------------------

def sanitize_folder(name: str) -> str:
    """Filesystem-safe folder name. Replace '/' (path sep) with '&'."""
    return name.replace('/', '&').rstrip()


# ---- Tag parsing & registry ------------------------------------------------

def parse_tag(tag: str):
    m = re.match(r'^\s*([^\s]+?)\.\s+(.+)\s*$', tag.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def read_front_matter(path: str):
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    if not content.startswith('---'):
        return [], [], content
    parts = content.split('---', 2)
    if len(parts) < 3:
        return [], [], content
    fm = parts[1]
    tags, cats = [], []
    for line in fm.split('\n'):
        s = line.lstrip()
        if s.startswith('tags:'):
            m = re.match(r'^\s*tags:\s*\[(.*?)\]', line)
            if m:
                tags = [t.strip() for t in m.group(1).split(',') if t.strip()]
        elif s.startswith('categories:'):
            m = re.match(r'^\s*categories:\s*\[(.*?)\]', line)
            if m:
                cats = [t.strip() for t in m.group(1).split(',') if t.strip()]
    return tags, cats, content


def build_registry() -> dict:
    """Scan all posts to map prefix -> full sanitized tag name.
       e.g. {'1.1.2': '1.1.2. Deep Learning'}
    """
    registry = {}
    for root, dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            tags, cats, _ = read_front_matter(os.path.join(root, f))
            for item in tags + cats:
                parsed = parse_tag(item)
                if parsed:
                    prefix, _ = parsed
                    if prefix not in registry:
                        registry[prefix] = sanitize_folder(item)
    return registry


# ---- Leaf determination (same heuristic used previously) -------------------

FILENAME_RULES = [
    (re.compile(r'\(GAN\)', re.I),                                                 '1.1.2.1.1. GAN'),
    (re.compile(r'\(VITON\)', re.I),                                               '1.1.2.1.2. VITON'),
    (re.compile(r'\(diffusion\)|\(DPM\)|DDPM|SBGM|\bNFD\b|blog-trans\)diffusion', re.I), '1.1.2.1.3. Diffusion'),
    (re.compile(r'\(cnn\)', re.I),                                                 '1.1.2.1. Computer Vision'),
    (re.compile(r'\(OpenCV-by-Python\)', re.I),                                    '1.1.3.3. OpenCV'),
    (re.compile(r'PyTorch.*translation|torchvision', re.I),                        '1.1.3.1. PyTorch'),
    (re.compile(r'\(ETRI-Research-Student\)', re.I),                               '6.1.1. ETRI'),
    (re.compile(r'\(KIDA\)', re.I),                                                '6.1.2. KIDA'),
    (re.compile(r'\(Masters-Life\)', re.I),                                        '6.3. Studies'),
    (re.compile(r'\(English-Conversation\)|my-grammar-coach', re.I),               '3.2. English'),
]


def is_ancestor(ancestor_parts, descendant_parts):
    return len(ancestor_parts) < len(descendant_parts) and \
        descendant_parts[:len(ancestor_parts)] == ancestor_parts


def determine_leaf(path, tags, categories):
    fname = os.path.basename(path)

    # 1) BookReview: prefer the 5.X book tag
    if '5. BookReview' in path:
        for t in tags:
            if re.match(r'5\.\d+\.', t):
                return t

    # 2) Filename pattern
    for rx, mapped in FILENAME_RULES:
        if rx.search(fname):
            return mapped

    # 3) Deepest leaf among the post's own tags
    parsed = []
    for t in tags:
        p = parse_tag(t)
        if p:
            parsed.append((t, p[0].split('.')))

    leaves = []
    for tag, prefix in parsed:
        leaf = True
        for other_tag, other_prefix in parsed:
            if other_tag == tag:
                continue
            if is_ancestor(prefix, other_prefix):
                leaf = False
                break
        if leaf:
            leaves.append((tag, prefix))

    if leaves:
        leaves.sort(key=lambda x: (-len(x[1]), x[0]))
        return leaves[0][0]

    # 4) Fallback: top-level category
    return categories[0] if categories else None


# ---- Build target folder path from a leaf tag -------------------------------

def build_target_dir(leaf_tag, registry):
    parsed = parse_tag(leaf_tag)
    if not parsed:
        return None
    prefix, _ = parsed
    parts = prefix.split('.')
    folders = []
    for i in range(len(parts)):
        key = '.'.join(parts[:i + 1])
        folder = registry.get(key)
        if not folder:
            # Fallback if no entry exists for this intermediate prefix
            folder = key + '.'
        folders.append(folder)
    return os.path.join(POSTS_ROOT, *folders)


# ---- Front matter: remove categories: line (Jekyll will derive from path) ----

def strip_categories_from_fm(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    if not content.startswith('---'):
        return False
    parts = content.split('---', 2)
    if len(parts) < 3:
        return False
    _, fm, body = parts
    new_lines = []
    removed = False
    for line in fm.split('\n'):
        if line.lstrip().startswith('categories:'):
            removed = True
            continue
        new_lines.append(line)
    if not removed:
        return False
    new_content = '---' + '\n'.join(new_lines) + '---' + body
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


# ---- Main ------------------------------------------------------------------

def main():
    dry = '--dry-run' in sys.argv

    registry = build_registry()
    print(f'Registry built: {len(registry)} entries')

    moves = []
    skip_unchanged = 0
    for root, dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            path = os.path.join(root, f)
            tags, cats, _ = read_front_matter(path)
            leaf = determine_leaf(path, tags, cats)
            if not leaf:
                continue
            target_dir = build_target_dir(leaf, registry)
            if not target_dir:
                continue
            target_path = os.path.join(target_dir, f)
            if os.path.normpath(path) == os.path.normpath(target_path):
                skip_unchanged += 1
                continue
            moves.append((path, target_path, leaf))

    print(f'\nMoves planned: {len(moves)} (already in place: {skip_unchanged})')

    if dry:
        for old, new, leaf in moves[:15]:
            print(f'  [{leaf}]')
            print(f'    {old}')
            print(f'    -> {new}')
        if len(moves) > 15:
            print(f'  ... and {len(moves) - 15} more')
        return

    # Execute moves
    print('\n=== Moving files ===')
    for old, new, _ in moves:
        os.makedirs(os.path.dirname(new), exist_ok=True)
        shutil.move(old, new)
    print(f'Moved {len(moves)} files.')

    # Strip front-matter `categories:`
    print('\n=== Stripping front-matter categories ===')
    stripped = 0
    for root, dirs, fs in os.walk(POSTS_ROOT):
        for f in fs:
            if not f.endswith('.md'):
                continue
            if strip_categories_from_fm(os.path.join(root, f)):
                stripped += 1
    print(f'Stripped categories from {stripped} files.')

    # Cleanup: remove now-empty old top-level dirs
    print('\n=== Removing empty old dirs ===')
    for entry in os.listdir(POSTS_ROOT):
        d = os.path.join(POSTS_ROOT, entry)
        if os.path.isdir(d):
            try:
                # Only remove if empty (won't remove dirs that now contain subdirs)
                if not os.listdir(d):
                    os.rmdir(d)
                    print(f'  removed empty: {d}')
            except OSError:
                pass


if __name__ == '__main__':
    main()
