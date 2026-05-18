"""
Rename Korean-named .md files in _posts/ to English filenames (semantic
translation), and update internal links in all post bodies.

Run on WSL so _posts/6. etc./ (trailing-dot dir) is handled.

Front-matter `title:` is intentionally NOT changed (per user request).
"""
import os
import re
import sys

# Old filename (just the basename) -> new filename (basename, with .md)
# All files live somewhere under _posts/. Date prefix preserved.
RENAMES = {
    # ----- 1. Computer Engineering -----
    '2021-10-12-CMake는-무엇일까.md':                              '2021-10-12-What-is-CMake.md',
    '2021-10-20-(OpenCV-by-Python)-1-세팅.md':                     '2021-10-20-(OpenCV-by-Python)-1-Setup.md',
    '2021-10-21-(OpenCV-by-Python)-2-Lenna-출력하기.md':            '2021-10-21-(OpenCV-by-Python)-2-Display-Lenna.md',
    '2021-10-24-(OpenCV-by-Python)-3-동영상-출력하기.md':           '2021-10-24-(OpenCV-by-Python)-3-Display-Video.md',
    '2021-11-02-머신러닝에서-파이프라인이란.md':                     '2021-11-02-What-is-Pipeline-in-Machine-Learning.md',
    '2022-01-12-Git 체크섬,-Git-Checksum.md':                       '2022-01-12-Git-Checksum.md',
    '2022-01-14-인공지능에서-입력-데이터-스트림이란.md':              '2022-01-14-What-is-Input-Data-Stream-in-AI.md',
    '2022-01-15-비용함수.md':                                       '2022-01-15-Cost-Function.md',
    '2022-01-15-선형-회귀-알고리즘.md':                              '2022-01-15-Linear-Regression-Algorithm.md',
    '2022-01-15-인공지능에서-모델-파라미터란.md':                    '2022-01-15-What-is-Model-Parameter-in-AI.md',
    '2022-01-24-TensorFLow-데이터-표현.md':                          '2022-01-24-TensorFlow-Data-Representation.md',
    '2022-01-24-k-최근접-이웃-알고리즘.md':                          '2022-01-24-k-Nearest-Neighbors-Algorithm.md',
    '2022-01-24-지도-학습.md':                                      '2022-01-24-Supervised-Learning.md',
    '2022-01-25-Python-인스턴스와-객체의-차이점.md':                '2022-01-25-Python-Instance-vs-Object.md',
    '2022-02-01-비지도-학습.md':                                    '2022-02-01-Unsupervised-Learning.md',
    '2022-02-04-강화-학습.md':                                      '2022-02-04-Reinforcement-Learning.md',
    '2022-02-04-배치-학습.md':                                      '2022-02-04-Batch-Learning.md',
    '2022-02-04-준지도-학습.md':                                    '2022-02-04-Semi-Supervised-Learning.md',
    '2022-02-05-모델-기반-학습.md':                                 '2022-02-05-Model-Based-Learning.md',
    '2022-02-05-사례-기반-학습.md':                                 '2022-02-05-Instance-Based-Learning.md',
    '2022-02-06-k-l-겹-교차-검증.md':                               '2022-02-06-k-l-Fold-Cross-Validation.md',
    '2022-02-06-k-겹-교차-검증.md':                                 '2022-02-06-k-Fold-Cross-Validation.md',
    '2022-02-06-교차-검증.md':                                      '2022-02-06-Cross-Validation.md',
    '2022-02-06-홀드아웃-검증.md':                                  '2022-02-06-Holdout-Validation.md',
    '2022-02-07-머신러닝에서-컴포넌트란.md':                        '2022-02-07-What-is-Component-in-Machine-Learning.md',
    '2022-02-08-평균-제곱근-오차-RMSE.md':                          '2022-02-08-Root-Mean-Square-Error-RMSE.md',
    '2022-02-10-Gitblog-Latex-설정하기.md':                         '2022-02-10-Setup-Latex-on-Gitblog.md',
    '2022-02-11-평균-절대-오차-MAE.md':                             '2022-02-11-Mean-Absolute-Error-MAE.md',
    '2022-02-12-Pandas-기본.md':                                    '2022-02-12-Pandas-Basics.md',
    '2022-02-14-Python-용어집-해석.md':                             '2022-02-14-Python-Glossary-Translation.md',
    '2022-03-28-Kernel이란.md':                                     '2022-03-28-What-is-Kernel.md',
    '2022-04-04-Windows-Wsl2-docker-내부에-pytorch,-cuda-설치하기.md': '2022-04-04-Install-PyTorch-CUDA-in-Wsl2-Docker.md',
    '2022-04-04-유클리디안-공간.md':                                '2022-04-04-Euclidean-Space.md',
    '2022-04-08-제곱근-오차-MSE.md':                                '2022-04-08-Mean-Squared-Error-MSE.md',
    '2022-04-17-Python-복사에-대하여.md':                           '2022-04-17-About-Python-Copy.md',
    '2022-04-18-Chrome-좋아요-일괄-삭제.md':                        '2022-04-18-Chrome-Bulk-Delete-Likes.md',
    '2022-04-19-유클리드-공간에서-R의-뜻.md':                       '2022-04-19-Meaning-of-R-in-Euclidean-Space.md',

    # ----- 3. Language -----
    '2022-03-05-5언9품사.md':                                                  '2022-03-05-5-Words-9-Parts-of-Speech.md',
    '2022-04-24-(영어회화)-0.-Setting-my-goal.md':                              '2022-04-24-(English-Conversation)-0-Setting-my-goal.md',
    '2022-04-25-(영어회화)-1.-Let-me-introduce-myself.md':                      '2022-04-25-(English-Conversation)-1-Let-me-introduce-myself.md',
    '2022-04-26-(영어회화)-2.Explain-NN.md':                                    '2022-04-26-(English-Conversation)-2-Explain-NN.md',
    '2023-02-11-my-grammar-coach-standard-추가문법-1.md':                       '2023-02-11-my-grammar-coach-standard-additional-grammar-1.md',
    '2023-02-12-my-grammar-coach-standard-추가문법-2.md':                       '2023-02-12-my-grammar-coach-standard-additional-grammar-2.md',
    '2023-02-12-my-grammar-coach-standard-추가문법-3.md':                       '2023-02-12-my-grammar-coach-standard-additional-grammar-3.md',
    '2023-02-12-my-grammar-coach-standard-추가문법-4.md':                       '2023-02-12-my-grammar-coach-standard-additional-grammar-4.md',
    '2023-02-12-my-grammar-coach-standard-추가문법-5.md':                       '2023-02-12-my-grammar-coach-standard-additional-grammar-5.md',
    '2023-02-13-my-grammar-coach-standard-추가문법-6.md':                       '2023-02-13-my-grammar-coach-standard-additional-grammar-6.md',
    '2023-02-14-my-grammar-coach-standard-추가문법-7.md':                       '2023-02-14-my-grammar-coach-standard-additional-grammar-7.md',
    '2023-02-20-my-grammar-coach-standard-추가문법-8.md':                       '2023-02-20-my-grammar-coach-standard-additional-grammar-8.md',
    '2023-02-23-my-grammar-coach-standard-추가문법-10.md':                      '2023-02-23-my-grammar-coach-standard-additional-grammar-10.md',
    '2023-02-23-my-grammar-coach-standard-추가문법-9.md':                       '2023-02-23-my-grammar-coach-standard-additional-grammar-9.md',

    # ----- 5. BookReview -----
    '2021-07-07-(프로그래머를-위한-선형대수)-0.-서론.md':                                          '2021-07-07-(Linear-Algebra-for-Programmers)-0-Introduction.md',
    '2021-09-26-(OpenCV-4로-배우는-컴퓨터-비전과-머신-러닝)-0-서론.md':                            '2021-09-26-(OpenCV-4-CV-and-ML)-0-Introduction.md',
    '2021-09-27-(OpenCV-4로-배우는-컴퓨터-비전과-머신-러닝)-1-세팅.md':                            '2021-09-27-(OpenCV-4-CV-and-ML)-1-Setup.md',
    '2021-09-28-(OpenCV-4로-배우는-컴퓨터-비전과-머신-러닝)-2-Lenna 출력하기.md':                  '2021-09-28-(OpenCV-4-CV-and-ML)-2-Display-Lenna.md',
    '2022-01-13-(Hands-On-Machine-Learning-2)-0.-서론.md':                                          '2022-01-13-(Hands-On-Machine-Learning-2)-0-Introduction.md',
    '2022-01-13-(Hands-On-Machine-Learning-2)-1.-한눈에-보는-머신러닝.md':                          '2022-01-13-(Hands-On-Machine-Learning-2)-1-ML-at-a-glance.md',
    '2022-02-07-(Hands-On-Machine-Learning-2)-2.-머신러닝-프로젝트-처음부터-끝까지.md':              '2022-02-07-(Hands-On-Machine-Learning-2)-2-ML-Project-End-to-End.md',
    '2022-03-10-(실전-예제로-배우는-GAN)-0.-서론.md':                                                '2022-03-10-(Hands-on-GAN-Examples)-0-Introduction.md',
    '2022-03-13-(실전-예제로-배우는-GAN)-1.-생성적-적대-신경망이란.md':                              '2022-03-13-(Hands-on-GAN-Examples)-1-What-is-GAN.md',
    '2022-03-24-(프로그래머를-위한-선형대수)-1.-벡터,-행렬,-행렬식.md':                              '2022-03-24-(Linear-Algebra-for-Programmers)-1-Vectors-Matrices-Determinants.md',
    '2022-03-26-(한-걸음씩-알아가는-선형대수학)-0.-서론.md':                                         '2022-03-26-(Linear-Algebra-Step-by-Step)-0-Introduction.md',
    '2022-03-27-(한-걸음씩-알아가는-선형대수학)-7.-고윳값과-고유벡터.md':                            '2022-03-27-(Linear-Algebra-Step-by-Step)-7-Eigenvalues-and-Eigenvectors.md',
    '2022-03-28-(한-걸음씩-알아가는-선형대수학)-1.-일차방정식과-행렬.md':                            '2022-03-28-(Linear-Algebra-Step-by-Step)-1-Linear-Equations-and-Matrices.md',
    '2022-03-28-(한-걸음씩-알아가는-선형대수학)-6.-행렬식과-역행렬.md':                              '2022-03-28-(Linear-Algebra-Step-by-Step)-6-Determinants-and-Inverse-Matrices.md',
    '2022-04-03-(선형대수와-통계학으로-배우는-머신러닝-with-파이썬)-3.-머신러닝을-위한-선형대수.md':  '2022-04-03-(ML-with-Linear-Algebra-and-Stats-Python)-3-Linear-Algebra-for-ML.md',
    '2022-04-04-(파이토치-첫걸음)-0.-서론.md':                                                       '2022-04-04-(PyTorch-First-Steps)-0-Introduction.md',
    '2022-04-04-(파이토치-첫걸음)-1.-딥러닝에-대하여.md':                                            '2022-04-04-(PyTorch-First-Steps)-1-About-Deep-Learning.md',
    '2022-04-04-(파이토치-첫걸음)-2.-파이토치.md':                                                   '2022-04-04-(PyTorch-First-Steps)-2-PyTorch.md',
    '2022-04-04-(파이토치-첫걸음)-3.-선형회귀분석.md':                                               '2022-04-04-(PyTorch-First-Steps)-3-Linear-Regression-Analysis.md',
    '2022-04-10-(선형대수와-통계학으로-배우는-머신러닝-with-파이썬)-4.-머신러닝을-위한-통계학(1).md': '2022-04-10-(ML-with-Linear-Algebra-and-Stats-Python)-4-Statistics-for-ML-1.md',
    '2022-04-12-(선형대수와-통계학으로-배우는-머신러닝-with-파이썬)-4.-머신러닝을-위한-통계학(2).md': '2022-04-12-(ML-with-Linear-Algebra-and-Stats-Python)-4-Statistics-for-ML-2.md',
    '2022-04-13-(파이토치-첫걸음)-5.-합성곱-신경망.md':                                              '2022-04-13-(PyTorch-First-Steps)-5-Convolutional-Neural-Network.md',
    '2022-04-14-(혼공머신)-7.-인공신경망.md':                                                       '2022-04-14-(Honkong-Machine)-7-Artificial-Neural-Network.md',
    '2022-04-30-(미술관에-GAN-딥러닝-실전-프로젝트)-1.-생성-딥러닝을-소개합니다-1.md':              '2022-04-30-(GAN-Art-Gallery-Project)-1-Introducing-Generative-Deep-Learning-1.md',
    '2022-05-01-(미술관에-GAN-딥러닝-실전-프로젝트)-0.-서론.md':                                    '2022-05-01-(GAN-Art-Gallery-Project)-0-Introduction.md',

    # ----- 6. etc. -----
    '2021-07-01-이-블로그는-어떤곳인가요.md':                       '2021-07-01-About-this-blog.md',
    '2021-10-15-상대방의-행동을-이해하는-방법.md':                  '2021-10-15-How-to-understand-others-actions.md',
    '2021-10-17-아이의-미성숙과-현대-부모의-책임.md':               '2021-10-17-Child-immaturity-and-parents-responsibility.md',
    '2021-10-17-이별의-앞에서.md':                                  '2021-10-17-Before-the-farewell.md',
    '2022-01-11-소크라테스-공부법.md':                              '2022-01-11-Socrates-study-method.md',
    '2022-01-11-회피본능.md':                                       '2022-01-11-Avoidance-instinct.md',
    '2022-02-06-솔직함에-대하여.md':                                '2022-02-06-About-honesty.md',
    '2022-02-11-직장에서-대화에-실패하는-이유.md':                  '2022-02-11-Why-conversations-fail-at-work.md',
    '2022-02-18-근로장학생을-마무리하며.md':                        '2022-02-18-Wrapping-up-work-study.md',
    '2022-03-19-현대-대학-교육과정에-대한-생각과-제안.md':         '2022-03-19-Thoughts-on-modern-university-curriculum.md',
    '2022-03-29-하루-한끼.md':                                      '2022-03-29-One-meal-a-day.md',
    '2022-04-09-(석사-생활)-1.-후회.md':                            '2022-04-09-(Masters-Life)-1-Regret.md',
}


def slug_of(basename: str) -> str:
    """Strip YYYY-MM-DD- prefix and .md suffix to get the URL slug."""
    name = basename[:-3] if basename.endswith('.md') else basename
    m = re.match(r'^\d{4}-\d{2}-\d{2}-(.*)$', name)
    return m.group(1) if m else name


def build_slug_map() -> dict[str, str]:
    return {slug_of(old): slug_of(new) for old, new in RENAMES.items()}


def update_links_in_file(path: str, slug_map: dict[str, str], dry_run: bool) -> bool:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = content
    for old_slug, new_slug in slug_map.items():
        if old_slug == new_slug:
            continue
        # Match the slug only when followed by .html to avoid touching unrelated text.
        # Slugs may contain regex metacharacters, so escape.
        pat = re.escape(old_slug) + r'\.html'
        new_content = re.sub(pat, new_slug + '.html', new_content)

    if new_content == content:
        return False
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    return True


def find_actual_path(posts_root: str, basename: str) -> str | None:
    """Locate a file by basename inside posts_root (depth-1 by default)."""
    for entry in os.listdir(posts_root):
        d = os.path.join(posts_root, entry)
        if not os.path.isdir(d):
            continue
        candidate = os.path.join(d, basename)
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> int:
    dry = '--dry-run' in sys.argv
    posts_root = '_posts'
    slug_map = build_slug_map()

    # 1) Update links in EVERY post (old slugs may be referenced from non-Korean files too)
    print('=== Updating internal links ===')
    link_changed = 0
    for root, dirs, fs in os.walk(posts_root):
        for f in fs:
            if not f.endswith('.md'):
                continue
            path = os.path.join(root, f)
            if update_links_in_file(path, slug_map, dry):
                link_changed += 1
                print(f'  links: {path}')
    print(f'  Total link-updated files: {link_changed}')

    # 2) Rename files
    print('\n=== Renaming files ===')
    rename_count = 0
    missing = []
    for old_name, new_name in RENAMES.items():
        path = find_actual_path(posts_root, old_name)
        if not path:
            missing.append(old_name)
            continue
        new_path = os.path.join(os.path.dirname(path), new_name)
        if os.path.exists(new_path):
            print(f'  SKIP (target exists): {new_path}')
            continue
        if dry:
            print(f'  {old_name}  ->  {new_name}')
        else:
            os.rename(path, new_path)
            print(f'  renamed: {old_name}  ->  {new_name}')
        rename_count += 1

    print(f'\n  Total {"would rename" if dry else "renamed"}: {rename_count} / {len(RENAMES)} files')
    if missing:
        print(f'\n  MISSING ({len(missing)}):')
        for m in missing:
            print(f'    {m}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
