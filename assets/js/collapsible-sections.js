// 포스트 본문의 h1~h6 섹션을 접었다 펼칠 수 있게 만든다.
// - 처음에는 h1, h2 섹션만 접힌 상태(COLLAPSED_MAX_LEVEL), h3~h6은 펼쳐진 상태
// - 제목을 클릭(또는 Enter/Space)하면 해당 섹션이 펼쳐짐
// - URL 해시나 목차 링크로 제목에 이동하면 상위 섹션까지 자동으로 펼쳐짐
(function () {
  'use strict';

  // 이 레벨 이하(h1~h2)의 제목은 처음에 접어 둔다. 그보다 깊은 제목은 펼친 채로 시작한다.
  var COLLAPSED_MAX_LEVEL = 2;

  function level(el) {
    return el && /^H[1-6]$/.test(el.tagName) ? +el.tagName[1] : 0;
  }

  // 같은 부모 안에서, 제목 뒤에 오는 형제들을 다음 같은/상위 레벨 제목 전까지 묶는다.
  function wrapSections(container, lv) {
    var headings = Array.prototype.filter.call(container.children, function (el) {
      return level(el) === lv;
    });

    headings.forEach(function (h) {
      var body = document.createElement('div');
      body.className = 'section-body';
      var next = h.nextSibling;
      while (next) {
        var l = next.nodeType === 1 ? level(next) : 0;
        if (l && l <= lv) break;
        var cur = next;
        next = next.nextSibling;
        body.appendChild(cur);
      }
      h.parentNode.insertBefore(body, h.nextSibling);

      h.classList.add('section-toggle');
      h.setAttribute('role', 'button');
      h.setAttribute('tabindex', '0');
      setExpanded(h, lv > COLLAPSED_MAX_LEVEL);

      h.addEventListener('click', function (e) {
        if (e.target.closest && e.target.closest('a')) return; // 제목 안의 링크는 그대로 동작
        setExpanded(h, !isExpanded(h));
      });
      h.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          setExpanded(h, !isExpanded(h));
        }
      });

      // 브라우저 페이지 내 검색(Ctrl+F)으로 숨긴 내용이 찾아지면 자동으로 펼친다.
      body.addEventListener('beforematch', function () { setExpanded(h, true); });

      // 하위 레벨 제목도 재귀적으로 처리
      for (var sub = lv + 1; sub <= 6; sub++) wrapSections(body, sub);
    });
  }

  function isExpanded(h) {
    return h.getAttribute('aria-expanded') === 'true';
  }

  function setExpanded(h, open) {
    var body = h.nextElementSibling;
    if (!body || !body.classList.contains('section-body')) return;
    h.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
      body.removeAttribute('hidden');
    } else {
      body.setAttribute('hidden', 'until-found');
    }
  }

  // 특정 요소를 감싸고 있는 모든 상위 섹션을 펼친다.
  function revealElement(el) {
    var node = el;
    while (node && node !== document.body) {
      if (node.classList && node.classList.contains('section-body')) {
        var h = node.previousElementSibling;
        if (h && h.classList.contains('section-toggle')) setExpanded(h, true);
      }
      node = node.parentNode;
    }
    if (el.classList && el.classList.contains('section-toggle')) setExpanded(el, true);
  }

  function revealHash(hash) {
    if (!hash) return;
    var id = decodeURIComponent(hash.replace(/^.*#(h-)?/, ''));
    var target = id && document.getElementById(id);
    if (target) revealElement(target);
  }

  document.addEventListener('DOMContentLoaded', function () {
    var content = document.querySelector('.post-content');
    if (!content) return;

    for (var lv = 1; lv <= 6; lv++) wrapSections(content, lv);

    // 처음 열 때 해시가 있으면 해당 섹션을 먼저 펼친다(hashlocate가 load 시점에 스크롤함).
    revealHash(window.location.hash);
    window.addEventListener('hashchange', function () { revealHash(window.location.hash); });

    // 목차/본문 안의 #링크 클릭 시, 스크롤 계산(hashlocate)보다 먼저 섹션을 펼친다.
    window.addEventListener('click', function (e) {
      var a = e.target.closest && e.target.closest('a[href*="#"]');
      if (a) revealHash(a.getAttribute('href'));
    }, true);
  });
})();
