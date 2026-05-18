---
---
(function () {
  function parseEntry(entry) {
    var m = entry.match(/^\s*([^\s]+?)\.\s+(.+)\s*$/);
    if (!m) return { prefix: [entry], display: entry, localDisplay: entry, name: entry, prefixStr: entry };
    var prefixStr = m[1];
    var name = m[2];
    var prefix = prefixStr.split('.').filter(function (s) { return s.length > 0; });
    // Local display = last prefix segment + name (e.g. "1.1.1. Machine Learning" -> "1. Machine Learning")
    var lastSeg = prefix[prefix.length - 1];
    var localDisplay = lastSeg + '. ' + name;
    return { prefix: prefix, display: entry, localDisplay: localDisplay, name: name, prefixStr: prefixStr };
  }

  function keyFromPrefixSlice(arr, end) {
    return arr.slice(0, end).join('.');
  }

  function createNode(key, display) {
    return {
      key: key,
      display: display,
      children: {},
      childOrder: [],
      posts: []
    };
  }

  function naturalCompare(a, b) {
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
  }

  window.CategoryTree = {
    build: function (opts) {
      var container = document.getElementById(opts.containerId);
      if (!container) return;

      var data;
      try {
        var dataEl = document.getElementById(opts.dataId || 'tree-data');
        data = JSON.parse(dataEl.textContent);
      } catch (e) {
        console.error('CategoryTree: failed to parse tree data', e);
        return;
      }

      var compact = !!opts.compact;
      var highlightUrl = opts.highlightUrl || null;
      var autoExpand = !!opts.autoExpandToHighlight;
      var hideDates = !!opts.hideDates;

      var root = createNode('', 'root');

      function ensurePath(prefixArr, finalDisplay) {
        var cur = root;
        for (var i = 0; i < prefixArr.length; i++) {
          var k = keyFromPrefixSlice(prefixArr, i + 1);
          if (!cur.children[k]) {
            var isLeaf = (i === prefixArr.length - 1);
            // Fallback display for unnamed intermediates: just the last numeric segment + dot
            var fallback = prefixArr[i] + '.';
            cur.children[k] = createNode(k, isLeaf ? finalDisplay : fallback);
            cur.childOrder.push(k);
          } else if (i === prefixArr.length - 1 && finalDisplay) {
            cur.children[k].display = finalDisplay;
          }
          cur = cur.children[k];
        }
        return cur;
      }

      // Each post's `categories` array now comes from its directory hierarchy
      // (Jekyll auto-derives one category per directory segment under _posts/).
      // Sort by prefix length to ensure outermost-first walk regardless of
      // how Jekyll orders them, then build the path and place the post at the leaf.
      function categoryDepth(cat) {
        var p = parseEntry(cat);
        return p.prefix.length;
      }

      data.posts.forEach(function (post) {
        var cats = (post.categories || []).slice();
        if (cats.length === 0) return;

        cats.sort(function (a, b) { return categoryDepth(a) - categoryDepth(b); });

        var cur = root;
        cats.forEach(function (catName) {
          var p = parseEntry(catName);
          var key = p.prefix.join('.');
          if (!cur.children[key]) {
            cur.children[key] = createNode(key, p.localDisplay);
            cur.childOrder.push(key);
          }
          cur = cur.children[key];
        });

        cur.posts.push(post);
      });

      function sortTree(node) {
        node.childOrder.sort(naturalCompare);
        node.posts.sort(function (a, b) { return b.date.localeCompare(a.date); });
        node.childOrder.forEach(function (k) { sortTree(node.children[k]); });
      }
      sortTree(root);

      function countSubtree(node, seen) {
        seen = seen || new Set();
        node.posts.forEach(function (p) { seen.add(p.url); });
        node.childOrder.forEach(function (k) { countSubtree(node.children[k], seen); });
        return seen.size;
      }

      // Mark ancestor node keys for highlighted post (may exist in multiple branches).
      var highlightPath = new Set();
      if (highlightUrl) {
        (function markPaths(node, ancestors) {
          if (node.posts.some(function (p) { return p.url === highlightUrl; })) {
            ancestors.forEach(function (a) { highlightPath.add(a); });
          }
          node.childOrder.forEach(function (k) {
            ancestors.push(node.children[k].key);
            markPaths(node.children[k], ancestors);
            ancestors.pop();
          });
        })(root, []);
      }

      function renderPost(post) {
        var li = document.createElement('li');
        li.className = 'tree-post';
        li.dataset.search = (post.title + ' ' + post.date).toLowerCase();
        if (highlightUrl && post.url === highlightUrl) {
          li.classList.add('current');
        }

        if (!hideDates) {
          var date = document.createElement('span');
          date.className = 'tree-post-date';
          date.textContent = post.date;
          li.appendChild(date);
        }

        var link = document.createElement('a');
        link.className = 'tree-post-link';
        link.href = post.url;
        link.textContent = post.title;
        li.appendChild(link);

        return li;
      }

      function renderNode(node, depth) {
        var li = document.createElement('li');
        li.className = 'tree-node';
        li.dataset.depth = depth;
        li.dataset.name = node.display.toLowerCase();

        var hasChildren = node.childOrder.length > 0;
        var hasPosts = node.posts.length > 0;
        var isExpandable = hasChildren || hasPosts;

        var header = document.createElement('div');
        header.className = 'tree-node-header';
        if (!isExpandable) header.classList.add('tree-node-leaf');

        var toggle = document.createElement('span');
        toggle.className = 'tree-toggle';
        toggle.textContent = isExpandable ? '▶' : '·';
        header.appendChild(toggle);

        var name = document.createElement('span');
        name.className = 'tree-name';
        name.textContent = node.display;
        header.appendChild(name);

        var count = document.createElement('span');
        count.className = 'tree-count';
        count.textContent = countSubtree(node);
        header.appendChild(count);

        if (isExpandable) {
          header.addEventListener('click', function () {
            li.classList.toggle('expanded');
          });
        }
        li.appendChild(header);

        if (isExpandable) {
          var content = document.createElement('div');
          content.className = 'tree-node-content';

          if (hasChildren) {
            var childUl = document.createElement('ul');
            childUl.className = 'tree-children';
            node.childOrder.forEach(function (k) {
              childUl.appendChild(renderNode(node.children[k], depth + 1));
            });
            content.appendChild(childUl);
          }

          if (hasPosts) {
            var postsUl = document.createElement('ul');
            postsUl.className = 'tree-posts';
            node.posts.forEach(function (p) {
              postsUl.appendChild(renderPost(p));
            });
            content.appendChild(postsUl);
          }

          li.appendChild(content);
        }

        if (autoExpand && highlightPath.has(node.key)) {
          li.classList.add('expanded');
        }

        return li;
      }

      // Clear container before rendering (idempotent)
      container.innerHTML = '';
      if (compact) container.classList.add('tree-compact');

      var rootUl = document.createElement('ul');
      rootUl.className = 'tree-list tree-root-list';
      root.childOrder.forEach(function (k) {
        rootUl.appendChild(renderNode(root.children[k], 1));
      });
      container.appendChild(rootUl);

      // Scroll current post into view (sidebar use case)
      if (autoExpand && highlightUrl) {
        setTimeout(function () {
          var cur = container.querySelector('.tree-post.current');
          if (cur && cur.scrollIntoView) {
            cur.scrollIntoView({ block: 'center', behavior: 'auto' });
          }
        }, 0);
      }

      // Optional search input
      if (opts.searchInputId) {
        var searchInput = document.getElementById(opts.searchInputId);
        if (searchInput) {
          // Filter logic, factored out so URL ?q= can trigger it on load.
          function runFilter() {
            var q = searchInput.value.trim().toLowerCase();
            var allNodes = container.querySelectorAll('.tree-node');
            var allPosts = container.querySelectorAll('.tree-post');

            if (q === '') {
              allNodes.forEach(function (n) {
                n.style.display = '';
                n.classList.remove('expanded');
              });
              allPosts.forEach(function (p) { p.style.display = ''; });
              return;
            }

            allPosts.forEach(function (p) {
              var hit = p.dataset.search.indexOf(q) !== -1;
              p.style.display = hit ? '' : 'none';
            });

            function nodeMatches(node) {
              var nameHit = node.dataset.name.indexOf(q) !== -1;
              var anyChildShown = false;

              node.querySelectorAll(':scope > .tree-node-content > .tree-children > .tree-node').forEach(function (child) {
                if (nodeMatches(child)) anyChildShown = true;
              });

              var anyPostShown = Array.prototype.some.call(
                node.querySelectorAll(':scope > .tree-node-content > .tree-posts > .tree-post'),
                function (p) { return p.style.display !== 'none'; }
              );

              var visible = nameHit || anyChildShown || anyPostShown;
              node.style.display = visible ? '' : 'none';
              if (visible && (anyChildShown || anyPostShown || nameHit)) {
                node.classList.add('expanded');
              }
              return visible;
            }

            container.querySelectorAll('.tree-root-list > .tree-node').forEach(nodeMatches);
          }

          searchInput.addEventListener('input', runFilter);

          // Pre-fill from URL ?q= so chip links land users with the right filter.
          try {
            var params = new URLSearchParams(window.location.search);
            var initial = params.get('q');
            if (initial) {
              searchInput.value = initial;
              runFilter();
            }
          } catch (e) { /* older browser: skip */ }
        }
      }

      // Optional "collapse all" button
      if (opts.collapseAllId) {
        var collapseBtn = document.getElementById(opts.collapseAllId);
        if (collapseBtn) {
          collapseBtn.addEventListener('click', function () {
            container.querySelectorAll('.tree-node.expanded').forEach(function (n) {
              n.classList.remove('expanded');
            });
          });
        }
      }
    }
  };
})();
