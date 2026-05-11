(function () {
  'use strict';

  /* ============================================
     Utility
  ============================================ */
  const on = (el, evt, fn) => el.addEventListener(evt, fn);
  const qs = (s, ctx) => (ctx || document).querySelector(s);
  const qsa = (s, ctx) => Array.from((ctx || document).querySelectorAll(s));

  /* ============================================
     Navigation — scroll shadow
  ============================================ */
  const nav = qs('.nav');
  on(window, 'scroll', () => {
    nav.classList.toggle('nav--scrolled', window.scrollY > 40);
  });

  /* ============================================
     Stats Counter — animate on scroll
  ============================================ */
  function animateCounters() {
    const nums = qsa('.stat__num[data-count]');
    if (!nums.length) return;

    nums.forEach(el => {
      const target = parseInt(el.getAttribute('data-count'), 10);
      const duration = 2000;
      const start = performance.now();

      function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(eased * target);

        if (target >= 1000) {
          el.textContent = current >= 1000
            ? Math.round(current / 1000) + 'K'
            : String(current);
        } else {
          el.textContent = String(current);
        }

        if (progress < 1) requestAnimationFrame(update);
        else el.textContent = target >= 1000 ? Math.round(target / 1000) + 'K' : String(target);
      }

      requestAnimationFrame(update);
    });
  }

  /* ============================================
     Scroll Reveal — IntersectionObserver
  ============================================ */
  function initReveal() {
    const elMap = new Map();

    const obs = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el = entry.target;

          if (el.classList.contains('reveal')) {
            el.classList.add('reveal--visible');
          }

          if (el.closest('.reveal-children')) {
            const children = qsa('> *', el.closest('.reveal-children'));
            // Stagger children with a small delay
            children.forEach((child, i) => {
              setTimeout(() => child.classList.add('revealed'), i * 100);
            });
          }

          obs.unobserve(el);
        }
      });
    }, { threshold: 0.15 });

    qsa('.reveal').forEach(el => obs.observe(el));

    // Observe parent of reveal-children to stagger children
    qsa('.reveal-children').forEach(parent => {
      obs.observe(parent);
    });
  }

  /* ============================================
     Tabs — ARIA tab pattern
  ============================================ */
  function initTabs() {
    const tabs = qsa('.tab');
    const panels = qsa('.panel');
    const indicator = qs('.tabs__indicator');
    const tablist = qs('[role="tablist"]');
    if (!tabs.length || !tablist) return;

    function moveIndicator(tab) {
      if (!indicator) return;
      const rect = tab.getBoundingClientRect();
      const parentRect = tablist.getBoundingClientRect();
      indicator.style.width = rect.width + 'px';
      indicator.style.transform = `translateX(${rect.left - parentRect.left}px)`;
    }

    function activateTab(tab) {
      const targetId = tab.getAttribute('aria-controls');

      tabs.forEach(t => {
        t.setAttribute('aria-selected', 'false');
        t.setAttribute('tabindex', '-1');
      });

      tab.setAttribute('aria-selected', 'true');
      tab.setAttribute('tabindex', '0');
      tab.focus();

      panels.forEach(p => {
        p.classList.remove('panel--active');
        p.hidden = true;
      });

      const panel = document.getElementById(targetId);
      if (panel) {
        panel.classList.add('panel--active');
        panel.hidden = false;
      }

      moveIndicator(tab);
    }

    // Click handlers
    tabs.forEach(tab => {
      on(tab, 'click', () => activateTab(tab));
    });

    // Keyboard navigation
    on(tablist, 'keydown', (e) => {
      const current = qs('.tab[aria-selected="true"]');
      if (!current) return;
      let idx = tabs.indexOf(current);
      let newIdx = -1;

      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        newIdx = (idx + 1) % tabs.length;
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        newIdx = (idx - 1 + tabs.length) % tabs.length;
      } else if (e.key === 'Home') {
        e.preventDefault();
        newIdx = 0;
      } else if (e.key === 'End') {
        e.preventDefault();
        newIdx = tabs.length - 1;
      }

      if (newIdx >= 0) activateTab(tabs[newIdx]);
    });

    // Set initial indicator
    const active = qs('.tab[aria-selected="true"]');
    if (active) moveIndicator(active);
  }

  /* ============================================
     Video Sync — Side-by-side comparison
  ============================================ */
  function initVideoSync() {
    const early = document.getElementById('vid-early');
    const late = document.getElementById('vid-late');
    const btnPlay = document.getElementById('btn-sync-play');
    const btnPause = document.getElementById('btn-sync-pause');
    const btnReset = document.getElementById('btn-sync-reset');

    if (!early || !late || !btnPlay) return;

    let syncing = false;

    function syncPlay() {
      if (syncing) return;
      syncing = true;
      early.play();
      late.play();
      syncing = false;
    }

    function syncPause() {
      if (syncing) return;
      syncing = true;
      early.pause();
      late.pause();
      syncing = false;
    }

    function syncReset() {
      early.pause();
      late.pause();
      early.currentTime = 0;
      late.currentTime = 0;
    }

    // Mirror play/pause between the two
    on(early, 'play', syncPlay);
    on(late, 'play', syncPlay);
    on(early, 'pause', () => { if (!early.ended) syncPause(); });
    on(late, 'pause', () => { if (!late.ended) syncPause(); });

    on(btnPlay, 'click', syncPlay);
    on(btnPause, 'click', syncPause);
    on(btnReset, 'click', syncReset);
  }

  /* ============================================
     Init
  ============================================ */
  document.addEventListener('DOMContentLoaded', () => {
    animateCounters();
    initReveal();
    initTabs();
    initVideoSync();
  });

})();
