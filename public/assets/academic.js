// Toggle mobile menu, theme, filters, bibtex panels, and footer info
(function(){
  const navToggle = document.querySelector('.nav-toggle');
  const menu = document.querySelector('.menu');
  if (navToggle && menu) {
    navToggle.addEventListener('click', () => {
      const open = menu.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', String(open));
    });
  }

  // Theme toggle
  const themeToggle = document.getElementById('theme-toggle');
  function setTheme(mode){ document.documentElement.dataset.theme = mode; localStorage.setItem('theme', mode); }
  const saved = localStorage.getItem('theme'); if (saved) setTheme(saved);
  if (themeToggle){
    themeToggle.addEventListener('click', ()=>{
      const cur = document.documentElement.dataset.theme || 'auto';
      const next = cur === 'dark' ? 'light' : cur === 'light' ? 'auto' : 'dark';
      setTheme(next);
    });
  }

  // Year, last modified
  const y = document.getElementById('year'); if (y) y.textContent = new Date().getFullYear();
  const lm = document.getElementById('last-modified'); if (lm) lm.textContent = new Date(document.lastModified).toISOString().slice(0,10);

  // Privacy toggle
  const p = document.getElementById('privacy');
  const pl = document.querySelector('.privacy-link');
  if (p && pl){ pl.addEventListener('click', (e)=>{ e.preventDefault(); const h=p.hasAttribute('hidden'); if(h) p.removeAttribute('hidden'); else p.setAttribute('hidden',''); p.scrollIntoView({behavior:'smooth'}) }); }

  // Publication filters
  const fy = document.getElementById('filter-year');
  const ft = document.getElementById('filter-type');
  const fs = document.getElementById('filter-search');
  const items = Array.from(document.querySelectorAll('.pub-item'));
  function applyFilters(){
    const yv = fy ? fy.value : 'all';
    const tv = ft ? ft.value : 'all';
    const sv = (fs ? fs.value.trim().toLowerCase() : '');
    items.forEach(li=>{
      const matchYear = (yv==='all' || li.dataset.year===yv);
      const matchType = (tv==='all' || li.dataset.type===tv);
      const text = li.textContent.toLowerCase();
      const matchSearch = (!sv || text.includes(sv));
      li.style.display = (matchYear && matchType && matchSearch) ? '' : 'none';
    });
  }
  [fy, ft, fs].forEach(el => el && el.addEventListener('input', applyFilters));
  applyFilters();

  // BibTeX toggle
  document.querySelectorAll('.bibtex-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const pre = btn.closest('.pub-item').querySelector('.bibtex');
      const isHidden = pre.hasAttribute('hidden');
      if (isHidden) pre.removeAttribute('hidden'); else pre.setAttribute('hidden', '');
      btn.setAttribute('aria-expanded', String(isHidden));
    });
  });
})();