// 简单交互：移动端菜单、主题切换、年份与隐私说明显示
(function() {
  const navToggle = document.querySelector('.nav-toggle');
  const navMenu = document.querySelector('#nav-menu');
  const themeToggle = document.querySelector('#theme-toggle');
  const privacy = document.querySelector('#privacy');
  const privacyLink = document.querySelector('.privacy-link');

  // 移动端菜单
  if (navToggle && navMenu) {
    navToggle.addEventListener('click', () => {
      const open = navMenu.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', String(open));
    });
  }

  // 主题切换（localStorage 记忆）
  function setTheme(mode) {
    document.documentElement.dataset.theme = mode;
    localStorage.setItem('theme', mode);
  }
  const storedTheme = localStorage.getItem('theme');
  if (storedTheme) setTheme(storedTheme);
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = document.documentElement.dataset.theme || 'auto';
      const next = current === 'dark' ? 'light' : current === 'light' ? 'auto' : 'dark';
      setTheme(next);
    });
  }

  // 年份
  const year = document.querySelector('#year');
  if (year) year.textContent = new Date().getFullYear();

  // 隐私说明折叠
  if (privacy && privacyLink) {
    privacyLink.addEventListener('click', (e) => {
      e.preventDefault();
      const isHidden = privacy.hasAttribute('hidden');
      if (isHidden) privacy.removeAttribute('hidden'); else privacy.setAttribute('hidden', '');
      privacyLink.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
})();
