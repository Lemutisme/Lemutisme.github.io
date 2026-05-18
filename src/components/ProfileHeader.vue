<template>
  <header class="profile-header">
    <div class="profile-content">
      <div class="profile-text">
        <h1 class="name">{{ profile.name }}</h1>
        <p class="intro" v-html="profile.intro"></p>
        <ul class="research-interests">
          <li v-for="(interest, index) in profile.researchInterests" :key="index">
            <strong>{{ interest.title }}:</strong> <span v-html="interest.description"></span>
          </li>
        </ul>
      </div>
      <div class="profile-sidebar">
        <div class="profile-image">
          <img src="/images/profile.jpg" :alt="`Profile photo of ${profile.name}`" />
        </div>
        <div class="icon-links">
          <a v-for="(link, index) in profile.links" :key="index" :href="link.url" target="_blank" rel="noopener" class="icon-link" :title="link.label">
            <svg v-if="link.label.includes('Email')" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-mail"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
            <svg v-else-if="link.label.includes('Scholar')" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"></path><path d="M6 12v5c3 3 9 3 12 0v-5"></path></svg>
            <svg v-else-if="link.label.includes('GitHub')" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-github"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
          </a>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
import { profileData } from '../data';
const profile = profileData;
</script>

<style scoped>
.profile-header {
  margin-bottom: 3rem;
  padding: 2rem;
  background: var(--card-bg);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.profile-header:hover {
  box-shadow: var(--shadow-md);
}
.profile-content {
  display: flex;
  flex-direction: column-reverse;
  gap: 2rem;
  align-items: center;
}
@media (min-width: 768px) {
  .profile-content {
    flex-direction: row;
    align-items: flex-start;
  }
}
.profile-text {
  flex: 1;
}
.name {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  color: var(--text-color);
  font-family: var(--heading-font);
}
.intro {
  font-size: 1.1rem;
  line-height: 1.6;
  margin-bottom: 1.5rem;
  color: var(--text-light);
}
.research-interests {
  list-style-type: disc;
  padding-left: 1.5rem;
  margin-bottom: 2rem;
  color: var(--text-light);
  line-height: 1.6;
}
.research-interests li {
  margin-bottom: 0.5rem;
}
.research-interests strong {
  color: var(--text-color);
}
.profile-sidebar {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
}
.icon-links {
  display: flex;
  justify-content: center;
  gap: 1rem;
}
.icon-link {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  background: var(--bg-color);
  color: var(--primary-color);
  border: 1px solid var(--border-color);
  border-radius: 50%;
  transition: all 0.2s ease;
}
.icon-link:hover {
  background: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
  transform: translateY(-3px);
  box-shadow: var(--shadow-sm);
}
.profile-image img {
  width: 200px;
  height: 200px;
  object-fit: cover;
  border-radius: 50%;
  border: 4px solid white;
  box-shadow: var(--shadow-md);
  transition: transform 0.3s ease;
}
.profile-image img:hover {
  transform: scale(1.05);
}
</style>
