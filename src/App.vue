<template>
  <div class="page-wrapper">
    <main class="container">
      <ProfileHeader />
      <NewsSection />
      <PublicationsSection />
      <HonorsSection />
      <ServicesSection />
    </main>

    <footer class="footer">
      <div id="globe-container" style="width: 100px; height: 100px; margin: 1rem auto;">
        <!-- The globe script is in index.html, we just need the container -->
      </div>
      <p>&copy; {{ new Date().getFullYear() }} Duo Zhou. Built with Vue 3.</p>
    </footer>
  </div>
</template>

<script setup>
import { onMounted } from 'vue';
import ProfileHeader from './components/ProfileHeader.vue';
import NewsSection from './components/NewsSection.vue';
import PublicationsSection from './components/PublicationsSection.vue';
import HonorsSection from './components/HonorsSection.vue';
import ServicesSection from './components/ServicesSection.vue';

onMounted(() => {
  // Load the ClustrMaps script dynamically to avoid Vue hydration issues
  const script = document.createElement('script');
  script.type = 'text/javascript';
  script.id = 'clstr_globe';
  script.src = '//clustrmaps.com/globe.js?d=OyHOj7tI7JgQh3wi5Y7ceOXED8lvucb0mvSDnMOej0o';
  document.getElementById('globe-container')?.appendChild(script);
});
</script>

<style scoped>
.page-wrapper {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.container {
  flex: 1;
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
}
.footer {
  text-align: center;
  padding: 2rem;
  background-color: var(--card-bg);
  color: var(--text-muted);
  font-size: 0.9rem;
  margin-top: 4rem;
  border-top: 1px solid var(--border-color);
}
</style>
