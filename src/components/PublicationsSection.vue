<template>
  <section class="section">
    <h2 class="section-title">Selected Papers</h2>
    <p class="note">
      * indicates contributed equally &nbsp;&nbsp; <sup>†</sup> indicates corresponding author
    </p>
    <div class="publications">
      <div v-for="(pub, index) in publications" :key="index" class="pub-item">
        <h3 class="papertitle" v-html="pub.title"></h3>
        <p class="authors" v-html="pub.authors"></p>
        <p class="venue" v-html="pub.venue"></p>
        <div class="pub-links" v-if="pub.links && pub.links.length > 0">
          <a v-for="(link, lIndex) in pub.links" :key="lIndex" :href="link.url" target="_blank" rel="noopener" class="pub-btn">
            {{ link.label }}
          </a>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { publicationsData } from '../data';
const publications = publicationsData;
</script>

<style scoped>
.note {
  color: var(--text-muted);
  font-size: 0.9rem;
  margin-bottom: 1.5rem;
}
.publications {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}
.pub-item {
  background: var(--card-bg);
  padding: 1.5rem;
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  border: 1px solid var(--border-color);
}
.pub-item:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-md);
}
.papertitle {
  font-size: 1.25rem;
  color: var(--text-color);
  margin-bottom: 0.5rem;
  font-weight: 700;
  line-height: 1.4;
}
.authors {
  color: var(--text-light);
  margin-bottom: 0.5rem;
  font-size: 1rem;
  line-height: 1.5;
}
.venue {
  font-style: italic;
  color: var(--primary-color);
  margin-bottom: 1rem;
  font-size: 0.95rem;
}
.pub-links {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.pub-btn {
  display: inline-block;
  padding: 0.35rem 0.75rem;
  font-size: 0.85rem;
  font-weight: 600;
  text-decoration: none;
  background-color: var(--bg-color);
  color: var(--primary-color);
  border: 1px solid var(--primary-color);
  border-radius: 4px;
  transition: all 0.2s ease;
}
.pub-btn:hover {
  background-color: var(--primary-color);
  color: white;
}
</style>
