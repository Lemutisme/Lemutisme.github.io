<template>
  <section class="section">
    <h2 class="section-title">News</h2>
    <div class="news-list">
      <div v-for="(item, index) in news" :key="index" class="news-item">
        <span class="news-date">{{ item.date }}</span>
        <span class="news-body" v-html="item.body"></span>
      </div>
    </div>
  </section>
</template>

<script setup>
import { newsData } from '../data';
const news = newsData;
</script>

<style scoped>
.news-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.news-item {
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background: var(--card-bg);
  border-radius: var(--border-radius);
  border-left: 4px solid var(--primary-color);
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s ease;
}
.news-item:hover {
  transform: translateX(5px);
}
@media (min-width: 600px) {
  .news-item {
    flex-direction: row;
    align-items: baseline;
    gap: 1rem;
  }
}
.news-date {
  font-weight: 600;
  color: var(--primary-color);
  min-width: 100px;
  font-size: 0.95rem;
}
.news-body {
  color: var(--text-color);
  line-height: 1.5;
}
</style>
