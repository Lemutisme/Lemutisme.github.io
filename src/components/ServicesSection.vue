<template>
  <section class="section">
    <h2 class="section-title">Selected Services</h2>
    <ul class="list">
      <li v-for="(service, index) in services" :key="index" class="list-item">
        <span class="bullet"></span>
        <span v-html="service"></span>
      </li>
    </ul>
  </section>
</template>

<script setup>
import { servicesData } from '../data';
const services = servicesData;
</script>

<style scoped>
.list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  list-style: none;
  padding: 0;
}
.list-item {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  color: var(--text-color);
  line-height: 1.5;
}
.bullet {
  margin-top: 0.5rem;
  min-width: 6px;
  height: 6px;
  background-color: var(--primary-color);
  border-radius: 50%;
  display: inline-block;
}
</style>
