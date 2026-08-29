<template>
  <div class="site-shell">
    <a class="skip-link" href="#main">Skip to content</a>

    <header class="topbar" aria-label="Primary navigation">
      <a class="brand" href="#top" aria-label="Duo Zhou, home">
        <span class="brand-mark">DZ</span>
        <span class="brand-path">~/duo-zhou</span>
      </a>
      <nav class="nav-links" aria-label="Page sections">
        <a href="#research">01 / research</a>
        <a href="#news">02 / news</a>
        <a href="#papers">03 / papers</a>
        <a href="#service">04 / service</a>
      </nav>
    </header>

    <main id="main">
      <section id="top" class="hero section-grid" aria-labelledby="hero-title">
        <div class="hero-copy pixel-window">
          <div class="window-bar" aria-hidden="true">
            <span></span><span></span><span></span>
            <p>profile.md</p>
          </div>
          <div class="hero-content">
            <p class="eyebrow">TRUSTWORTHY AI / FROM OPTIMIZATION TO SELF-IMPROVING AGENTS</p>
            <h1 id="hero-title">Duo Zhou<span class="cursor" aria-hidden="true">_</span></h1>
            <p class="role">Ph.D. Student · Computer Science · UIUC</p>
            <p class="intro" v-html="profile.intro"></p>
            <div class="profile-links" aria-label="Profile links">
              <a
                v-for="link in profile.links"
                :key="link.label"
                :href="link.url"
                :target="link.url.startsWith('mailto:') ? undefined : '_blank'"
                rel="noopener"
                class="pixel-button"
              >
                <span aria-hidden="true">↗</span> {{ link.label }}
              </a>
            </div>
          </div>
        </div>

        <aside id="research" class="research-panel" aria-labelledby="research-title">
          <div class="panel-label">
            <span class="status-dot" aria-hidden="true"></span>
            <span>research.log</span>
          </div>
          <h2 id="research-title">What I work on</h2>
          <ol class="research-list">
            <li v-for="interest in profile.researchInterests" :key="interest.code">
              <span class="research-code">{{ interest.code }}</span>
              <div>
                <h3>{{ interest.title }}</h3>
                <p v-html="interest.description"></p>
              </div>
            </li>
          </ol>
        </aside>
      </section>

      <section id="news" class="content-section" aria-labelledby="news-title">
        <div class="section-heading">
          <div>
            <p class="section-index">02 / UPDATE STREAM</p>
            <h2 id="news-title">News</h2>
          </div>
          <p class="heading-note">recent.log</p>
        </div>
        <div class="news-list">
          <article v-if="news[0]" class="news-row news-row-featured">
            <span class="news-number">01</span>
            <time>{{ news[0].date }}</time>
            <p v-html="news[0].body"></p>
          </article>
          <details v-if="news.length > 1" class="news-archive">
            <summary>
              <span>Show {{ news.length - 1 }} earlier updates</span>
              <span class="summary-icon" aria-hidden="true">+</span>
            </summary>
            <div class="news-archive-list">
              <article
                v-for="(item, index) in news.slice(1)"
                :key="`${item.date}-${item.body}`"
                class="news-row"
              >
                <span class="news-number">{{ String(index + 2).padStart(2, '0') }}</span>
                <time>{{ item.date }}</time>
                <p v-html="item.body"></p>
              </article>
            </div>
          </details>
        </div>
      </section>

      <section id="papers" class="content-section" aria-labelledby="papers-title">
        <div class="section-heading papers-heading">
          <div>
            <p class="section-index">03 / SELECTED OUTPUT</p>
            <h2 id="papers-title">Publications</h2>
          </div>
          <p class="heading-note">
            * equal contribution · † corresponding ·
            <a
              href="https://scholar.google.com/citations?user=QnBzRsIAAAAJ&hl=en"
              target="_blank"
              rel="noopener"
            >Complete publication list on Google Scholar ↗</a>
          </p>
        </div>

        <div class="publication-list">
          <article
            v-for="(pub, index) in publications"
            :key="pub.title"
            class="publication-card"
            :class="{ featured: pub.featured }"
          >
            <div class="pub-index" aria-hidden="true">
              <span>{{ pub.year }}</span>
              <strong>{{ String(index + 1).padStart(2, '0') }}</strong>
            </div>
            <div class="pub-body">
              <div class="pub-meta">
                <span class="venue-chip">{{ pub.venueShort }}</span>
                <span v-if="pub.featured" class="new-chip">NEW</span>
              </div>
              <h3 v-html="pub.title"></h3>
              <p class="authors" v-html="pub.authors"></p>
              <p class="venue" v-html="pub.venue"></p>
            </div>
            <div v-if="pub.links?.length" class="pub-links" aria-label="Publication resources">
              <a
                v-for="link in pub.links"
                :key="link.url"
                :href="link.url"
                target="_blank"
                rel="noopener"
              >{{ link.label }} ↗</a>
            </div>
          </article>
        </div>
      </section>

      <section id="service" class="content-section bottom-grid" aria-label="Honors and academic service">
        <div class="ledger-panel">
          <div class="section-heading compact">
            <div>
              <p class="section-index">04A / MILESTONES</p>
              <h2>Honors</h2>
            </div>
          </div>
          <ul class="ledger-list">
            <li v-for="(honor, index) in honors" :key="honor">
              <span>{{ String(index + 1).padStart(2, '0') }}</span>
              <p>{{ honor }}</p>
            </li>
          </ul>
        </div>

        <div class="ledger-panel service-panel">
          <div class="section-heading compact">
            <div>
              <p class="section-index">04B / COMMUNITY</p>
              <h2>Service</h2>
            </div>
          </div>
          <ul class="ledger-list">
            <li v-for="(service, index) in services" :key="service">
              <span>{{ String(index + 1).padStart(2, '0') }}</span>
              <p>{{ service }}</p>
            </li>
          </ul>
        </div>
      </section>
    </main>

    <footer class="footer">
      <p><span class="footer-mark">DZ</span> Built for the open web · {{ new Date().getFullYear() }}</p>
      <a href="#top">back_to_top ↑</a>
    </footer>
  </div>
</template>

<script setup>
import {
  honorsData,
  newsData,
  profileData,
  publicationsData,
  servicesData,
} from './data';

const profile = profileData;
const news = newsData;
const publications = publicationsData;
const honors = honorsData;
const services = servicesData;
</script>
