---
layout: home
title: "Learning curve"
---

{% for post in site.posts %}
  <article class="post-preview">
    <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
    <p>{{ post.excerpt }}</p>
  </article>
{% endfor %}