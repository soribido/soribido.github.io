---
title: "blog"
layout: archive
permalink: /blog/
sidebar_main: true    
---


{% assign posts = site.categories.blog %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}