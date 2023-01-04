---
title: "etc"
layout: archive
permalink: /etc/
sidebar:
    nav: "docs"
sidebar_main: true
---


{% assign posts = site.categories.etc %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}