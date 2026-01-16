---
layout: single
title: "도커파일 빌드"
categories:
    - Docker
tag:
    - [docker, container, dockerfile]
author_profile: false
sidebar:
    nav: "docs"
---

```bash
docker build -f Dockerfile -t ros2_humble .
```
`-f` : 파일 지정
`-t` : 태그 지정
`.` : 현재 경로에서 빌드
