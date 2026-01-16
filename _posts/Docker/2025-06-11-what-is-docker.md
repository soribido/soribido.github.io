---
layout: single
title: "도커란 무엇인가"
categories:
    - Docker
tag:
    - [docker, container]
author_profile: false
sidebar:
    nav: "docs"
---

가상머신(VM), 컨테이너, Conda환경과의 차이점을 통해 도커의 개념을 정리하고자 한다.

| 구분    | 가상 머신 (VM)             | 컨테이너 (Docker 등)           | conda 가상환경               |
| ----- | ---------------------- | ------------------------- | ------------------------ |
| 격리 수준 | OS 단위 격리 (Guest OS 포함) | 프로세스 단위 격리 (공유 OS 위에서 실행) | 패키지·라이브러리 수준 격리          |
| 실행 단위 | OS 전체                  | 어플리케이션                    | Python 환경                |
| 목적    | 완전한 OS 환경 제공           | 애플리케이션 배포 및 실행 격리         | Python 패키지 충돌 방지 및 버전 관리 |

즉 가상 머신은 OS하나를 새로 더 만들어서 실행하는 것이고 (예를 들어 윈도우의 Hyper-V같은 경우 OS를 iso파일로 구워서 새로 설치하는 형식이다.)
컨테이너는 Host OS와 커널은 공유하지만 파일시스템, 어플리케이션은 분리한다.
Conda가상환경은 python 패키지 수준에서 격리를 실행한다.

<center><img src='{{"/assets/images/post-docker-intro/vm_container_comparison.jpeg" | relative_url}}' width="80%"></center>
<br>

한 서버에서 사용자가 여러명이라면 파일시스템 I/O, 포트, 환경변수가 모두 분리된 컨테이너가 conda보다 유리하다.
1명이라면 도커는 container runtime을 한번 거치므로 이론상 conda가 더 빠르지만 크게 유의미한 수치는 아니다.
