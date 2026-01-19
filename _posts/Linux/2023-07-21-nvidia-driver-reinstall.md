---
layout: single
title: "nvidia-driver 재설치"
categories:
    - Linux
tag:
    - nvidia-driver
author_profile: false
sidebar:
    nav: "docs"
---

## 1. 기존 nvidia driver 삭제

재설치 시 현재 설치된 nvidia driver를 먼저 삭제한다. (처음 설치하는 경우 2번으로 이동)

```bash
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
```

보통 위 명령어를 실행해도 남아있는 경우가 있다. 이름이 아닌 설명에 nvidia가 있는 것은 지울 필요없다. (ex: extra....)

```bash
sudo dpkg -l | grep nvidia
```

위 명령어 실행 후 남아있는 것을 수동으로 삭제한다.

```bash
sudo apt-get remove --purge xxx
```

## 2. 드라이버 설치

처음 설치 혹은 재설치 시 모두 삭제한 이후 아래 명령어를 실행한다.

```bash
ubuntu-drivers devices
sudo apt-get install nvidia-driver-535
sudo apt-get install dkms nvidia-modprobe
sudo apt-get update
sudo apt-get upgrade
sudo reboot
```

첫 번째 명령어는 권장 드라이버 목록을 보여준다. 컴퓨터의 사양을 알고 있다면 nvidia 홈페이지에서 직접 검색하는 것도 좋다.

드라이버를 설치한 후에는 재부팅을 권장하며, 재부팅 이후에 `nvidia-smi`를 입력했을 때 정상적으로 출력된다면 성공이다.
