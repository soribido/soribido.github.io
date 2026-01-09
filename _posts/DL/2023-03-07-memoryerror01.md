---
layout: single
title:  "Docker memory error"
categories: 
    - DL
tag:
    - [error, memory, CUDA, GPU]    
author_profile: false
sidebar:
    nav: "docs"
---

* 상황 : torch dataloader를 사용했으나 메모리 오류 발생  
* 원인 : 도커 컨테이너 생성 시 shared memory(--shm-size)가 기본적으로 64MB(or256MB)로 설정되어 있기 때문에 발생  
* 해결 방법(1)  
1. 도커 컨테이너 생성시 옵션으로 --shm-size=2G 와 같이 설정해주기  

* 해결 방법(2)
1. service docker stop 으로 도커 서비스 종료   
2. /var/lib/docker/containers/<container-id >/hostconfig.json 에 "ShmSize" 수정  
기본적으로 B단위로 설정된것으로 보이며, 필자의 경우 8GB로 설정하기 위해 8589934592를 입력하였다. (8x1024x1024x1024)  
3. service docker start 로 도커 서비스 시작

+) 수정은 .json 파일경로까지 접근한 다음 vim hostconfig.json 을 통해 가능하다.  
+) 도커 컨테이너 내부 접속 후 df -h 명령어를 통해 shm-size 확인이 가능하다  


Ref)  
https://github.com/docker/cli/issues/1278  