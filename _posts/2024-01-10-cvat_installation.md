---
layout: single
title:  "Ubuntu CVAT(Computer Vision Annotation Tool) 설치 가이드"
categories: 
    - etc
tag:
    - [CVAT, linux, ubuntu]    
author_profile: false
sidebar:
    nav: "docs"
---

## CVAT란
CVAT(Computer Vision Annotation Tool)은 오픈소스 이미지 데이터 라벨링 툴의 일종이다.  
기본적으로 detection, segmentation 등에 해당하는 라벨링 기능을 제공하며 기술의 발전으로 인해 AI 모델을 통한 Auto or Semi-auto 라벨링 생성도 가능하다.  
[cvat.ai](cvat.ai)에서 제한적으로 서비스 이용이 가능하며 많은 양의 라벨링 혹은 AI 모델을 사용하고 싶으면 라이센스 비용을 지불해야 하는 구조이다. 하지만 로컬 서버에 직접 설치한다면 CVAT를 무료로 사용할 수 있다.  
설치 및 기능 등 CVAT에 대한 각종 추가적인 정보는 [공식 문서](https://opencv.github.io/cvat/docs/)를 통해 확인하도록 하자.  

## CVAT 설치 방법
설치에 관한 것들은 [공식 문서 설치 가이드](https://opencv.github.io/cvat/docs/administration/basics/installation/)를 참고하여 설치하는 것을 권장한다.  
본 포스팅은 우분투 기준으로 설명하고 있으므로 본인의 서버 환경이 윈도우나 MAC이라면 설치 가이드를 참고하자.
  
CVAT는 도커 컴포즈(docker compose)로 구성된다.  
도커 컴포즈란 특정 서비스나 어플리케이션을 가동시키기 위해 여러 개의 컨테이너를 가동시켜야 하는 경우 이를 묶어서 관리하기 위한 도구 정도로 이해하면 좋을 것 같다.  

설치 가이드의 경우 Ubuntu 18.04로 되어 있는데, 20.04에서도 정상적으로 설치된다. 아래의 명령어를 터미널에 입력하여 우분투 버전을 확인할 수 있다.

```bash
lsb_release -a
```
### 도커 엔진 및 도커 컴포즈 설치
아래의 명령어를 터미널에 입력하여 도커와 도커 컴포즈를 다운로드 받을 수 있다.
```bash
sudo apt-get update
sudo apt-get --no-install-recommends install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg-agent \
  software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
  "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) \
  stable"
sudo apt-get update
sudo apt-get --no-install-recommends install -y \
  docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

* 시스템 부팅시 docker가 시작되도록 설정하고 도커실행하기
```bash
sudo systemctl enable docker && service docker start
```

* 잘 설치되었는지 확인 (active 나오면 됨)
```bash
service docker status
```

### 도커 권한 부여
아래의 명령어를 터미널에 입력하여 도커 권한을 부여한다. 여기에서 $user는 사용자 계정명이다.  
예를 들어, 나의 계정이 kang이라면 sudo usermod -aG docker kang이라고 입력하면 된다.
```
sudo groupadd docker
sudo usermod -aG docker $USER
```

### cvat 소스코드 다운로드
깃을 이용하여 최신 소스코드를 클론한다. git이 없다면 다운로드 해주도록 하자.

```bash
git clone https://github.com/opencv/cvat
cd cvat
```

### CVAT_HOST 환경 변수 설정
만약 네트워크나 다른 기기에서 CVAT 접속을 원한다면 export 명령어를 통해 환경변수를 설정하자.  
여기에서 your-ip-address는 우분투 서버의 IP주소이다(포트 필요없음)
```bash
export CVAT_HOST=your-ip-address
```

IP 주소를 모르겠다면 터미널에 다음 명령어를 입력하자.
```bash
hostname -I
```

예를 들어, 나의 IP 주소가 10.10.10.10이라면 export CVAT_HOST=10.10.10.10을 입력하면 된다.

### 컨테이너 가동
아래의 명령어를 터미널에 입력하자. 해당 과정은 처음에는 시간이 조금 걸린다. 
```bash
docker compose up -d
```
이후 docker ps를 터미널에 입력했을 때 여러개의 컨테이너가 가동되고 있는지를 확인하자.


### 슈퍼유저 설정
정상적으로 설치하였다면 컨테이너 내부에 접속하여 superuser를 생성한다. 슈퍼유저는 관리자 패널을 사용하여 사용자에게 그룹을 할당할 수 있다고 한다.
```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

여기에서 ID와 비밀번호를 입력하라는 메세지가 뜰 텐데 비밀번호가 너무 짧으면 경고가 나온다.
하지만 y를 누르면 그대로 진행이 가능하다.


### 크롬 브라우저 설치
CVAT는 크롬 브라우저만 지원한다고 한다.
```bash
curl https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
sudo apt-get update
sudo apt-get --no-install-recommends install -y google-chrome-stable
```

### CVAT 접속
CVAT의 포트는 기본적으로 8080으로 설정된다. 로컬에서 실행한다면 localhost:8000으로 접속하고 만약 다른 기기에서 접속하고 싶다면 IP:8080을 크롬 브라우저에 입력한다.(ex. 10.10.10.10:8080)

여기에서 ID와 비밀번호를 입력하는 창이 나온다면 성공한 것이다. 앞서 설정했던 슈퍼유저 ID와 비밀번호를 입력하면 CVAT를 사용할 수 있다.  

### AI 모델 사용을 위한 추가 설치
여기까지가 기본적인 설치 방법이다.  
하지만 아직 부족한 점이 있다.  
AI 모델은 어떻게 사용할 수 있는 걸까?
이를 위해서는 추가적인 설치가 필요하다.  

아래의 명령어를 입력하여 컨테이너의 작동을 중단시키자.
```bash
docker compose down
```

docker ps를 입력하면 아까와 달리 여러 개의 컨테이너가 보이지 않을 것이다.
</br>

여기에서 아래의 명령어를 실행한다. 경로에 유의하자.
```bash
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```
</br>

이제 만약 컨테이너의 작동을 멈추고 싶다면 아래의 명령어를 실행한다.
```
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down
```
</br>

딥러닝 모델 deploy를 도와줄 nuclio가 필요하다. 아래의 version은 docker-compose.serverless.yml 파일을 열어서 확인하도록 하자.

```bash
wget https://github.com/nuclio/nuclio/releases/download/<version>/nuctl-<version>-linux-amd64
```
</br>

* components/serverless/docker-compse.serverless.yml 일부
```yaml
services:
  nuclio:
    container_name: nuclio
    image: quay.io/nuclio/dashboard:1.11.24-amd64
    restart: always
```
</br>

예시의 경우 1.11.24이므로 아래와 같이 실행하면 된다.
```bash
wget https://github.com/nuclio/nuclio/releases/download/1.11.24/nuctl-1.11.24-linux-amd64
```

이후 권한설정 명령을 실행한다. version은 앞선 버전과 동일하게 작성하면 된다.
```bash
sudo chmod +x nuctl-<version>-linux-amd64
sudo ln -sf $(pwd)/nuctl-<version>-linux-amd64 /usr/local/bin/nuctl
```

### AI 모델 deploy
이후 AI 모델을 deploy한다. 아래는 예시이다.
```
./serverless/deploy_cpu.sh serverless/openvino/dextr
./serverless/deploy_cpu.sh serverless/openvino/omz/public/yolo-v3-tf
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam
./serverless/deploy_gpu.sh serverless/pytorch/foolwood/siammask
```

이런 식으로 하나하나 모델을 deploy하게 되면 그 모델을 라벨링에 사용할 수 있다.  
해당 형식에 맞게 도커 이미지나 export할 모델 형식 및 다운로드 링크 등등도 사용자가 설정하면 custom 모델도 사용이 가능한 것으로 보인다.(다만 어려울 수 있음)    

## 발생할 수 있는 이슈
* CVAT 구동에는 약간의 시간이 걸린다. 바로 접속이 되지 않는다면 조금 기다린 후에 접속한다.
* docker logs [컨테이너명] 을 통해 어디에서 오류가 나는지 확인 할 수 있다.
* 가동되는 서버의 디스크 용량이 90% 이상 차 있을 경우 CVAT 서버는 가동되지 않을 수 있다.
* 지원되는 여러 AI 모델을 deploy할 수 있으나 각각이 차지하는 도커 이미지의 용량이 꽤 커서 문제가 발생할 수 있다.

## 추가
CVAT는 여러 기능을 제공하는 툴로서 서버에 CVAT를 구축해 놓고 여러 기기에서 접근하여 라벨링이 가능한 장점이 있다.  
다만 특정 AI 모델을 사용한 라벨링 시 (ex. SAM) 시간이 생각보다 오래 걸린다는 단점이 존재한다.  


