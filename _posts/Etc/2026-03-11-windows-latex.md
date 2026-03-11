---
layout: single
title: "Windows LaTeX 설정 (with VSCode)"
categories:
    - etc
tag: [latex, vscode, tex, prism]
author_profile: false
sidebar:
    nav: "docs"
---

논문 작업을 위해서 LaTeX를 사용할 경우가 있다.  
보통은 Overleaf와 같은 서비스를 이용하지만 때로는 로컬 작업이 필요할 수도 있다.  
이런 경우에 Windows 기준 세팅법을 정리하고자 한다.

## 1. TeX 배포판(엔진) 설치
* https://www.tug.org/texlive/windows.html 접속
* Easy install 실행파일(`install-tl-windows.exe`) 다운로드
* 기본 설정으로 변경없이 설치 (설치 시간이 생각보다 오래 걸리니 주의)
* 버전확인
```
pdflatex --version
latexmk --version
```

Note) 배포판으로 TeXLive를 선택하였지만 MiKTex를 사용하여도 된다.
## 2. VSCode Extension 다운로드
* Extension - LaTeX Workshop 설치

## 3. 테스트
* VSCode에서 tex파일을 빌드한다. (Build LaTeX Project 클릭 or Ctrl + Alt + B)
* 빌드된 pdf파일을 본다. (View LaTeX PDF file 클릭 or Ctrl + Alt + V)


## TMI)
최근(26년 1~2월경) OpenAI에서 prism이라는 것을 공개하였다.  
Overleaf처럼 웹에서 latex형식으로 논문을 편집할 수 있는데 바로 채팅을 통해 모델에게 수정을 요청할 수 있다. 또한 overleaf와 유사하게 링크를 통해 다른 사람을 초대하여 같이 작업이 가능하다.  
개인적인 감상으로는 현재는 latex 형식에 특화된 편집 도우미 느낌이 좀 강하며 내용의 전문성이나 관련 논문 검색의 경우 약간 아쉬울 수 있으며, 질문에 대한 처리 시간이 좀 걸린다.
