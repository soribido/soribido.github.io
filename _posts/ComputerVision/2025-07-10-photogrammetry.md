---
layout: single
title: "포토그래메트리"
categories:
    - ComputerVision
tag:
    - [photogrammetry, 3d, computer vision]
author_profile: false
sidebar:
    nav: "docs"
---

### 포토그래메트리란
* Photogrammetry: 여러 각도에서 촬영된 사진을 이용해 3D 모델을 생성하는 기술

### 포토그래메트리의 과정
1. 이미지 획득
	* 촬영 장비(드론 등)를 이용하여 대상을 촬영
	* 일반적으로 대상의 모든 면을 촬영
	* 중첩(70~80%권장), 피사계 심도 중요
	* 필요에 따라 화이트밸런스(날씨 등에 따라) 조절
    <center><img src='{{"/assets/images/post-photogrammetry/step1_capture.jpeg" | relative_url}}' width="80%"></center>

2. 특징 추출 및 이미지 정합
	* 매칭 알고리즘 (SIFT, SURF, ORB 등)을 이용하여 특징점 검출
	* 특짐점 매칭을 통해 이미지 간 상대적인 위치 관계를 추정

3. 구조 복원 (SfM; Structure from Motion)
	*  특징점 매칭 결과를 기반으로 카메라 위치와 3D 포인트 클라우드 계산
    <center><img src='{{"/assets/images/post-photogrammetry/step3_sfm.jpeg" | relative_url}}' width="80%"></center>

4. 고밀도 포인트 클라우드 생성 (MVS; Multi-View Stereo)
	*  초기 sparse 포인트 클라우드 기반으로 Dense Reconstruction 수행

5. 메쉬 재구성
	* 고밀도 포인트 클라우드로부터 surface mesh 생성
	* Possion surface reconstruction, Delaunay triangulation 등의 알고리즘 사용
    <center><img src='{{"/assets/images/post-photogrammetry/step5_mesh.jpeg" | relative_url}}' width="80%"></center>

6. 텍스쳐 매핑 (Texture Mapping)
	* 원본 이미지에서 색상 정보를 추출하여 메쉬에 입힘
    <center><img src='{{"/assets/images/post-photogrammetry/step6_texture.jpeg" | relative_url}}' width="80%"></center>

7. 후처리
	* 메쉬 간소화, hole filling, 노이즈 제거, 색 보정 등
	* 이후 export
