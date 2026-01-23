---
layout: single
title: "Colmap 사용법 (for Gaussian Splatting)"
categories:
    - ComputerVision
tag: [colmap, gaussian splatting, 3d]
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---

Gaussian Splatting을 사용하기 위해서는 pose가 필요하다. 이 때 도움을 주는 툴이 Colmap이다. colmap은 거의 GS에 반필수적인 요소이다.

이미지가 있다고 가정할 떄 GS를 위한 GUI를 이용한 colmap 사용법과 주의점 등을 정리해보고자 한다.
(자동화가 필요하다고 하면 코드를 통해서도 가능하다.)

---

### 사족

사족을 붙이자면, pose란 카메라가 공간상에서 어디에 있고, 어디를 보고 있는지를 나타내는 위치와 자세 정보이다.

먼저 카메라 좌표계에서 월드 좌표계로 변환하기 위한 4x4행렬이 필요하다.

$$T = \begin{bmatrix}R&t\\0&1\\ \end{bmatrix}$$

이를 extrinsic parameter(외부 파라미터)라 한다. 보통 extrinsic이라 부른다.

intrinsic parameter는 카메라 자체의 광학적 특성을 나타내는 3x3행렬, 왜곡 계수로 표현된다. 행렬 $K$는 필수적이고 왜곡계수는 말그대로 카메라의 왜곡을 다항식으로 근사하고 그때의 계수인데 왜곡되지 않은 이미지를 얻으려면 필요하나 필수는 아니다.

$$K = \begin{bmatrix}f_{x}&0&c_{x}\\0&f_{y}&c_{y}\\0&0&1\\ \end{bmatrix}$$

---

### 사용법

Ubuntu에 colmap이 설치되어 있다고 가정한다. (경험상 colmap 버전별, GPU 사용/미사용에 따라 성능이 조금씩 다르다. 최신버전이라고 더 feature를 잘 찾아주는 것이 아니다)

<br>

**1. colmap을 통해 실행할 폴더를 생성해주고 안에 images에 원본 이미지를 images_2에 resize된(2배 다운샘플링) 이미지를 넣어준다.(downsampling은 필수아님)**

![Folder structure]({{"/assets/images/post-colmap/step1_folder_structure.jpeg" | relative_url}})

<br>

**2. colmap을 실행한다.**

```bash
colmap gui
```

<br>

**3. 프로젝트 설정**

1. New project를 선택하고 Database탭에 New버튼을 눌러 images의 상위 폴더를 지정하여 이곳에 database.db로 저장해준다. (colmap이 이미지간 feature matching결과와 절차를 저장하는 데이터베이스 파일이다.)
2. Images 탭에 Select를 선택하여 images폴더를 지정해준다.
3. Save를 눌러준다. database.db파일이 생성된 것을 확인해주자.

![New project]({{"/assets/images/post-colmap/step3_new_project.jpeg" | relative_url}})

![Database setup]({{"/assets/images/post-colmap/step3_database_setup.jpeg" | relative_url}})

![Select images folder]({{"/assets/images/post-colmap/step3_select_images_folder.jpeg" | relative_url}})

![Database created]({{"/assets/images/post-colmap/step3_database_created.jpeg" | relative_url}})

![Project complete]({{"/assets/images/post-colmap/step3_project_complete.jpeg" | relative_url}})

![Project view]({{"/assets/images/post-colmap/step3_project_view.jpeg" | relative_url}})

<br>

**4. Feature Extraction**

1. Processing - Feature Extraction 선택
2. 카메라 모델에 맞는 카메라 모델 선택. 일반적인 카메라라면 핀홀 카메라이므로 PINHOLE 혹은 SIMPLE_PINHOLE 선택. 만약 fisheye라면 OPENCV_FISHEYE 선택
3. Shared for all images 선택(폴더에 대해 같은 파라미터를 공유함) 및 다른 파라미터 원하는대로 설정 후 Extract.
   - 주의점으로 [colmap 공식문서](https://colmap.github.io/tutorial.html) 참고

![Feature extraction menu]({{"/assets/images/post-colmap/step4_feature_extraction_menu.jpeg" | relative_url}})

![Feature extraction settings]({{"/assets/images/post-colmap/step4_feature_extraction_settings.jpeg" | relative_url}})

![Feature extraction result]({{"/assets/images/post-colmap/step4_feature_extraction_result.jpeg" | relative_url}})

<br>

**5. Feature matching**

각 feature point를 매칭해주는 과정이다.

1. Processing - Feature matching 선택
2. Exhaustive(모든 이미지를 사용) 를 선택하고 Run (만약 데이터셋이 너무 크다면 방식이나 parameter를 조절한다.)

![Feature matching menu]({{"/assets/images/post-colmap/step5_feature_matching_menu.jpeg" | relative_url}})

![Feature matching result]({{"/assets/images/post-colmap/step5_feature_matching_result.jpeg" | relative_url}})

+) Processing-Database management를 통해 이미지별로 어떤 이미지의 특징점들과 매칭이 되었는지 확인할 수 있다. Matches 탭은 이미지 pair간 특징점 매칭이 시도된 결과를 나타내고 Two-view geometries 탭은 Matches 탭의 결과에서 기하학적 검증을 수행하여 이후 Reconstruction에 사용할 유효한 매칭에 해당한다.

![Database management]({{"/assets/images/post-colmap/step5_database_management.jpeg" | relative_url}})

<br>

**6. Reconstruction**

다른 시점에서 촬영된 이미지들을 Structure-from-Motion (SfM)을 통해 reconstruction을 수행한다.

1. Reconstruction - Start reconstruction 을 선택하여 reconstruction을 수행한다. 이 과정은 시간이 오래 걸릴 수 있다.

![Reconstruction start]({{"/assets/images/post-colmap/step6_reconstruction_start.jpeg" | relative_url}})

![Reconstruction progress]({{"/assets/images/post-colmap/step6_reconstruction_progress.jpeg" | relative_url}})

![Reconstruction result]({{"/assets/images/post-colmap/step6_reconstruction_result.jpeg" | relative_url}})

<br>

**7. Export**

1. File - Export model을 선택한다.
2. images가 있는 폴더에 sparse 라는 이름을 가진 폴더를 생성해준다.
3. 그 안에 0라는 폴더를 하나 더 만들어주고 여기에 export한다.
4. 그 안에 cameras.bin, images.bin, points3D.bin이 생성되었는지 확인한다. (txt확장자로 생성되어도 된다. 나머지는 gaussian splatting에 필수적인 요소는 아니다.)

![Export menu]({{"/assets/images/post-colmap/step7_export_menu.jpeg" | relative_url}})

![Sparse folder]({{"/assets/images/post-colmap/step7_sparse_folder.jpeg" | relative_url}})

![Export path]({{"/assets/images/post-colmap/step7_export_path.jpeg" | relative_url}})

![Export result]({{"/assets/images/post-colmap/step7_export_result.jpeg" | relative_url}})

---

### Colmap data 살펴보기

<br>

**cameras.bin**

cameras.bin 파일을 읽어서 하나만 살펴보면 아래와 같이 카메라 고유 ID, 카메라 모델 종류, 이미지 w,h , 내부 파라미터를 가지고 있는 것을 알 수 있다.

사실 intrinsic의 경우에는 colmap을 통해 추정을 했지만 실제로는 카메라 사에서 자체적으로 제공한다던지 혹은 calibration을 통해 어느 정도 정확한 $f_{x}$, $f_{y}$, $c_{x}$, $c_{y}$ 를 알고 있는 경우가 종종 존재한다. 이 경우에는 cameras.bin 파일을 미리 형식에 맞춰서 변환해주는 것이 좋다. (물론 초점거리를 계속 변화시킬 수도 있기 때문에 이는 상황에 맞게 설정하자)

```
Camera(id=75, model='PINHOLE', width=1920, height=1080, params=array([1650.65533107, 1667.17165573, 960. , 540. ]))
```

<br>

**images.bin**

images.bin 파일은 각 이미지의 extrinsic 정보를 담고 있다.
- rotation (qvec, 쿼터니언으로 표현)
- translation (tvec)

**주의할점**: extrinsic이 정의하는 변환 방향은 World to Camera이다. 사람에 따라서 world2cam이렇게 쓰기도 하고 cw라 쓰기도 한다. 이는 행렬의 특성때문에 순서가 이렇게 된것으로, world2cam, cam2lidar가 있다고 하면 world2lidar를 구하려면 간단히 lc * cw = lw : world2lidar가 되기 때문이다.

- camera_id: cameras.bin의 어떤 intrinsic을 쓸것인지
- name: 이미지명
- xys: 이 이미지에서 추출된 2D 키포인트 좌표들 \[x,y]꼴
- points3D_ids: 각 2D 키포인트가 어떤 3D 포인트에 매칭되는지 (매칭안되는 원소는 -1)

```
Image(id=75, qvec=array([0.96563358, 0.02816726, 0.23713259, 0.10259884]), tvec=array([ 2.52792054, -0.99279952, 2.96554468]), camera_id=75, name='000074.png', xys=array([[ 32.46893311, 2.05210352], [ 1.63333821, 14.8482132 ], [ 106.16053009, 23.63771057], ..., [ 304.80853271, 455.40713501], [1088.6776123 , 712.21472168], [ 584.40997314, 326.87509155]]), point3D_ids=array([-1, -1, -1, ..., -1, -1, -1]))
```

<br>

**points3D.bin**

points3D.txt 파일로 대체하여 설명한다. reconstruction의 결과이며 중요한 것은 위치 (X,Y,Z), 색상(R,G,B) 이다.

error는 알아두면 좋은 부분인데 reprojection error로 어떤 3D point가 관측될 때 실제 이미지에서 키포인트 u,v와 3D point를 intrinsic으로 투영했을때의 u,v의 거리를 구한것(여러 이미지에서 관측이면 평균)으로 차이가 작을수록 여러 뷰에서 매칭이 잘 되었다고 할 수 있다. 이를 이용하여 threshold를 두어 값이 크면 버릴 수도 있다.

```
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 11642
1 3.9798015797131929e-002 -7.1024961247506124e-002 -9.2437755061088991e-002 38 17 14 0 26 0 25 0
2 5.7785406251221381e-002 -9.8236976537123433e-002 -9.5571134164849122e-002 94 38 24 0 26 1 25 1
3 -0.11885891392474095 -0.12020242255965741 8.644229616666578e-002 15 23 28 0 26 2 25 2
```

---

### Colmap txt변환 (.bin->.txt)

아무래도 bin파일은 커스텀 read함수가 필요하다 보니 txt파일로 바꾸는게 알아보기 쉬울때가 많다. colmap은 txt변환을 내부적으로 지원한다.

```bash
colmap model_converter --input_path [bin파일있는 경로] --output_path [저장경로] --output_type TXT
```
