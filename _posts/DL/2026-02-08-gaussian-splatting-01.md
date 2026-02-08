---
layout: single
title: "Gaussian Splatting (01) - 논문 리뷰"
categories:
    - DL
tag: gaussian splatting
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---

Gaussian Splatting(GS)는 3D reconstruction의 한 분야로 [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) 논문에서 소개된 획기적인 기법이다. 해당 기법의 출시 이후 NeRF를 연구하던 연구자들이 대부분 Gaussian splatting쪽으로 넘어갔다.

---

### NeRF(Neural Radiance Field)와의 차이점

Radiance는 Ray casting이라고도 하며 카메라에서 특정 화소를 통과하는 광선을 따라 여러 지점을 일정 간격으로 샘플링하고 (원 논문에서는 128개로 샘플링) 이 점들의 ($x$, $y$, $z$, $\theta$, $\pi$)가 주어질 때 MLP를 통해 $RGB\sigma$ (color, density)를 예측하는 것이다.

즉 NeRF는 scene(3D 공간) 자체를 학습하는 것은 아니고 새로운 시점에서 그 scene을 표현할 수 있는 모델의 파라미터를 학습하는 것이다.

GS는 아래에서 설명하겠지만 scene 자체를 3D gaussian으로 간주하고 gaussian의 파라미터(즉, scene 자체)를 학습한다.

Novel view에서 NeRF는 결국 딥러닝 모델의 추론이 필요하기 때문에 실시간 렌더링은 어려운 단점이 있었지만 GS는 실시간 렌더링이 가능하여 이 점에서 큰 이점이 있다. (**즉 3DGS는 딥러닝이 아니다!**)

<br>

#### TMI

실제 NeRF 구현에서는 $\theta$, $\pi$ 는 방향벡터로 바뀐다 $d_{x}$

실제로는 rays_o, rays_d 임베딩 벡터로 변환되고 concat되어 보통
- B = 배치 (N_rays * N_samples를 B로 쪼갬)
- input shape : (B, 63+27)
- output shape: (B, 4)

가 된다.

---

### 설명

Gaussian spatting은 딥러닝이 아닌 일종의 최적화이지만 GPU를 사용한다.
(미분 가능한 렌더링 기반의 연속 최적화 문제를 GPU에서 푼다)

우선 공간에 시작할 3D 가우시안을 뿌린다. 여기에서 최적화할 수 있는 것은 position, scale, rotation, color(SH coefficient), opacity이다. scale, rotation을 묶어 covariance라고 한다.

![Gaussian splatting overview]({{"/assets/images/post-gaussian-splatting/gs_overview.jpeg" | relative_url}})

\[그림] Gaussian splatting overview

![Optimization algorithm]({{"/assets/images/post-gaussian-splatting/gs_optimization_algorithm.jpeg" | relative_url}})

\[그림] 최적화 알고리즘

<br>

3D 가우시안이 무엇이냐?

![3D Gaussian visualization]({{"/assets/images/post-gaussian-splatting/3d_gaussian_ellipsoid.jpg" | relative_url}})

\[그림] 3D Gaussian 시각적으로 보기

그림을 보면 타원체(ellipsoid)가 여러 개 있는 것을 볼 수 있다. (이것이 mesh처럼 실체화될 수 있는 실제로 존재하는 object는 아니고 rendering의 재료라고 보는 것이 정확하다.)

SFM(Structure from motion)을 통해 매칭된 포인트를 가우시안의 중심으로 3D gaussian을 생성하고 이들의 위치, 공분산, 색상, 투명도를 최적화하는 것이다. (\*최적화하면서 개수가 늘어날 수 있다.)

최적화 방식은 렌더링 시에 보는 시점에서 3D gaussian을 2D로 projection하고 (overview 그림의 2D projection 부분) 이들을 rasterization을 통해 alpha blending해서 생성된 view를 원본 이미지와 최대한 비슷하게 만드는 것을 목표로 한다. (논문에서 $\lambda$ 는 0.2)

$$
L = (1 - \lambda)\,L_1 + \lambda\,L_{\mathrm{D\text{-}SSIM}}
$$

이 때 이미지를 만드는 부분을 rasterization이라고 하며 이 과정은 전통적인 rasterization과 다르게 gaussian의 연속 분포를 화면에 splat하는 방식이다. Rasterization에 대해서는 할 이야기가 많은데 다음 포스팅에서 설명하고자 한다.

<br>

\* 보충 설명을 하자면 최적화 중 densification(split, clone, prune)을 통해 가우시안의 개수가 증가될 수 있는데 rasterization을 통해 개별 gaussian의 gradient를 구할 수 있다.

사용자 지정 threshold를 통해 너무 크거나 투명도가 일정 수치 이하면 제거하고(prune), positional gradient가 일정 threshold 이상인 gaussian은 해당 영역의 재구성이 덜 된 것으로 판단하고(gaussian이 큰 경우=over-reconstruction, 작은 경우=under reconstruction) 큰 gaussian은 2개로 나누고(split), small gaussian은 gradient 방향으로 clone한다.

보통 이 과정을 수행하면 gaussian의 개수가 늘어나게 된다.

이 과정은 처음에 소개한 overview 그림의 Adaptive Density Control 부분에 해당한다. (일정 threshold라는 부분에 불만을 느껴 이를 수학적으로 더 해석하고자 하는 연구도 존재한다.)

<br>

결과적으로는 NeRF 계열 모델들에 대해 이미지 복원 성능에 대한 정량적인 지표가 비슷하거나 우수하고 렌더링 속도(FPS)는 훨씬 빠르다.

![NeRF comparison table]({{"/assets/images/post-gaussian-splatting/nerf_comparison_table.jpg" | relative_url}})

---

### Reference

* https://arxiv.org/abs/2308.04079
