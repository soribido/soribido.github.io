---
layout: single
title: "[실험] LiDAR-Image fusion-based 3D scene reconstruction using point upsampling network"
categories:
    - Robot
tag: [lidar-image fusion, point upsampling, projection, lidar, reconstruction]
author_profile: false
sidebar:
    nav: "docs"
---

단일 이미지와 sparse한 lidar points를 fusion하여 3d scene을 구성하고자 한 실험 일부 기록용

LiDAR가 16채널이라 조밀하지 않은 포인트들로 구성되어 있는데 이를 point upsampling 네트워크를 쓰면 dense한 포인트를 만들어 낼 수 있지 않을까 하는 생각에서 출발.

## Process

1. Robot dog에서 front image랑 global_x,y,z, 이미지상 좌표(uv), RGB값, 로봇 odom_yaw,x,y, local x,y,z를 csv파일로 저장. (* odom_z는 0, 지면은 -0.7)

2. global_x, global_y, global_z만 따로 모아 xyz파일로 생성

3. Point upsampling 모델에 넣어 16배 upsampling (output: xyz파일)

4. csv로 기록했던 글로벌 좌표와 색은 그대로 매칭 (이것만 하면 그림1)

5. 새롭게 생성된 point cloud를 이미지상에 projection시켜 매칭되는 점은 이미지상의 색깔을 부여 (* projection을 위해서는 카메라 내부 파라미터를 알아야 함)

6. projection 실패한 포인트 (color NaN) KNN으로 최근접 색으로 보충

![Point cloud top view]({{"/assets/images/post-lidar-fusion/pointcloud_topview.jpeg" | relative_url}})

![Point cloud side view]({{"/assets/images/post-lidar-fusion/pointcloud_sideview.jpeg" | relative_url}})
