---
layout: single
title: "ROS2-003. Turtlesim 01 - Turtlesim과 서비스"
categories:
    - Robot
tag: ROS2
author_profile: false
sidebar:
    nav: "docs"
---

```bash
ros2 run turtlesim turtlesim_node
```

```bash
ros2 node info /turtlesim
```

```
/turtlesim
  Subscribers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /turtle1/cmd_vel: geometry_msgs/msg/Twist
  Publishers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
    /turtle1/color_sensor: turtlesim/msg/Color
    /turtle1/pose: turtlesim/msg/Pose
  Service Servers:
    /clear: std_srvs/srv/Empty
    /kill: turtlesim/srv/Kill
    /reset: std_srvs/srv/Empty
    /spawn: turtlesim/srv/Spawn
    /turtle1/set_pen: turtlesim/srv/SetPen
    /turtle1/teleport_absolute: turtlesim/srv/TeleportAbsolute
    /turtle1/teleport_relative: turtlesim/srv/TeleportRelative
    /turtlesim/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /turtlesim/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
    /turtlesim/get_parameters: rcl_interfaces/srv/GetParameters
    /turtlesim/list_parameters: rcl_interfaces/srv/ListParameters
    /turtlesim/set_parameters: rcl_interfaces/srv/SetParameters
    /turtlesim/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
  Service Clients:

  Action Servers:
    /turtle1/rotate_absolute: turtlesim/action/RotateAbsolute
  Action Clients:
```

<br>

```bash
ros2 service list
```

* 서비스: 두 노드 간에 데이터를 주고받는 방식 (client가 server에 request하고 response받음)

```bash
ros2 service type /turtle1/teleport_absolute
```

teleport_absolute 서비스의 데이터 정의를 확인

`turtlesim/srv/TeleportAbsolute` 패키지명/폴더명/서비스타입 이름 (.srv파일명)
.srv파일에 저장을 data type을 해둠

![TeleportAbsolute service definition]({{"/assets/images/post-ros2-turtlesim/teleport_absolute_srv.jpeg" | relative_url}})

그림: github 경로 및 srv파일내용

`---`를 기준으로 윗부분은 request데이터, 아래부분은 Response데이터이다.
즉 이 서비스에서는 x,y,theta를 입력으로 받으며 return은 없는 함수이다. (단순 이동만 하므로 굳이 반환 안해도 된다는 컨셉)

```bash
ros2 interface show turtlesim/srv/TeleportAbsolute
```

* 두 개의 명령어를 한번에 연결

```bash
ros2 interface show $(ros2 service type /turtle1/teleport_absolute)
```

<br>

![Robot coordinate system]({{"/assets/images/post-ros2-turtlesim/robot_coordinate_system.jpeg" | relative_url}})

보통 모바일 로봇의 좌표계는 바라보는 방향이 x축, 각도는 반시계 방향이 +이며 x축에서 반시계 방향으로 90의 위치가 y축이다. 보통 ROS에서는 radian단위를 사용한다.

<br>

**서비스 호출**

`ros2 service call <서비스명> <서비스정의> "데이터"`

```bash
ros2 service call /turtle1/teleport_absolute turtlesim/srv/TeleportAbsolute "{x: 0.5, y: 0.5, theta: 1.57}"
ros2 service call /turtle1/teleport_absolute turtlesim/srv/TeleportAbsolute "{x: 2, y: 1, theta: 1.57}"
```

* 주의점: {x: 2, y: 1, theta: 1.57}에서 `:` 왼쪽은 붙이고 오른쪽은 띄우기

![Teleport service result]({{"/assets/images/post-ros2-turtlesim/teleport_result.jpeg" | relative_url}})

좌측하단=(0,0)

<br>

**서비스 리셋**

```bash
ros2 service type /reset std_srvs/srv/Empty
```

<br>

**네임스페이스**

```
/turtle1/cmd_vel
/turtle2/cmd_vel
```

`cmd_vel`이라는 동일한 토픽 이름이지만, 각각의 네임스페이스로 구분됨. turtle1, turtle2가 각각 자신의 속도 명령을 구분해서 받는 것

```bash
ros2 service type /spawn
ros2 interface show turtlesim/srv/Spawn
ros2 service call /spawn turtlesim/srv/Spawn "{x: 3, y: 3, theta: 0, name: ''}"
```

새로운 거북이가 생성됨

![Spawn new turtle]({{"/assets/images/post-ros2-turtlesim/spawn_turtle.jpeg" | relative_url}})
