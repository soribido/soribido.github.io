---
layout: single
title: "ROS2-006. Turtlesim 04 - 액션"
categories:
    - Robot
tag: ROS2
author_profile: false
sidebar:
    nav: "docs"
---

![Action concept]({{"/assets/images/post-ros2-action/ros2_action_concept.gif" | relative_url}})

\[그림] Action의 개념

1. Action 서버를 구현하는 노드에 클라이언트 노드가 먼저 서비스를 목표로 요청(request)
2. 서버가 응답(response)
3. 목표(Goal)을 달성할때까지 중간을 토픽으로 피드백해줌(feedback topic)
4. 끝나면 Result 서비스를 이용

<br>

![Action structure]({{"/assets/images/post-ros2-action/ros2_action_structure.jpeg" | relative_url}})

\[그림] Action 구성 (간략)

<br>

```bash
# 터미널 1
ros2 run turtlesim turtlesim_node
```

```bash
# 터미널 2
ros2 run turtlesim turtle_teleop_key
```

키보드의 화살표로 움직일 수 있는 상태가 됨

```bash
# 터미널 3
ros2 action list
```

명령을 수행하면 `/turtle1/rotate_absolute` 라는 결과가 나오고 이 데이터 타입을 알려면 info 옵션을 사용하거나 `ros2 action list -t`를 통해 알 수 있다.
그 결과 `turtlesim/action/RotateAbsoulte`라는 데이터 타입을 사용함을 알 수 있다.

```bash
ros2 interface show turtlesim/action/RotateAbsoulte
```

![Interface show result]({{"/assets/images/post-ros2-action/ros2_action_interface_show.jpeg" | relative_url}})

theta가 Goal (최종 각도), delta가 출발 위치에서 각도 차이, remaining이 남은 각도(feedback)에 해당한다.

<br>

`ros2 action send_goal <action_name> <action_type> <values>`

```bash
ros2 action send_goal /turtle1/rotate_absolute turtlesim/action/RotateAbsolute "{theta: 3.14}"
```

![Send goal result]({{"/assets/images/post-ros2-action/ros2_action_send_goal.jpeg" | relative_url}})

<br>

```bash
rqt_graph
```

![RQT graph action]({{"/assets/images/post-ros2-action/ros2_action_rqt_graph.jpeg" | relative_url}})
