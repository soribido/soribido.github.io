---
layout: single
title: "ROS2-004. Turtlesim 02 - 토픽"
categories:
    - Robot
tag: ROS2
author_profile: false
sidebar:
    nav: "docs"
---

![Topic concept]({{"/assets/images/post-ros2-topic/topic_concept.gif" | relative_url}})

\[그림] Topic

Topic: 토픽의 이름과 데이터(메시지)의 구조만 알고 있으면 누구나 구독 가능

```bash
ros2 run turtlesim turtlesim_node
ros2 topic list
```

```
/parameter_events
/rosout
/turtle1/cmd_vel
/turtle1/color_sensor
/turtle1/pose
```

<br>

Topic 역시 type조회 가능

```bash
ros2 topic type /turtle1/pose
```

```
turtlesim/msg/Pose
```

<br>

귀찮을 경우

```bash
ros2 topic list -t
```

```
ros2 topic list -t
/parameter_events [rcl_interfaces/msg/ParameterEvent]
/rosout [rcl_interfaces/msg/Log]
/turtle1/cmd_vel [geometry_msgs/msg/Twist]
/turtle1/color_sensor [turtlesim/msg/Color]
/turtle1/pose [turtlesim/msg/Pose]
```

<br>

토픽은 publisher랑 subscriber가 있는데 구분해서 볼 수 있으면 좋다.

```bash
ros2 topic info /turtle1/pose
```

<br>

`/turtle1/cmd_vel` 토픽의 메세지 type은 `geometry_msgs/msg/Twist`이고 `interface show` 명령어로 보면 linear, angular 2개의 벡터를 인자로 받는다.

```bash
ros2 topic pub --once /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

![Topic publish with --once option]({{"/assets/images/post-ros2-topic/topic_once_result.jpeg" | relative_url}})

--once 옵션을 사용했으므로 거북이가 x축으로 2만큼 이동한다.

<br>

만약 --rate 옵션을 사용하면 rate 주기로 토픽을 발행한다.

```bash
ros2 topic pub --rate 1 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 1.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}"
```

rate가 1이므로 1Hz주기로 토픽을 발행한다.

![Topic publish with --rate option]({{"/assets/images/post-ros2-topic/topic_rate_circle.jpeg" | relative_url}})

<br>

### Reference

https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html
