---
layout: single
title:  "Hex 컬러 코드 rgb로 변환하기"
categories: 
    - Python
tag:
    - [python, color]    
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---
## 헥스 코드란
이미지 관련 작업을 하다 보면 ```#e6194b```와 같은 색상 코드를 마주할 일이 있다.  
이는 16진수로 표현된 색이다.  
```cv2.puttext```와 같은 함수를 사용할 때는 RGB 형식의 튜플로 변환해 줄 필요가 있다.  
아래는 RGB 형식의 튜플로 변환해 주는 함수 및 예시이다.

```python
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
```

```python
import numpy as np
import matplotlib.pyplot
DEFAULT_COLOR_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

def fill_color(img, color_hex):
    rgb_color = hex_to_rgb(color_hex)
    print('rgb_color',rgb_color)
    img[:, :] = rgb_color

img = np.zeros((256, 256, 3), dtype=np.uint8)

# 색상 코드 랜덤 추출
n = np.random.randint(0,len(DEFAULT_COLOR_PALETTE))
color_hex = DEFAULT_COLOR_PALETTE[n]

# 색상 코드를 RGB 튜플로 변환하여 이미지에 채우고 시각화
fill_color(img, color_hex)
plt.imshow(img)
```

예시 실행 결과:
<center><img src='{{"/assets/images/post-hex2rgb/result.PNG" | relative_url}}' width="90%"></center>
<br>



## 작동 원리
코드가 정상적으로 돌아가는 것을 확인했다면 어떻게 6자리의 문자열로 색상을 표현하는 것인지 알아보자.  
헥스 코드, 16진수에 답이 있다.  
```#e6194b```에서 #을 제외하고 ```e6```, ```19```, ```4b```가 각각 R, G, B에 색상값에 해당한다.  
16진법을 사용하며 0~9는 그대로, 10~15는 a~f를 사용한다.

```python
print(int("a",16)) # 10
print(int("f",16)) # 15
```

```e6```은 $14\times 16^{1}+6\times 16^{0}=230$,  
```19```은 $1\times 16^{1}+9\times 16^{0}=25$,  
```4b```은 $4\times 16^{1}+11\times 16^{0}=75$ 가 되어 (230,25,75)가 된다.  
두자리 까지 표시하므로 0~255의 값을 표현할 수 있고 최소값은 00, 최대값은 ff로 각각 0,255가 된다.
