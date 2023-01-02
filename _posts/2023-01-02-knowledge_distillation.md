---
layout: single
title:  "지식 증류(Knowledge Distillation)"
categories: 
    - deep learning
tag:
    - [knowledge distillation]
author_profile: false
sidebar:
    nav: "docs"
---

# 지식 증류(Knowledge Distillatioin)
딥러닝에서 지식 증류란 teacher model **T**로부터 추출한 지식을 student model **S**로 전수하는 것을 의미한다.  
이 개념은 Hinton 등이 2015년에 발표한 Distilling the knowledge in a neural network(https://arxiv.org/abs/1503.02531) 에 소개되었다.  

간단한 배경으로는 앙상블과 같은 복잡한 모델의 경우 성능은 뛰어나나 이를 일반 사용자용으로 배포하는 것은 부담이 크기 때문에 배포를 위해 
간단한(inference가 빠른) 모델에 복잡한 모델의 지식을 전수한다. 

이 분야는 최근까지도 꾸준히 연구되어 다양한 아이디어들이 제시되고 있지만 여기서는 기본적인 지식 증류의 원리를 정리하고자 한다.

## soft label
softmax를 출력층의 활성함수로 사용하는 분류 문제를 생각해 보면, 최종적으로 각 클래스에 대한 확률(0~1)을 산출하게 된다.  
A=0.1, 
가장 높은 확률을 가지는 클래스가 해당 class로 결정되게 되는데, 지식 증류에서는 다른 클래스에 대한 값들도 
출력값의 분포를 soft하게 만드는지 확인하기 위해 간단한 코드를 작성해보았다.

```python
import numpy as np

for i in range(1,4):
    for T in range(1,6):
        q = np.exp(i/T)/(np.exp(1/T)+np.exp(2/T)+(np.exp(3/T)))
        print('i=', i, 'T=', T, q.round(3))
```
실행해 보면 T가 커질수록 기존의 softmax값이 큰 것은 그 값을 낮추고 작은 것은 높여서 분포를 전체적으로 완화하는 것을 확인할 수 있다.

## distillation loss