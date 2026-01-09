---
layout: single
title:  "[Python]Traceback을 이용한 오류 출력"
categories: 
    - Python
tag:
    - [python]    
author_profile: false
sidebar:
    nav: "docs"
---

# Traceback을 사용한 오류 출력
Traceback을 이용하여 오류를 출력할 수 있다.  
아래의 예시를 보자.  
```python
import traceback

def func_divide0():
    a = 1 / 0  # ZeroDivisionError 

try:
    func_divide0()
except Exception as e:
    # Use traceback to print detailed information about the exception
    traceback.print_exc()
    
    # Save as variable and print
    # trace = traceback.format_exc()
    # print(trace)
```
func_devide0 함수는 0으로 나눗셈을 하기 때문에 오류가 날 것이다.  
이에 대한 오류를 보고 싶을 때, traceback을 이용하여 오류를 출력할 수 있다.  
예시의 주석처럼 변수로 처리할 수도 있다.

예시는 굉장히 단순한 것이어서 traceback을 사용하지 않아도 오류가 출력될 수 있지만  
코드의 구조가 복잡하여 main.py를 구동하면 여러 코드들이 같이 돌아가는 경우 유용하게 쓰일 수 있다.


