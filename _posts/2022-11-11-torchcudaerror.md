---
layout: single
title:  "파이토치(PyTorch) CUDA error: no kernel image is available for execution on the device"
categories: 
    - Error
tag:
    - [PyTorch, CUDA, GPU]    
author_profile: false
sidebar:
    nav: "docs"
---

* 상황 : 도커 컨테이너에서 torch로 GPU 연산 수행 시 CUDA error: no kernel image is available for execution on the device 오류 발생  
* 원인 : 서버 1에서 사용하던 이미지를 서버2에 복사하여 새 도커 컨테이너 환경 구축하였지만 GPU 오류 발생.(GPU에 맞는 환경변수가 달라서 오류 발생)  
* 해결 방법  
1. [!url](https://developer.nvidia.com/cuda-gpus) 에 접속하여 사용하고자 하는 GPU의 Compute Capability 확인(ex RTX 3090 = 8.6)  
2. docker exec -it [컨테이너명] bash 로 컨테이너 내부 접속후 export TORCH_CUDA_ARCH_LIST=8.6 입력(본인 GPU 해당 숫자로 입력, 3090의 경우8.6)  
3. 컨테이너 내부에서 pytorch 삭제 후 재설치

정상 동작 확인  
```python
import torch
torch.rand(5).to("cuda:1") # 1번 GPU 사용(미지정시 기본적으로0)
```

출력결과: tensor([0.3369, 0.8064, 0.1281, 0.8275, 0.7496], device='cuda:1')