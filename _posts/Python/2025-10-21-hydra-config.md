---
layout: single
title: "Hydra를 통한 config관리"
categories:
    - Python
tag:
    - [python, hydra, config]
author_profile: false
sidebar:
    nav: "docs"
---

## 배경
딥러닝 코드를 보다 보면 `@hydra.main`으로 표시된 코드를 심심찮게 볼 수 있다.
config파일의 세팅을 쉽게 관리하고자 함인데, argparse나 다른 라이브러리를 편하게 써왔던 사람이라면 그대로 써도 무방하다고 생각한다.
하지만 hydra에 대해 알고 있으면 코드 파악이 쉽기 때문에 이번 기회에 hydra의 용법을 정리하고자 한다.

## 설치
```bash
pip install hydra-core
```
주의점으로는 설치는 `hydra-core`로 설치하고 hydra로 import 한다.
`omegaconf`가 설치되어 있지 않다면 omegaconf가 같이 설치된다.

## 사용 예시 1) 기본 사용법
```
hydra_test/
 ├─ config.yaml
 └─ main.py
```
<br>

**config.yaml**
```yaml
name: "soribido"
model:
  type: "resnet50"
  lr: 0.001
```
<br>

**main.py**
```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"name: {cfg.name}, model: {cfg.model.type}, lr: {cfg.model.lr}")

if __name__=="__main__":
    main()
```

내부적으로는 config_path, config_name을 os.path.join처럼 경로를 결합해서 사용한다.
config_name은 yaml을 생략해도 되나 지정해도 된다.
main의 cfg는 hydra가 yaml파일을 읽어서 생성한 config객체이고 이 객체의 타입은 앞서 import한 DictConfig가 된다.
딕셔너리와 유사한 기능을 제공하기 때문에
```python
cfg.name
cfg['name']
cfg.model.lr
```
이 가능하다.
이 설정은 실행 시에 덮어쓰기가 가능하다.(물론 기본 yaml파일 안의 내용은 변경되지 않는다)
예시:  `python main.py model.lr=0.01`
 version_base는 버전 호환성 관련 옵션인데 그냥 None으로 설정하면 크게 문제없다.
<br>

```bash
python main.py
```

<center><img src='{{"/assets/images/post-hydra/output.jpeg" | relative_url}}' width="80%"></center>
<br>

## 사용 예시 2) 다중 설정
```
hydra_test/
 ├─ main.py
 └─ config/
     ├─ config.yaml
     ├─ model/
     │   ├─ resnet.yaml
     │   └─ vit.yaml
     └─ dataset/
         ├─ cifar.yaml
         └─ imagenet.yaml
```
<br>

**main.py**
```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"name: {cfg.name}, model: {cfg.model.type}, lr: {cfg.model.lr}, batch_size: {cfg.model.batch_size}")
    print(f"dataset name: {cfg.dataset.name}, dataset path: {cfg.dataset.path}")

if __name__=="__main__":
    main()
```
<br>

**config/config.yaml**
```yaml
name: "soribido"
defaults:
  - model: resnet
  - dataset: cifar
  - _self_
epochs: 50
```
<br>

**config/model/resnet.yaml**
```yaml
type: "resnet50"
lr: 0.001
batch_size: 16
```
<br>

**config/model/vit.yaml**
```yaml
type: "vit_base"
lr: 0.0001
batch_size: 32
```
<br>

**config/dataset/cifar.yaml**
```yaml
name: cifar10
path: /data/cifar10
```
<br>

**config/dataset/imagenet.yaml**
```yaml
name: imagenet
path: /data/imagenet
```

hydra의 장점은 설정파일을 여러개 쓸 수 있다는 점이다.
대신에 이렇게 사용하고 싶은 설정들은 defaults에 미리 명시를 해 주어야 한다.
`_self_`는 합성 순서에 관련된 키워드로 defaults의 맨 끝에 명시하는 것이 일반적이다.(명시하는 것 자체로 warning을 방지하는 데 도움이 된다.)
<br>

**config.yaml**
```yaml
defaults:
  - model: resnet
  - _self_

model:
  lr: 0.01
```
<br>

**model/resnet.yaml**
```yaml
type: resnet50
lr: 0.001
```

model/resnet.yaml -> config.yaml순으로 되어 결국 lr은 0.01이 된다.
만약에 `_self_`가 앞에 있으면
config.yaml -> model/resnet.yaml이 되어 lr은 0.001이 된다.
보통 기본값은 그룹에서 정의하고 특정 값을 메인에서 덮어쓰는 경우가 많아 뒤에 놓는 것이 일반적이며 사용자의 성향에 따라 `_self_`를 앞에 두고 개별 파일에서 변경해도 상관은 없다.
