---
layout: single
title:  "Custom Dataloader for Tensorflow"
categories: 
    - DL
tag:
    - [deep learning, tensorflow, keras, dataloader, memory]    
author_profile: false
sidebar:
    nav: "docs"
---
tensorflow/keras 에서 모델 학습을 진행할때 미니배치만큼만 가져와서 메모리에 올리는 코드.  
dataloader를 사용하지 않으면 일반적으로 모델+모든 데이터가 메모리에 올라간다.  
pytorch의 dataloader와 유사하다.  
keras의 Sequence 모듈을 상속받아 dataloader를 정의한다.  
GPU 메모리 부족(OOM, out of memory) 오류를 방지할 수 있다.  

```python
from tensorflow.keras.utils import Sequence
class Dataloader(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

		# batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
				# sampler의 역할(index를 batch_size만큼 sampling해줌)
        # indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        # batch_x = [self.x[i] for i in indices]
        # batch_y = [self.y[i] for i in indices]

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)


train_loader = Dataloader(X_train, y_train, 128)
valid_loader = Dataloader(X_valid, y_valid, 128)
# test_loader = Dataloader(x, y, 128)

history = model.fit(train_loader, epochs=300, 
                    validation_data=valid_loader,
                    callbacks=[cb]
                    )
```