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
이 개념은 Hinton 등의 [Distilling the knowledge in a neural network(2015)](https://arxiv.org/abs/1503.02531) 에 소개되었다.(NeurIPS 2014 발표)  

간단한 배경으로는 앙상블과 같은 복잡한 모델의 경우 성능은 뛰어나나 이를 일반 사용자용으로 배포하는 것은 부담이 크기 때문에 배포를 위해 
간단한(inference가 빠른) 모델에 복잡한 모델의 지식을 전수한다. 

이 분야는 최근까지도 꾸준히 연구되어 다양한 아이디어들이 제시되고 있지만 여기서는 기본적인 지식 증류의 개념과 실제 적용 컨셉을 정리하고자 한다.  

## soft label
softmax를 출력층의 활성함수로 사용하는 분류 문제를 생각해 보면, 최종적으로 각 클래스에 대한 확률(0~1)을 산출하게 된다.  
어떤 샘플의 클래스별 확률이 A=0.1, B=0.3 C=0.6인 경우를 생각해 보면
가장 높은 확률을 가지는 C 클래스가 해당 class로 결정되게 되는데, 지식 증류에서는 다른 클래스에 대한 값들도 의미가 있다고 생각한다.  
C 클래스로 예측되고, B클래스의 특징도 일부 가지면서, A클래스의 특징도 약간 가지는 것으로 해석하는 것이다.  
클래스 예측 확률이 낮은 경우도 충분히 반영하기 위해 출력값의 분포를 soft하게 만드는 컨셉을 적용한다.  

$$q_{i}=\frac{exp(z_{i}/T)}{\sum _{j}exp(z_{j}/T)}$$

수식을 보면 softmax인데 지수 부분에 T로 나누는 것을 확인할 수 있다. 이 T는 temperature의 약자로 T가 클수록 확률분포를 soft하게 만든다.  
간단한 코드를 통해 실제로 soft하게 만드는지 확인해 볼 수 있다.  

```python
import numpy as np

for i in range(1,4):
    for T in range(1,6):
        q = np.exp(i/T)/(np.exp(1/T)+np.exp(2/T)+(np.exp(3/T)))
        print('i=', i, 'T=', T, q.round(3))
```
실행해 보면 T가 커질수록 기존의 softmax값이 큰 것은 그 값을 낮추고 작은 것은 높여서 분포를 전체적으로 완화하는 것을 확인할 수 있다.  

## Distillation

[keras에서 공식적으로 제공하는 코드](https://keras.io/examples/vision/knowledge_distillation/#introduction-to-knowledge-distillation)를 통해 실제로 어떻게 구성되는지를 살펴보자.  

```python
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
```        

지식 증류는 손실함수가 존재하여 T의 지식을 S에 전수하는 과정이다.
T의 soft prediction과 S의 soft prediction의 차이를 ``distillation loss``로 표현하고 여기에 S의 손실함수가 추가적으로 더해진다.(두 개의 비율은 가중치 $\alpha$ 를 통해 조절)  
분류 문제로 가정하면 cross entropy가 S의 손실함수로 적용되고, 일반적으로 distillation loss는 KL divergence(KLD, 두 확률분포의 차이)를 사용한다.
여기에서 scale을 위해 T의 제곱을 곱해주는데, T를 이용하여 soft label로 만들면 근사적으로 T의 제곱에 반비례하기 때문에 다시 T의 제곱을 곱해주는 것이다.
이에 대한 증명은 논문에 나와있는데, softmax를 활성함수로 사용했을 때 cross entropy의 미분을 이용하여 근사적으로 T의 제곱에 반비례함을 보여주고 있다.
실제로 KLD $p(x)\textrm{log}\frac{p(x)}{q(x)}$ 를 구현하여 실험하여도 근사적으로 제곱에 T의 제곱에 반비례함을 확인할 수 있다.(로그의 밑과 관계없이)    

결론적으로 지식 증류는 모델의 어떠한 지식(knowledge)을 남길(distillation) 것인가이기 때문에 teacher모델에 비해서는 성능이 떨어질 수밖에 없고, 지식을 추출하는 과정, 지식을 전수하는 과정, 어떠한 task에 적용할 것인가 등등의 관점에서 여러 가지 연구가 이루어지고 있다.