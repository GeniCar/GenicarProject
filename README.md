# GenicarProject

프로젝트는 **쏘카x멋사 AI 엔지니어 육성 부트캠프 해커톤 과정(2021.12.20 ~ 2022.01.07)** 동안 진행되었습니다.
<br/><br/>



## 프로젝트 배경
프로젝트 주제는 **딥러닝 기반 이미지 분석을 통한 안전 운전 보조 시스템**으로 안전 운전을 위해 운전자를 보조할 수 있도록 차량 주변의 상황을 이해하는 모델을 개발하는 것이 목표입니다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/29950822/144051063-e921c436-3203-4487-a9ca-65365dbca82a.jpeg"  width="50%" />
</p>

현재, 자율주행 기술을 ‘첨단 운전자 보조 시스템(ADAS: Advanced Driver Assistance System)’은 주로 카메라를 통해 얻는 이미지 데이터와 레이더, 라이더를 통해 얻는 신호 데이터를 통해서 이루어지고 있습니다. 특히, 이미지 데이터는 주변 사물을 인식하는 객체 인식(Object Detection) 기술과 이미지 분할(Image Segmentation)을 주로 활용하는 것으로 알려져 있습니다.

하지만, 객체 인식과 이미지 분할 기술들 또한 100% 정확도의 성능을 보이기는 힘들기 때문에 이를 보조하는 기술이 필요할 것으로 생각됩니다.

이러한 보조 역할을 하기 위해 저희는 연속적인 이미지 프레임을 입력으로 받아 프레임간의 관계를 추론하여 전체적인 상황을 이해할 수 있는 모델을 사용하였습니다. 프로젝트에 활용한 모델은 [Temporal Relational Reasoning in Videos](https://arxiv.org/pdf/1711.08496.pdf)에서 제안된 모델로 동영상을 입력받아 상황을 이해하는 모델입니다.


본 프로젝트를 통한 저희의 기여점은 다음과 같습니다.
##### `1. 기존에 존재하는 모델을 새로운 도메인, 문제를 해결하는데 활용 가능한 것을 실험을 통해 증명`
##### `2. 자율주행에 주로 활용되고 있는 기술을 보완할 수 있는 방법론을 제시`
<br/><br/>



## 모델 시연 영상

### 1. 가까워짐 

|![가까움동영상](https://user-images.githubusercontent.com/60677948/148168155-bd464592-461f-4092-baed-46300ac820b3.gif)|![가까운 시연](https://user-images.githubusercontent.com/60677948/148168378-4152794b-7712-4226-9654-fac9806816d2.gif)|
|:-:|:-:|

### 2. 멀어짐

|![멀어짐동영상](https://user-images.githubusercontent.com/60677948/148167658-14242e2a-fc0f-48fd-9da0-cd5d8445aa3e.gif)|![멀어짐시연](https://user-images.githubusercontent.com/60677948/148167680-58a2ba9d-dad7-400c-b8f4-2619348f7597.gif)|
|:-:|:-:|

### 3. 사고
|![사고동영상](https://user-images.githubusercontent.com/60677948/148164863-992e0019-21c0-4cc9-b70d-8694fc3ffa82.gif)|![사고시연](https://user-images.githubusercontent.com/60677948/148163259-b3565a22-ba73-425f-bb94-17012a87df36.gif)|
|:-:|:-:|

<br/><br/>




## 데이터셋
### V 1.0.0
2개의 클래스 분류
- 전방 차량과의 거리 가까워짐
- 전방 차량과의 거리 멀어짐

|![가까워짐](https://user-images.githubusercontent.com/29950822/147995965-0506401f-c9be-43ff-9726-a7c11a57d27f.gif)|![멀어짐](https://user-images.githubusercontent.com/29950822/147997077-d665e47f-7838-4239-845a-967d1d62074c.gif)|
|:-:|:-:|
|가까워짐|멀어짐|
<br/>

### V 1.0.1
2개의 클래스 분류 (+ 주/야간 환경 변화 추가)
- 전방 차량과의 거리 가까워짐
- 전방 차량과의 거리 멀어짐
- (new) 야간 전방 차량과의 거리 가까워짐
- (new) 야간 전방 차량과의 거리 멀어짐

|![가까워짐 (야간)](https://user-images.githubusercontent.com/29950822/147997532-c982de1a-4ced-4aa1-a084-c0265c8f3a50.gif)|![멀어짐 (야간)](https://user-images.githubusercontent.com/29950822/147997623-96eefa0c-b3fd-4e08-84ef-6d94bb831c09.gif)|
|:-:|:-:|
|가까워짐 (야간)|멀어짐 (야간)|
<br/>

### V 2.0.0
3개의 클래스 분류
- 전방 차량과의 거리 가까워짐
- 전방 차량과의 거리 멀어짐
- (new) 전방 사고 발생

|![사고 발생](https://user-images.githubusercontent.com/60677948/148338522-3d6dc80c-cdff-45f9-9a32-846a9dcb9daf.gif)|
|:-:|
|사고 발생|

<br/><br/>


## 모델 구조

<br/><br/>



## 코드
본 프로젝트의 코드는 [zhoubolei/TRN-pytorch](https://github.com/zhoubolei/TRN-pytorch)을 기반으로 하였으며 프로젝트의 목적에 맞추어 수정하였습니다.

### Requierments
```
python=3.7.12
pytorch=1.10.0+cu111
pillow=8.0.0
pyyaml=5.4.1
opencv=3.4.2
moviepy=1.0.1
```

### Train
```bash
python main.py --arch {model archtecture} --num_segments 8 --consensus_type TRNmultiscale --batch-size {batch size}
```

### Test
```bash
python test_video.py --arch {model archtecture} --weights {weight path} --frame_folder {test data path}
```
<br/><br/>



## 실험 방법 및 결과

### 모델 학습 파라미터
```
num_segment = 8
learning_rate = 0.001 
batch_size = 8
drop_out = 0.8
```

### 1차 실험 (이진분류)
  - 데이터셋
    - v 1.0.0의 주간 가까워짐/멀어짐 
    - 총 개수 : 68
  
  - 사용 모델
    - BNInception, ResNet18, Resnet50, Resnet101

  - 실험 반영점
    - 기존 논문에서 학습한 pre-train 모델인 BNInception과 그 외의 pre-train CNN 모델 비교
    - IOT에 반영하기 위한 가벼운 모델 탐색
  
  - 결과 및 개선 방향
    - 4개의 모델에서 accuracy: 90% 이상 Loss : 0 에 수렴
    - 정확도와 Loss의 결과가 좋게 나왔지만 학습 수렴 곡선이 일정하지  

|BNInception|ResNet18|
|:-:|:-:|
|<img src = "https://github.com/GeniCar/GenicarProject/blob/main/plots/first_binary_cls_BNInception.png" width="500" height="280">|<img src = "https://github.com/GeniCar/GenicarProject/blob/main/plots/first_binary_cls_resnet18.png" width="500" height="280">|

|ResNet50|ResNet101|
|:-:|:-:|
|<img src = "https://github.com/GeniCar/GenicarProject/blob/main/plots/first_binary_cls_resnet50.png" width="500" height="280">|<img src = "https://github.com/GeniCar/GenicarProject/blob/main/plots/first_binary_cls_resnet101.png" width="500" height="280">|

### 2차 실험 (사고 클래스 추가)
  - 데이터셋
    - v 2.0.0의 주간 및 야간 가까워짐/멀어짐, 사고 (사고 데이터의 경우 내 차가 사고 난 영상 + 다른차가 사고난 영상)
    - 총 개수 : 196
    - 노이즈 데이터(인코딩 및 영상불량) 제거하지 않음
  
  - 사용 모델
    - BNInception, Resnet101

  - 실험 반영점
    - 사고 레이블 데이터 추가
    - 사고 레이블을 추가하면서 Resnet18,50은 학습 결과가 좋지 않아 제외 
  
  - 결과 및 개선 방향
    - accuracy: 80% 이하 Loss : 0.5 이상  
    - 이진분류보다 accuracy -10%, Loss +0.5
    - Data가 부족하다고 생각, 각 레이블당 최소 100개 이상 필요  

|BNInception|ResNet101|
|:-:|:-:|
|<img src = "https://github.com/GeniCar/GenicarProject/blob/main/plots/second_BNInception__noise_data196.png" width="500" height="280">|<img src = "https://github.com/GeniCar/GenicarProject/blob/main/plots/second_resnet101__noise_data196.png" width="500" height="280">|

### 3차 실험 (클래스당 130개의 데이터)
  - 데이터셋
    - 2차 실험과 동일
    - 총 개수 : 390
  
  - 사용 모델
    - BNInception, Resnet101

  - 실험 반영점
    - 2차 실험에 데이터 수 2배 증가 
    - 모델과 학습 파라미터 경우 2차 실험과 동일
  
  - 결과 및 개선 방향
    - 2차 실험보다 Resnet101 경우 accuracy가 10% 이상 감소  
    - 두 모델의 성능이 2차 실험에 비해 성능 하락
    - 데이터 확인 한 결과 데이터에 불량 데이터가 다수 발견 -> 불량 데이터 제거 
    - 사고 데이터의 경우 (내 차가 사고 난 영상 + 다른차가 사고난 영상) 존재 -> 일단 다른차가 사고난 영상만 훈련에 사용 그 후 내 차가 사고 난 영상만 사용     

|BNInception|ResNet101|
|:-:|:-:|
|<img src = "https://github.com/GeniCar/GenicarProject/blob/main/plots/third_model_BNInception_data384.png" width="500" height="280">|<img src = "https://github.com/GeniCar/GenicarProject/blob/main/plots/third_resnet101_data384.png" width="500" height="280">|

### 4차 실험 (노이즈 데이터 제외 및 사고 데이터 재정의)
  - 데이터셋
    - 3차 실험에서 노이즈 데이터 제외 및 사고 데이터 재정의
    - 다른차가 사고난 영상만 훈련에 사용 데이터 개수: 180
    - 내 차가 사고 난 영상: 226
  
  - 사용 모델
    - BNInception, Resnet101

  - 실험 반영점
    - 노이즈 데이터 제외 및 사고 데이터 재정의 
    - 모델과 학습 파라미터 경우 2차 실험과 동일

  - 결과 및 개선 방향 논의점
    - 아래의 결과 그래프는 내 차가 사고 난 영상 결과/ 다른차가 사고난 영상 결과는 3차 실험과 비슷함 
    - 두 모델 다 accuracy: 90% 내외 Loss: 0.4 내외
    - 2,3차 실험 대비 성능개선
    - 최종 실험 결과물

|BNInception|ResNet101|
|:-:|:-:|
|<img src = "https://user-images.githubusercontent.com/60677948/148338690-29c5b91f-b2b7-4c83-8f46-40943781a320.png" width="500" height="280">|<img src = "https://user-images.githubusercontent.com/60677948/148338847-fe1eda0e-2c21-4fb0-aa17-39568e1d1fe5.png" width="500" height="280">|


### 개선 방향

  - 4차 실험에서 노이즈가 없는 데이터를 추가하면 모델 성능 향상 기대
  - 컴퓨팅 자원이 충분하다면 Batch_size 8 이상으로 늘려서 모델 수렴에 도움
  - 노이즈 데이터를 자동으로 줄일 수 있는 방법 제시

<br/><br/>



## 향후 연구
- 실제 서비스로 활용되기 위한 실시간성 고려
  - [ ] 추론 속도 향상
  - [ ] 가벼운 모델

- 더욱 다양한 상황 판단을 위한 새로운 클래스 추가
  - [ ] 후방 및 측면 영상 활용 실험
<br/><br/>



## 참고
B. Zhou, A. Andonian, and A. Torralba. Temporal Relational Reasoning in Videos. European Conference on Computer Vision (ECCV), 2018.
```
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}

```
<br/><br/>

## 참여자
