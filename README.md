# GenicarProject

프로젝트는 **쏘카x멋사 AI 엔지니어 육성 부트캠프 해커톤 과정(2021.12.20 ~ 2022.01.07)** 동안 진행되었습니다.
<br/>

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
<br/>

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

|![가까워짐 (야간)](https://user-images.githubusercontent.com/29950822/147997799-59063bf1-887c-4ac9-a33b-8ca45fd4ac6c.gif)|
|:-:|
|사고 발생|
<br/>

## 모델 구조

<br/>

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
python main.py {dataset} RGB --arch {model archtecture} --num_segments 8 --batch-size {batch size}
```

### Test
```bash
python test_video.py --arch {model archtecture} --dataset {dataset} --weights {weight path} --frame_folder {test data path}
```
<br/>

## 실험 결과

<br/>

## 향후 연구
- 실제 자율주행에 활용되기 위한 실시간성 고려
- 더욱 다양한 클래스 (상황) 추가

<br/>

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
