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
python main.py --arch {model archtecture} --num_segments 8 --consensus_type TRNmultiscale --batch-size {batch size}
```

### Test
```bash
python test_video.py --arch {model archtecture} --weights {weight path} --frame_folder {test data path}
```
<br/>

## 실험 결과

### 1. 이진분류(가까워짐 멀어짐)


|BNInception![BNInception](https://github.com/GeniCar/GenicarProject/blob/main/plots/first_binary_cls_BNInception.png)|ResNet18![ResNet18](https://github.com/GeniCar/GenicarProject/blob/main/plots/first_binary_cls_resnet18.png)|
|:-:|:-:|

|ResNet50![ResNet50](https://github.com/GeniCar/GenicarProject/blob/main/plots/first_binary_cls_resnet50.png)|ResNet101![ResNet101](https://github.com/GeniCar/GenicarProject/blob/main/plots/first_binary_cls_resnet101.png)|
|:-:|:-:|

처음 이진분류를 적용한 결과 loss 값이 거의 제로에 가깝게 나타났고 정확도는 resnet18을 제외하고는 100%에 가깝게 나타났다는 것을 알 수 있었습니다. 
결과적인 성능이 좋게 나와서 추가적인 분류를 해보고자 사고 데이터를 추가하여 다중분류를 시도해 보았습니다.

<br/>

### 2. 첫번째 다중분류(가까워짐, 멀어짐, 사고발생여부)


|BNInception![BNInception](https://github.com/GeniCar/GenicarProject/blob/main/plots/second_BNInception__noise_data196.png)|ResNet101![ResNet101](https://github.com/GeniCar/GenicarProject/blob/main/plots/second_resnet101__noise_data196.png)|
|:-:|:-:|

첫번째 다중분류 결과는 생각과 다르게 정확도와 loss측면에서 만족스러운 성과를 얻지 못하였고, 이 문제에 대한 원인으로 확보한 데이터의 부족, 혹은 확보한 데이터 속 노이즈 존재에 따른 결과라고 생각하여 우선적으로 부족한 데이터를 더 추가하여 두번째 다중분류를 시도하였습니다.

<br/>

### 3. 두번째 다중분류(가까워짐, 멀어짐, 사고발생여부) _ 데이터 추가


|BNInception(data-384)![BNInception](https://github.com/GeniCar/GenicarProject/blob/main/plots/third_model_BNInception_data384.png)|BNInception(data-466)![BNInception](https://github.com/GeniCar/GenicarProject/blob/main/plots/third_model_BNInception_data466.png)|
|:-:|:-:|

|ResNet101(data-384)![ResNet101](https://github.com/GeniCar/GenicarProject/blob/main/plots/third_resnet101_data384.png)|ResNet101(data-466)![ResNet101](https://github.com/GeniCar/GenicarProject/blob/main/plots/third_resnet101_data466.png)|
|:-:|:-:|

두번째 다중분류를 수행한 결과 정확도와 로스값의 변화는 거의 없거나, 오히려 그 값이 소폭 줄어들었음을 알 수 있었습니다. 이를 해결하기 위해, 앞서 말했던 노이즈 문제를 생각했고, 이 점을 해결한 뒤 다시 모델을 작동시켜 그 결과를 확인해 보았습니다. 

<br/>

### 4. 세번째 다중분류(가까워짐, 멀어짐, 사고발생여부) _ 노이즈 제거

|BNInception(remove_noise)![BNInception](https://github.com/GeniCar/GenicarProject/blob/main/plots/fourth_BNInception_data180.png)|ResNet50(remove_noise)![ResNet50](https://github.com/GeniCar/GenicarProject/blob/main/plots/fourth_resnet50_data196_removenoise.png)|ResNet101(remove_noise)![ResNet101](https://github.com/GeniCar/GenicarProject/blob/main/plots/fourth_resnet101_remove_noise_data180.png)|
|:-:|:-:|:-:|

세번째 다중분류를 수행한 결과 정확도는 80%이상의 값을 BNInception와 ResNet101에서 보였음 확인할 수 있었고, loss값은 0.5 이하의 값을 가진다는 것을 확인할 수 있었습니다. 


작은 배치 사이즈를 사용하였고, 직접 수집한 데이터로 인해 발생한 에러 상황을 해결하기 위해 노이즈 제거를 하여 충분히 보완하여 나온 값으로, 결과적으로 모델의 성능이 크게 뒤떨어지지 않았다는 것을 볼 수 있었습니다. 세번째 다중분류의 결과로 resnet50(정확도와 loss값이 만족스럽지 않은)이 아닌 모델을 선택하여, 더 많은 데이터 수집과 데이터의 노이즈가 확실히 제거된다면 모델의 성능이 더 향상될 수 있다고 판단 할 수 있었습니다. 





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
