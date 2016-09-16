# Introduction
* **85%** : 2016년도 인터넷 데이터에서 픽셀 데이터가 차지하는 비율(출처: CISCO). 픽셀 데이터는 분석하기 매우 힘들다. 마치 우주의 `dark matter`라고 비유된다.
* **150시간** : Youtube에 1시간마다 업로드되는 동영상의 길이. 인력으로 `annotation`이나 `labeling`이 불가능한 수준에 이르렀다. 
* **결론** : `vision technology` 가 이렇게나 중요하다.
* **cs231n?** : `CNN`(Convolutional Neural Network)라는 도구로 `image classification`이라는 문제를 해결하는 것에 집중할 것이다.
# Historical Context
* 수강생의 **primary goal**은 **deep learning**일 것이다.
* 하지만 **problem domain**을 이해하지 않고는 **next model**의 **inventor**가 될 수 없다.
* 그리고 deep learning, 특히 **cnn**은 **vision domain**의 문제를 해결하기 위해 만들어졌다.
* **general problem domain**과 **tool** 사이는 절대 **de-coupling**되지 않는다.
* 그러므로 **vision domain**의 **historical context**를 이해할 필요가 있다!
## Hubel & Wiesel (1959)
* 고양이 뇌에(`primary visual cortex`로 추정되는) 전극을 꽂고 다양한 이미지를 보여주며 전기 신호를 측정했다.
* 물고기 이미지를 보여준다고 해서 전극에 스파크가 튀는 것이 아니었다.
* 오히려 이미지 판을 갈아끼우는 과정에서(옛날이라 컴퓨터가 아니라 수동...) 판의 모서리 움직임이 발생할때 스파크가 관측되었다.

**결론** : `primary visual cortex`에서 처음부터 **전체 이미지**를 인식하는 것이 아니다. **visual recognition**의 시작은, **simple structure**를 인식하는것에서 시작된다. 그리고 그것이 **deep neural net**의 modeling이다(!!)  

## Larry Roberts (1963)
* 사람은 각도/조도에 상관없이 **Block**을 잘 인식한다.
* 전체 이미지를 보는 것이 아니라, 이미지의 일부분인 **edge**를 인식하는 거라 추측할 수 있다.
* 컴퓨터로 **edge**를 추출하는 주제로 박사학위를 받았다.
* (그리고 DARPA에 가서 인터넷을 개발했다......?)

## MIT AI Group Summer Camp (1966)
* **Computer Vision**의 시작!

## David Marr (1970s)
* *Vision* 이라는 제목의 책을 출간.
* **human visual recognition**의 **hierarchical model**을 제시
* 이는 후에 **deep learning**의 **hierarchical model**에 중요한 **insight**로 역할 

## Viola & Jones (2001)
* **face detection**
* **black & white feature**를 사용해서 **real-time**으로 동작했다.
* 당시 스마트 디지털 카메라에 들어간 기술이다. 최초로 상업적으로 이용된 것.
* 이전의 연구들은 **modeling 3D object**를 목표로 했다.
* 이 연구를 기점으로 **recognition**으로 연구의 중심이 이동했다.
* 그럼으로 다시 **Artifical Intelligence**의 영역으로 진입한 것이다.

## David Lowe (1999)
* **object recognition**을 위한 새로운 **feature** 제시 : `SIFT`
* 사물을 인식할때 전체를 보고 인식하는 것이 아니라, 일부분(feature)를 가지고 인식할 수 있다.
* 예전에는 엔지니어들이 직접 **hard-code**로 feature를 뽑아내던 걸, 요새는 **deep neural network**가 알아서 해결해준다.

## IMAGENET
* 모델의 성능을 비교할 수 있는 **bench-mark**가 필요했다. 일종의 **global standard**. 
* 처음 시작은 **PASCAL Visual Object Challenge**였다. 하지만 20개의 카테고리는 실세계와 차이가 너무 컸다.
* 22K 카테고리 & 14M 이미지
* **amazon mechanical turk** 플랫폼 등에서 크라우드 소싱으로 이미지를 모았다.
* 2010년부터 매해 IMAGENET의 데이터를 가지고 **image recognition** 모델의 성능을 겨루는 **IMAGENET Large Scale Visual Recognition Challenge**가 열리고 있다.
* 2010년부터 2014년까지 매해 성능은 개선되었다.
* 2011년에는 **SIFT + SVM** 모델. **hierarchical model** 이지만 (SIFT이므로) **feature** 추출에 **learning flavor**가 없었다.
* 그 중에서도 **2012년**의 진보가 가장 컸고, 패러다임 자체를 바꿔버렸다.
* 2015년에도 여전히 `cnn` 아키텍쳐의 모델이 우승했다(151개의 layer).
* **Beginning of Deep Learning Evolution!!!**

# Overview
* **main focus** = `computer vision`, 중에서도 `visual recognition`, 중에서도 `image classification`.
* 인접 분야인 `object deteection`, `image captioning`에 대해서도 조금 다룬다.
* `cnn`이 **object recognition**에 아주 강력한 도구이다.
> **cnn**은 하룻밤에 만들어진 것이 아니다.

## LeCun(1998) vs SuperVision(2012)
* 2012년 **IMAGENET Challenge**의 우승 모델인 **SuperVision**은, 1998년 미국 우체국의 우편 번호 해독을 위해 개발된 **LeCun**의 그것과 아키텍쳐가 유사하다.
* 그렇지만 **2가지 큰 차이점**이 존재했다.
	* **GPU** : NVIDIA의 큰 기여가 있었다. 98년도에 비해서, 12년도의 컴퓨팅 파워가 대략 **x1000** 수준이 되었다. 
	* **Data** : LeCun의 모델이 기반한 **MNIST** 데이터 셋은 약 **10^7**이었지만, IMAGENET은 약 **10^14** 수준으로 비교되지 않을 정도로 컸다. 
* 그렇다면 이제 **computer vision**의 모든 문제가 해결된 것인가?
* **그렇지 않다** : 새로운 **quest**에 마주해있다. **beyond visual recognition...**
