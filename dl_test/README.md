# 🐕 Dog Similarity Search with Keypoint Analysis

강아지 이미지 유사도 검색 시스템 - SimCLR과 키포인트 분석을 결합한 고급 검색 엔진

## 🎯 프로젝트 개요

이 프로젝트는 **SimCLR(Self-Supervised Contrastive Learning)**과 **AP-10K 키포인트 검출**을 결합하여 강아지 이미지의 유사도를 정밀하게 분석하는 시스템입니다.

### ✨ 주요 기능

- **🔍 SimCLR 기반 시각적 유사도 검색**: Self-supervised 학습으로 강아지의 시각적 특징 학습
- **🦴 키포인트 검출 및 시각화**: AP-10K 모델을 사용한 17개 동물 키포인트 검출
- **📊 복합 유사도 계산**: SimCLR 70% + 키포인트 30% 가중 조합
- **🎨 고급 시각화**: 투명도와 색상을 적용한 전문적인 키포인트 시각화
- **🌐 웹 인터페이스**: React 기반 사용자 친화적 웹 애플리케이션

## 🏗️ 시스템 아키텍처

```
📁 프로젝트 구조
├── 🤖 backend/                 # FastAPI 백엔드
│   ├── main.py                 # API 서버
│   └── static/                 # 정적 파일
├── 🎨 frontend/                # React 프론트엔드
│   ├── src/
│   │   ├── App.js
│   │   ├── DogSimilarityVisualizer.js
│   │   └── FocusedImage.js
│   └── package.json
├── 🧠 training/                # 머신러닝 모델
│   ├── train.py                # SimCLR 모델 훈련
│   ├── extract_features.py     # 특징 추출
│   ├── search_similar_dogs.py  # 유사도 검색
│   ├── visualize_keypoints.py  # 키포인트 시각화
│   └── dataset.py              # 데이터셋 처리
└── 📦 models/                  # 훈련된 모델 파일
```

## 🚀 시작하기

### 1. 환경 설정

#### 시스템 요구사항
- **Python**: 3.10+ (권장: 3.10.6)
- **운영체제**: Windows 10/11, macOS, Linux
- **메모리**: 8GB RAM 이상 권장

#### 패키지 설치 (CPU 환경)

**방법 1: requirements.txt 사용 (권장)**
```bash
# Python 가상환경 생성
python -m venv venv_dog_kpt_final
source venv_dog_kpt_final/bin/activate  # Windows: venv_dog_kpt_final\Scripts\activate

# 모든 패키지 한번에 설치
pip install -r requirements.txt
```

**방법 2: 수동 설치**

```bash
# Python 가상환경 생성
python -m venv venv_dog_kpt_final
source venv_dog_kpt_final/bin/activate  # Windows: venv_dog_kpt_final\Scripts\activate

# 핵심 딥러닝 패키지
pip install torch==2.7.1+cpu torchvision==0.22.1+cpu torchaudio==2.7.1+cpu
pip install numpy==1.24.3

# 컴퓨터 비전 및 키포인트 검출
pip install mmcv==2.1.0
pip install mmpose==1.3.2
pip install mmdet==3.3.0
pip install mmengine==0.10.7

# 웹 프레임워크
pip install fastapi==0.115.14
pip install uvicorn==0.34.3
pip install python-multipart==0.0.20

# 이미지 처리
pip install pillow==11.2.1
pip install opencv-python==4.11.0.86

# 데이터 처리 및 유틸리티
pip install scikit-learn==1.7.0
pip install matplotlib==3.10.3
pip install tqdm==4.67.1
pip install transformers==4.53.0
pip install timm==1.0.16

# COCO 데이터셋 도구
pip install xtcocotools==1.14.3
pip install pycocotools==2.0.10

# 추가 의존성 패키지들
pip install addict==2.4.0
pip install annotated-types==0.7.0
pip install anyio==4.9.0
pip install certifi==2025.6.15
pip install click==8.2.1
pip install contourpy==1.3.2
pip install cycler==0.12.1
pip install filelock==3.13.1
pip install fonttools==4.58.4
pip install fsspec==2024.6.1
pip install huggingface-hub==0.33.1
pip install jinja2==3.1.4
pip install joblib==1.5.1
pip install json-tricks==3.17.3
pip install kiwisolver==1.4.8
pip install markdown-it-py==3.0.0
pip install markupsafe==2.1.5
pip install networkx==3.3
pip install packaging==25.0
pip install platformdirs==4.3.8
pip install pydantic==2.11.7
pip install pydantic-core==2.33.2
pip install pygments==2.19.2
pip install pyparsing==3.2.3
pip install python-dateutil==2.9.0.post0
pip install pyyaml==6.0.2
pip install regex==2024.11.6
pip install requests==2.32.4
pip install rich==14.0.0
pip install safetensors==0.5.3
pip install scipy==1.15.3
pip install shapely==2.1.1
pip install six==1.17.0
pip install starlette==0.46.2
pip install sympy==1.13.3
pip install termcolor==3.1.0
pip install terminaltables==3.1.10
pip install threadpoolctl==3.6.0
pip install tokenizers==0.21.2
pip install typing-extensions==4.12.2
pip install urllib3==2.5.0
pip install yapf==0.43.0
```

#### 주요 의존성 호환성
- **numpy==1.24.3**: xtcocotools와의 바이너리 호환성 확보
- **mmcv==2.1.0**: MMPose 1.3.2와 호환 (<2.2.0 요구사항)
- **torch==2.7.1+cpu**: CPU 전용 PyTorch 버전
- **Python 3.10+**: 모든 패키지와의 호환성 보장

### 2. 데이터셋 준비

Stanford Dogs Dataset을 `training/Images/` 폴더에 준비하세요.

### 3. 모델 훈련

```bash
# SimCLR 모델 훈련
python training/train.py

# 데이터베이스 특징 추출
python training/extract_db_features.py
```

### 4. 웹 애플리케이션 실행

```bash
# 백엔드 서버 시작
cd backend
python main.py

# 프론트엔드 서버 시작 (새 터미널)
cd frontend
npm install
npm start
```

## 🎨 키포인트 시각화

### 색상 체계
- 🔴 **빨간색**: 머리 부분 (눈, 귀, 코)
- 🟡 **노란색**: 앞다리 (어깨, 팔꿈치, 발목)
- 🟢 **초록색**: 목과 몸통
- 🟠 **주황색**: 뒷다리와 꼬리

### 투명도 설정
- **골격선**: 30% 투명도 (자연스러운 구조 표시)
- **키포인트**: 50% 투명도 (부드러운 점 표시)

## 📊 성능 지표

- **SimCLR 유사도**: 코사인 유사도 기반 시각적 유사성
- **키포인트 유사도**: L2 거리 기반 포즈 유사성
- **최종 복합 유사도**: SimCLR 70% + 키포인트 30%

## 🛠️ 기술 스택

### Backend
- **Python 3.10+**: 메인 개발 언어
- **PyTorch 2.7.1+cpu**: 딥러닝 프레임워크 (CPU 최적화)
- **TorchVision 0.22.1+cpu**: 컴퓨터 비전 라이브러리
- **TorchAudio 2.7.1+cpu**: 오디오 처리 라이브러리
- **MMPose 1.3.2**: 키포인트 검출 라이브러리
- **MMCV 2.1.0**: 컴퓨터 비전 기반 라이브러리
- **MMEngine 0.10.7**: MMPose 엔진
- **MMDet 3.3.0**: 객체 검출 라이브러리
- **FastAPI 0.115.14**: 고성능 웹 API 프레임워크
- **Uvicorn 0.34.3**: ASGI 서버
- **OpenCV 4.11.0.86**: 이미지 처리
- **NumPy 1.24.3**: 수치 연산 (호환성 최적화)
- **Pillow 11.2.1**: 이미지 처리

### Frontend
- **React 18+**: 사용자 인터페이스
- **JavaScript (ES6+)**: 모던 자바스크립트
- **CSS3**: 반응형 스타일링

### AI/ML
- **SimCLR**: Self-supervised contrastive learning
- **AP-10K**: 동물 키포인트 검출 모델 (17개 포인트)
- **ViT (Vision Transformer)**: 특징 추출 백본
- **Transformers 4.53.0**: 사전 훈련된 모델 활용
- **Scikit-learn 1.7.0**: 머신러닝 유틸리티
- **TIMM 1.0.16**: 사전 훈련된 모델 라이브러리
- **HuggingFace Hub 0.33.1**: 모델 허브 접근

### 데이터 처리
- **COCO Tools**: xtcocotools 1.14.3, pycocotools 2.0.10
- **Matplotlib 3.10.3**: 데이터 시각화
- **SciPy 1.15.3**: 과학적 계산
- **NetworkX 3.3**: 그래프 분석
- **Shapely 2.1.1**: 기하학적 처리

### 유틸리티 및 기타
- **Pydantic 2.11.7**: 데이터 검증
- **Rich 14.0.0**: 터미널 출력 개선
- **TQDM 4.67.1**: 진행률 표시
- **PyYAML 6.0.2**: YAML 파일 처리
- **Requests 2.32.4**: HTTP 클라이언트
- **Addict 2.4.0**: 딕셔너리 확장
- **CSS3**: 반응형 스타일링

### AI/ML
- **SimCLR**: Self-supervised contrastive learning
- **AP-10K**: 동물 키포인트 검출 모델 (17개 포인트)
- **ViT (Vision Transformer)**: 특징 추출 백본
- **Transformers 4.53.0**: 사전 훈련된 모델 활용
- **Scikit-learn 1.7.0**: 머신러닝 유틸리티

### 데이터 처리
- **COCO Tools**: xtcocotools 1.14.3, pycocotools 2.0.10
- **Pillow 11.2.1**: 이미지 처리
- **Matplotlib 3.10.3**: 데이터 시각화

## 📈 사용 방법

1. **이미지 업로드**: 웹 인터페이스에서 강아지 이미지 업로드
2. **유사도 검색**: 시스템이 자동으로 유사한 강아지들을 찾음
3. **키포인트 분석**: 각 강아지의 키포인트가 시각화됨
4. **결과 확인**: 복합 유사도 점수와 함께 결과 제공

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🛠️ 문제 해결

### 일반적인 설치 문제

#### 1. NumPy/xtcocotools 호환성 오류
```bash
# 오류: ValueError: numpy.dtype size changed, may indicate binary incompatibility
pip uninstall -y xtcocotools pycocotools
pip install --force-reinstall --no-cache-dir "numpy==1.24.3"
pip install --no-cache-dir xtcocotools pycocotools
```

#### 2. MMCV 버전 호환성 문제
```bash
# 오류: MMCV==2.2.0 is used but incompatible
pip install --force-reinstall --no-cache-dir "mmcv==2.1.0"
```

#### 3. 더미 모드 확인
시스템에서 실제 모델 로딩에 실패하면 자동으로 더미 모드로 전환됩니다.
`http://localhost:8001/health` 엔드포인트에서 시스템 상태를 확인할 수 있습니다.

#### 4. 메모리 부족 문제
- CPU 환경에서는 배치 크기를 줄이거나 이미지 해상도를 낮춰보세요
- 가상 메모리 설정을 늘리거나 브라우저 탭을 줄이세요

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👨‍💻 개발자

**GitHub Copilot & Human Developer**
- 💡 AI-Human 협업으로 완성된 혁신적인 프로젝트

## 🙏 감사의 말

- **Stanford Dogs Dataset**: 훈련 데이터 제공
- **MMPose Team**: 키포인트 검출 프레임워크
- **SimCLR Authors**: Self-supervised learning 알고리즘

---

🐕 **Happy Dog Searching!** 🔍✨
