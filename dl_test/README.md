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

```bash
# Python 가상환경 생성
python -m venv venv_dog_kpt_final
source venv_dog_kpt_final/bin/activate  # Windows: venv_dog_kpt_final\Scripts\activate

# 필요한 패키지 설치
pip install torch torchvision
pip install mmcv-full mmpose
pip install fastapi uvicorn
pip install pillow opencv-python numpy
```

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
- **Python 3.8+**
- **PyTorch**: 딥러닝 프레임워크
- **MMPose**: 키포인트 검출
- **FastAPI**: 웹 API 프레임워크
- **OpenCV**: 이미지 처리

### Frontend
- **React**: 사용자 인터페이스
- **JavaScript (ES6+)**
- **CSS3**: 스타일링

### AI/ML
- **SimCLR**: Self-supervised contrastive learning
- **AP-10K**: 동물 키포인트 검출 모델
- **ViT (Vision Transformer)**: 특징 추출 백본

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
