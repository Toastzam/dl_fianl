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

## 🏗️ 프로젝트 구조

```
📁 dl_final/
├── 🧠 dl_test/                    # 메인 애플리케이션
│   ├── 🤖 backend/                # FastAPI 백엔드
│   │   ├── main.py               # API 서버
│   │   └── static/               # 정적 파일
│   ├── 🎨 frontend/              # React 프론트엔드
│   │   ├── src/
│   │   │   ├── App.js
│   │   │   ├── DogSimilarityVisualizer.js
│   │   │   └── FocusedImage.js
│   │   └── package.json
│   ├── 🔬 training/              # 머신러닝 모델
│   │   ├── train.py              # SimCLR 모델 훈련
│   │   ├── extract_features.py   # 특징 추출
│   │   ├── search_similar_dogs.py # 유사도 검색
│   │   ├── visualize_keypoints.py # 키포인트 시각화 ⭐
│   │   └── dataset.py            # 데이터셋 처리
│   └── 📦 models/                # 훈련된 모델 파일
└── 🔧 mm_pose/                   # MMPose 프레임워크
    └── mmpose/                   # 키포인트 검출 라이브러리
        ├── configs/              # 모델 설정 파일
        ├── checkpoints/          # 사전훈련 모델
        └── mmpose/               # 핵심 라이브러리
```

## 🎨 키포인트 시각화 특징

### 🌈 색상 체계
- **🔴 빨간색**: 머리 부분 (눈, 귀, 코)
- **🟡 노란색**: 앞다리 (어깨, 팔꿈치, 발목)
- **🟢 초록색**: 목과 몸통
- **🟠 주황색**: 뒷다리와 꼬리

### ✨ 투명도 설정
- **골격선**: 30% 투명도 (자연스러운 구조 표시)
- **키포인트**: 50% 투명도 (부드러운 점 표시)

### 🦴 17개 키포인트
```
0: 코끝     1: 좌안     2: 우안     3: 좌귀     4: 우귀
5: 목       6: 좌어깨   7: 우어깨   8: 좌팔꿈치  9: 우팔꿈치
10: 좌발목  11: 우발목  12: 좌엉덩이 13: 우엉덩이 14: 좌무릎
15: 우무릎  16: 꼬리끝
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd dl_final

# Python 가상환경 생성
cd dl_test
python -m venv venv_dog_kpt_final
source venv_dog_kpt_final/bin/activate  # Windows: venv_dog_kpt_final\Scripts\activate

# 필요한 패키지 설치
pip install torch torchvision
pip install mmcv-full mmpose
pip install fastapi uvicorn
pip install pillow opencv-python numpy
```

### 2. MMPose 설정

```bash
# MMPose 체크포인트 다운로드
cd ../mm_pose/mmpose/checkpoints
# AP-10K 모델 다운로드 (hrnet_w32_ap10k_256x256-18aac840_20211029.pth)
```

### 3. 데이터셋 준비

Stanford Dogs Dataset을 `dl_test/training/Images/` 폴더에 준비하세요.

### 4. 모델 훈련

```bash
cd dl_test/training

# SimCLR 모델 훈련
python train.py

# 데이터베이스 특징 추출
python extract_db_features.py

# 키포인트 시각화 테스트
python visualize_keypoints.py
```

### 5. 웹 애플리케이션 실행

```bash
# 백엔드 서버 시작
cd dl_test/backend
python main.py

# 프론트엔드 서버 시작 (새 터미널)
cd dl_test/frontend
npm install
npm start
```

## 📊 성능 지표

- **SimCLR 유사도**: 코사인 유사도 기반 시각적 유사성
- **키포인트 유사도**: L2 거리 기반 포즈 유사성
- **최종 복합 유사도**: SimCLR 70% + 키포인트 30%

## 🛠️ 기술 스택

### Backend & AI
- **Python 3.8+**
- **PyTorch**: 딥러닝 프레임워크
- **MMPose**: 키포인트 검출
- **FastAPI**: 웹 API 프레임워크
- **OpenCV**: 이미지 처리

### Frontend
- **React**: 사용자 인터페이스
- **JavaScript (ES6+)**
- **CSS3**: 스타일링

### AI/ML Models
- **SimCLR**: Self-supervised contrastive learning
- **AP-10K**: 동물 키포인트 검출 모델
- **ViT (Vision Transformer)**: 특징 추출 백본
- **HRNet**: 키포인트 검출 백본

## 🎯 핵심 특징

### 🔥 혁신적인 키포인트 시각화
- **이중 투명도 시스템**: 골격선 30% + 키포인트 50%
- **색상 구분**: 해부학적 구조별 색상 분류
- **전문적인 외관**: 의료/연구용 수준의 시각화

### 🧠 지능적인 유사도 계산
- **다중 모달 접근**: 시각적 + 구조적 유사성
- **가중치 조합**: 최적화된 70:30 비율
- **실시간 처리**: 빠른 검색 및 분석

### 🎨 사용자 경험
- **직관적인 웹 인터페이스**
- **실시간 결과 시각화**
- **상세한 분석 정보 제공**

## 📈 사용 방법

1. **이미지 업로드**: 웹 인터페이스에서 강아지 이미지 업로드
2. **자동 분석**: 시스템이 SimCLR과 키포인트 분석 수행
3. **유사도 검색**: 데이터베이스에서 유사한 강아지들을 찾음
4. **시각화 결과**: 키포인트가 오버레이된 결과 이미지 제공
5. **상세 정보**: 각 유사도 점수와 분석 결과 확인

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👨‍💻 개발팀

**AI-Human 협업 프로젝트**
- 🤖 **GitHub Copilot**: AI 개발 파트너
- 👨‍💻 **Human Developer**: 창의적 아이디어 및 최적화

## 🙏 감사의 말

- **Stanford Dogs Dataset**: 훈련 데이터 제공
- **MMPose Team**: 키포인트 검출 프레임워크
- **SimCLR Authors**: Self-supervised learning 알고리즘
- **OpenMMLab**: 컴퓨터 비전 도구 생태계

---

🐕 **Happy Dog Searching with Advanced Keypoint Analysis!** 🔍✨

*"Every dog has its unique pose - let AI find the similarities!"*
