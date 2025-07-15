#  Dog Similarity S### 🐳 Docker Compose로 전체 스택 실행

전체 시스템을 한 번에 실행하는 가장 간단한 ##  개발 환경 설정ash
# 1. 프로젝트 클론
git clone <repository-url>
cd dl_final

# 2. Docker Compose로 전체 스택 빌드 및 실행
cd dl_test/docker
docker-compose up -d

# 3. 웹 브라우저에서 접속
# Frontend: http://localhost
# Backend API: http://localhost:8001/docs (FastAPI 문서)
```

### 🚀 시스템 구성

- **Frontend**: React + Material Tailwind (nginx 서빙)
- **Backend**: FastAPI + SimCLR + AP-10K 모델
- **Cache**: Redis
- **AI 모델**: 
  - SimCLR Vision Transformer (개 유사도 검색)
  - AP-10K HRNet (키포인트 검출)

### 📝 Docker 관리 명령어

```bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs backend  # 백엔드 로그
docker-compose logs frontend # 프론트엔드 로그

# 컨테이너 재시작
docker-compose restart

# 시스템 종료
docker-compose down

# 완전 정리 (볼륨 포함)
docker-compose down -v
```imCLR + AP-10K 키포인트 기반 강아지 유사도 검색 시스템

##  프로젝트 개요

강아지 이미지를 업로드하면 **SimCLR(Self-Supervised Contrastive Learning)**과 **AP-10K 키포인트 검출**을 결합하여 데이터베이스에서 유사한 강아지를 찾아주는 AI 검색 시스템입니다.

###  주요 기능

- ** SimCLR 기반 시각적 유사도 검색**: 128차원 벡터로 강아지의 시각적 특징 학습
- ** 키포인트 검출 및 시각화**: AP-10K 모델을 사용한 17개 동물 키포인트 검출
- ** 복합 유사도 계산**: SimCLR 80% + 키포인트 20% 가중 조합
- ** 실시간 키포인트 시각화**: 투명도와 색상을 적용한 전문적인 시각화
- ** 웹 인터페이스**: React 기반 사용자 친화적 웹 애플리케이션
- ** 데이터베이스 기반 벡터 저장**: MySQL을 활용한 효율적인 이미지 벡터 관리

##  빠른 시작 (Docker - 권장)

### Docker를 사용한 원클릭 실행

`ash
# 1. 프로젝트 클론
git clone <repository-url>
cd dl_final

# 2. Docker 컨테이너 빌드 및 실행
docker build -t dog-similarity-search .
docker run -p 8001:8001 -p 3000:3000 dog-similarity-search

# 3. 웹 브라우저에서 접속
# Backend API: http://localhost:8001
# Frontend Web: http://localhost:3000
`

### Docker Compose를 사용한 전체 스택 실행

`ash
# 개발 환경
docker-compose -f docker-compose.dev.yml up

# 운영 환경
docker-compose up
`

##  개발 환경 설정

### 사전 요구사항

- **Python**: 3.10.6 이상
- **Node.js**: 16.0 이상
- **MySQL**: 8.0 이상
- **CUDA**: 11.8 (GPU 사용 시, 옵션)

### 백엔드 설정

`ash
# 1. dl_test 디렉터리로 이동
cd dl_test

# 2. Python 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows

# 3. 의존성 패키지 설치
pip install -r requirements.txt

# 4. 환경 변수 설정 (.env 파일 생성)
DB_HOST=localhost
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=dog_similarity_db

# 5. 백엔드 서버 실행
cd backend
python simple_main.py
`

### 프론트엔드 설정

`ash
# 1. 프론트엔드 디렉터리로 이동
cd dl_test/frontend

# 2. Node.js 의존성 설치
npm install

# 3. 개발 서버 실행
npm start
`

##  핵심 API 엔드포인트

###  이미지 검색 API

`ash
# 강아지 유사도 검색
POST /api/upload_and_search/
Content-Type: multipart/form-data
Body: file (이미지 파일)

# 응답 예시
{
  "success": true,
  "query_image": "/uploads/my_dog.jpg",
  "query_keypoint_image": "/output_keypoints/my_dog_keypoints.jpg",
  "results": [
    {
      "rank": 1,
      "image_url": "https://example.com/similar_dog.jpg",
      "simclr_similarity": 0.89,
      "keypoint_similarity": 0.75,
      "combined_similarity": 0.86,
      "db_info": { ... }
    }
  ],
  "search_metadata": {
    "processing_time": 3.45,
    "algorithm": "SimCLR + AP-10K Hybrid AI",
    "model_version": "simclr_vit_dog_model_finetuned_v2"
  }
}
`

###  특징 벡터 추출 API

`ash
# 이미지에서 128차원 특징 벡터 추출
POST /api/extract_features/
Content-Type: multipart/form-data
Body: file (이미지 파일)

# URL에서 특징 벡터 추출
POST /api/extract_features_from_url/
Content-Type: application/json
Body: {"image_url": "https://example.com/dog.jpg"}
`

###  시스템 상태 확인

`ash
# 헬스체크
GET /health

# 응답 예시
{
  "status": "healthy",
  "models_available": true,
  "ap10k_model_loaded": true,
  "mode": "real_model",
  "message": "실제 모델 사용 가능"
}
`

##  성능 최적화 결과

###  Docker 최적화

- **이전**: 17GB (단일 스테이지 빌드)
- **최적화 후**: 4.77GB (멀티 스테이지 빌드)
- **개선율**: 72% 크기 감소

###  데이터베이스 아키텍처 개선

- **이전**: .npy 파일 기반 벡터 저장 (550MB+ 파일)
- **현재**: MySQL JSON 컬럼 기반 벡터 저장
- **장점**: 
  - 메모리 효율성 향상
  - 데이터 일관성 보장
  - 백업 및 복구 용이성
  - 확장성 개선

###  AI 모델 최적화

- **SimCLR 모델**: 128차원 압축 벡터
- **키포인트 검출**: AP-10K 17개 포인트
- **복합 유사도**: 가중 평균 (SimCLR 80% + Keypoint 20%)
- **평균 검색 시간**: 3-5초 (실제 모델 모드)

##  기술 스택

### Backend
- **FastAPI**: 고성능 Python 웹 프레임워크
- **PyTorch**: 딥러닝 모델 추론
- **MMPose**: 키포인트 검출 라이브러리
- **Pillow**: 이미지 처리
- **PyMySQL**: MySQL 데이터베이스 연동

### Frontend
- **React**: 사용자 인터페이스 라이브러리
- **TypeScript**: 정적 타입 언어
- **Axios**: HTTP 클라이언트

### AI/ML
- **SimCLR**: Self-Supervised Contrastive Learning
- **Vision Transformer (ViT)**: 이미지 특징 추출
- **AP-10K**: Animal Pose 키포인트 검출 모델

### Infrastructure
- **Docker**: 컨테이너화
- **MySQL**: 관계형 데이터베이스
- **Nginx**: 리버스 프록시 (옵션)

##  문제 해결

### 일반적인 설치 문제

**1. PyTorch CUDA 호환성 문제**
`ash
# CPU 버전 설치 (권장)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
`

**2. MMPose 설치 오류**
`ash
# 기본 패키지 먼저 설치
pip install openmim
mim install mmcv-full==1.7.1
pip install mmpose==0.29.0
`

**3. 데이터베이스 연결 오류**
- .env 파일에 올바른 MySQL 연결 정보 확인
- MySQL 서버 실행 상태 확인
- 방화벽 설정 확인

### 모델 관련 문제

**1. 모델 파일 누락**
- models/ 폴더에 simclr_vit_dog_model_finetuned_v2.pth 파일 확인
- 파일 크기: 약 85MB

**2. 더미 모드로 실행됨**
- 모델 파일 경로 확인
- Python 환경에서 PyTorch 설치 확인
- 로그에서 모델 로드 오류 메시지 확인

##  향후 개발 계획

- [ ] **GPU 가속 지원**: CUDA를 활용한 추론 속도 향상
- [ ] **배치 처리**: 다중 이미지 동시 검색
- [ ] **캐싱 시스템**: Redis를 활용한 검색 결과 캐싱
- [ ] **실시간 학습**: 새로운 이미지로 모델 지속 학습
- [ ] **모바일 앱**: React Native 기반 모바일 앱
- [ ] **클라우드 배포**: AWS/GCP 자동 배포 파이프라인

##  라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

##  기여하기

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

##  문의

프로젝트에 대한 문의사항이 있으시면 이슈를 등록해 주세요.

---

** Happy Dog Searching! **

## 🐳 Docker 추가 정보

### 시스템 요구사항

- **Docker**: 20.10 이상
- **Docker Compose**: 2.0 이상
- **메모리**: 최소 8GB RAM (AI 모델 로딩)
- **디스크**: 최소 10GB 여유 공간

### 빌드 시간 참고

- **Frontend**: 약 30초
- **Backend**: 약 8-10분 (mmcv 컴파일 포함)
- **총 시간**: 처음 빌드 시 약 10분

### Docker 이미지 크기

- **Frontend**: 약 85MB
- **Backend**: 약 4.6GB (AI 모델 포함)
- **Redis**: 약 60MB

### 포트 사용 현황

- **80**: Frontend (nginx)
- **8001**: Backend API (FastAPI)
- **6379**: Redis (내부 통신만)

## 🔍 FAQ

**Q: Docker 빌드가 너무 오래 걸려요**
A: mmcv 컴파일이 시간이 오래 걸립니다. 처음 빌드 후에는 캐시가 되어 빠릅니다.

**Q: 모델 파일을 찾을 수 없다고 나와요**
A: `models/` 폴더가 최상위 디렉터리에 있는지 확인하세요. docker-compose.yml의 볼륨 마운트가 올바른지 확인하세요.

**Q: 메모리 부족 오류가 발생해요**
A: Docker Desktop의 메모리 할당을 8GB 이상으로 늘려주세요.

**Q: Windows에서 실행이 안 돼요**
A: Windows의 경우 Docker Desktop과 WSL2가 필요합니다.

---

⭐ **문제가 있다면 Issues에 남겨주세요!**