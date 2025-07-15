# 🐳 Docker 사용법 가이드

## 📋 사전 준비

### 1. Docker 설치
- Windows: Docker Desktop for Windows 설치
- 다운로드: https://www.docker.com/products/docker-desktop/

### 2. Docker 설치 확인
```bash
docker --version
docker-compose --version
```

## 🚀 실행 방법

### 방법 1: 스크립트 사용 (권장)
```bash
# Windows
docker-run.bat

# Linux/Mac
chmod +x docker-run.sh
./docker-run.sh
```

### 방법 2: 직접 명령어 사용

#### 개발 모드 (코드 수정 시 자동 반영)
```bash
docker-compose -f docker-compose.dev.yml up -d
```

#### 프로덕션 모드
```bash
docker-compose up -d
```

## 📊 관리 명령어

### 컨테이너 상태 확인
```bash
docker-compose ps
```

### 로그 확인
```bash
# 전체 로그
docker-compose logs

# 실시간 로그
docker-compose logs -f

# 백엔드만
docker-compose logs -f backend
```

### 컨테이너 중지
```bash
docker-compose down
```

### 컨테이너 재시작
```bash
docker-compose restart
```

### 이미지 재빌드
```bash
docker-compose build --no-cache
```

## 🔧 트러블슈팅

### 포트 충돌 시
```bash
# 포트 사용 중인 프로세스 확인
netstat -ano | findstr :8001

# 프로세스 종료 (Windows)
taskkill /PID [PID번호] /F
```

### 볼륨 초기화
```bash
docker-compose down -v
docker system prune -a
```

### 컨테이너 내부 접속
```bash
docker-compose exec backend bash
```

## 📁 파일 구조 설명

```
dl_test/
├── Dockerfile              # 프로덕션용 도커파일
├── Dockerfile.dev          # 개발용 도커파일  
├── docker-compose.yml      # 프로덕션용 구성
├── docker-compose.dev.yml  # 개발용 구성
├── .dockerignore           # 도커 빌드 제외 파일
├── requirements.txt        # Python 의존성
├── docker-run.bat         # Windows 실행 스크립트
├── docker-run.sh          # Linux/Mac 실행 스크립트
└── .env                   # 환경 변수
```

## 🌐 접속 정보

- **API 서버**: http://localhost:8001
- **API 문서**: http://localhost:8001/docs
- **헬스체크**: http://localhost:8001/health
- **Redis**: localhost:6379 (옵션)

## ⚙️ 환경 변수 (.env 파일)

필수 환경 변수들을 .env 파일에 설정하세요:

```env
# 데이터베이스
DB_HOST=byhou.synology.me
DB_PORT=3370
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database

# 기타 설정
ENVIRONMENT=production
LOG_LEVEL=INFO
```
