@echo off
echo 🐳 Dog Similarity Search - Docker 빌드 및 실행
echo ================================================

REM 1. 이미지 빌드
echo 📦 도커 이미지 빌드 중...
docker-compose build

REM 2. 컨테이너 실행
echo 🚀 컨테이너 실행 중...
docker-compose up -d

REM 3. 상태 확인
echo 📊 컨테이너 상태 확인...
docker-compose ps

REM 4. 로그 확인 안내
echo 📋 로그를 보려면 다음 명령을 실행하세요:
echo docker-compose logs -f backend

echo.
echo ✅ 실행 완료!
echo 🌐 API 접속: http://localhost:8001
echo 🏥 헬스체크: http://localhost:8001/health

pause
