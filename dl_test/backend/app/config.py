"""
애플리케이션 설정 및 상수 정의
"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class Settings:
    """애플리케이션 설정"""
    
    # 기본 설정
    APP_TITLE = "Dog Similarity Search API"
    APP_DESCRIPTION = "SimCLR + AP-10K 키포인트 기반 강아지 유사도 검색"
    VERSION = "1.0.0"
    DEBUG = True
    
    # 서버 설정
    HOST = "0.0.0.0"
    PORT = 8001
    
    # CORS 설정
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://192.168.0.46:3000",
        "http://192.168.0.*:3000",
        "http://localhost:5173",
        "*"  # 개발용, 운영시에는 구체적 IP 설정
    ]
    
    # 경로 설정
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
    OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output_keypoints")
    STATIC_FOLDER = os.path.join(BACKEND_DIR, "static")
    
    # 모델 설정
    SIMCLR_WEIGHT = 0.9
    KEYPOINT_WEIGHT = 0.1
    SIMCLR_OUT_DIM = 128
    SIMCLR_IMAGE_SIZE = 224
    
    # SimCLR 모델 경로
    @property
    def SIMCLR_MODEL_PATH(self):
        base_path = os.path.join(self.BACKEND_DIR, '..', 'models', 'simclr_vit_dog_model_finetuned_v2.pth')
        if os.path.exists(base_path):
            return base_path
        # fallback: 프로젝트 루트의 models/ 경로 시도
        alt_path = os.path.join(self.BACKEND_DIR, '..', '..', 'models', 'simclr_vit_dog_model_finetuned_v2.pth')
        if os.path.exists(alt_path):
            return alt_path
        return base_path
    
    @property
    def SIMCLR_MODEL_VERSION(self):
        """SimCLR 모델 파일명(버전) 자동 추출"""
        path = self.SIMCLR_MODEL_PATH
        fname = os.path.basename(path)
        if fname.endswith('.pth'):
            fname = fname[:-4]
        return fname
    
    # 강아지 판별 임계값
    SIMCLR_MIN_SIM = 0.28
    MIN_KEYPOINTS = 10
    MIN_AVG_SCORE = 0.25
    
    def __init__(self):
        """디렉토리 생성"""
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(self.STATIC_FOLDER, exist_ok=True)

# 설정 인스턴스
settings = Settings()
