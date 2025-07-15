"""
의존성 주입 및 공통 컴포넌트
"""
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 전역 변수들
ap10k_model = None
device = None
visualizer = None
search_similar_dogs = None
MODELS_AVAILABLE = False

# 모델 로드 시도
try:
    import training.visualize_keypoints as vk
    import training.search_similar_dogs as ssd
    
    setup_ap10k_model = vk.setup_ap10k_model
    detect_and_visualize_keypoints = vk.detect_and_visualize_keypoints
    calculate_keypoint_similarity = vk.calculate_keypoint_similarity
    search_similar_dogs = ssd.search_similar_dogs
    
    MODELS_AVAILABLE = True
    print("✅ 모델 모듈 임포트 성공")
except ImportError as e:
    print(f"⚠️ 모델 모듈 임포트 실패: {e}")
    print("🔄 더미 모드로 실행됩니다")
    MODELS_AVAILABLE = False

def get_models():
    """모델 인스턴스들 반환"""
    return {
        'ap10k_model': ap10k_model,
        'device': device,
        'visualizer': visualizer,
        'models_available': MODELS_AVAILABLE
    }

def get_feature_service():
    """특징 추출 서비스 가져오기"""
    from app.services.feature_extraction_service import get_feature_service
    return get_feature_service()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    global ap10k_model, device, visualizer
    
    # 서버 시작 시 모델 로드
    if MODELS_AVAILABLE:
        try:
            print("🚀 AP-10K 모델 로딩 중...")
            ap10k_model, device, visualizer = setup_ap10k_model()
            print("✅ AP-10K 모델 로드 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("🔄 더미 모드로 계속 실행됩니다")
    else:
        print("🔄 모델 모듈이 없어 더미 모드로 실행됩니다")
    
    yield
    
    # 서버 종료 시 정리
    print("🔄 서버 종료 중...")
