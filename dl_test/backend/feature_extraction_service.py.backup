import os
# TensorFlow 메시지 숨기기
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import numpy as np
from PIL import Image
import io
import sys
from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FeatureExtractionService:
    """이미지에서 특징 벡터를 추출하는 서비스"""
    
    def __init__(self):
        self.feature_extractor = None
        self.transform = None
        self.device = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self):
        """모델 로드"""
        try:
            from training.extract_features import setup_feature_extractor
            self.feature_extractor, self.transform, self.device = setup_feature_extractor()
            self.is_loaded = True
            print("✅ 특징 추출 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            print("🔄 더미 벡터 모드로 실행됩니다")
            self.is_loaded = False
    
    def extract_features_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """이미지 바이트에서 특징 벡터 추출"""
        try:
            # PIL Image로 변환
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self._extract_features_from_image(image)
        except Exception as e:
            raise ValueError(f"이미지 처리 실패: {e}")
    
    def extract_features_from_path(self, image_path: str) -> np.ndarray:
        """이미지 파일 경로에서 특징 벡터 추출"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self._extract_features_from_image(image)
        except Exception as e:
            raise ValueError(f"이미지 파일 읽기 실패: {e}")
    
    def _extract_features_from_image(self, image: Image.Image) -> np.ndarray:
        """PIL Image에서 특징 벡터 추출"""
        if not self.is_loaded:
            # 더미 벡터 반환 (128차원)
            return np.random.rand(128).astype(np.float32)
        
        try:
            # 이미지 전처리
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 특징 추출
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
            
            # NumPy 배열로 변환
            return features.cpu().numpy().flatten().astype(np.float32)
            
        except Exception as e:
            print(f"❌ 특징 추출 실패: {e}")
            # 실패 시 더미 벡터 반환
            return np.random.rand(128).astype(np.float32)
    
    def get_vector_info(self) -> dict:
        """벡터 정보 반환"""
        return {
            "model_loaded": self.is_loaded,
            "vector_dimension": 128,
            "model_type": "SimCLR + ViT" if self.is_loaded else "Dummy",
            "device": str(self.device) if self.device else "CPU"
        }

# 전역 인스턴스 (싱글톤 패턴)
_feature_service = None

def get_feature_service() -> FeatureExtractionService:
    """특징 추출 서비스 인스턴스 반환"""
    global _feature_service
    if _feature_service is None:
        _feature_service = FeatureExtractionService()
    return _feature_service

router = APIRouter()

class ImageUrlRequest(BaseModel):
    imageUrl: str

@router.post("/api/extract_features_from_url")
@router.post("/api/extract_features_from_url/")
async def extract_features_from_url(req: ImageUrlRequest):
    service = get_feature_service()
    try:
        response = requests.get(req.imageUrl, timeout=10)
        response.raise_for_status()
        image_bytes = response.content
        vector = service.extract_features_from_bytes(image_bytes)
        return {"vector": vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 벡터 추출 실패: {e}")
