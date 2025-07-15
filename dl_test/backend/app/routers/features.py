"""
특징 추출 관련 API 라우터
"""
import requests
from urllib.parse import urlparse
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.dependencies import get_feature_service
from app.models.requests import ImageUrlRequest
from app.services.retrain_manager import trigger_retraining_if_needed

router = APIRouter(prefix="/api", tags=["features"])

@router.post("/extract_features/")
async def extract_features_api(file: UploadFile = File(...)):
    """이미지에서 특징 벡터만 추출해서 반환 (등록 시스템용 API)"""
    try:
        # 파일 내용 읽기
        file_content = await file.read()
        
        print(f"📁 벡터 추출 요청: {file.filename}")
        
        # 특징 벡터 추출
        feature_service = get_feature_service()
        vector = feature_service.extract_features_from_bytes(file_content)
        
        print(f"✅ 벡터 추출 완료: {vector.shape}")
        
        # 특징 추출 후 retraining 트리거 체크
        trigger_retraining_if_needed()
        
        return JSONResponse({
            "status": "success",
            "feature_vector": vector.tolist(),
            "vector_dimension": len(vector),
            "filename": file.filename,
            "model_info": feature_service.get_vector_info()
        })
        
    except Exception as e:
        print(f"❌ 벡터 추출 실패: {e}")
        return JSONResponse({
            "status": "error", 
            "message": str(e)
        }, status_code=500)

@router.post("/extract_features_from_url/")
async def extract_features_from_url_api(request: ImageUrlRequest):
    """이미지 URL에서 특징 벡터 추출해서 반환 (등록 시스템용 API)"""
    try:
        # 요청에서 이미지 URL 추출
        image_url = request.image_url
        if not image_url:
            return JSONResponse({
                "status": "error",
                "message": "image_url이 필요합니다"
            }, status_code=400)
        
        print(f"🌐 URL에서 벡터 추출 요청: {image_url}")
        
        # URL 유효성 검사
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return JSONResponse({
                "status": "error",
                "message": "유효하지 않은 URL입니다"
            }, status_code=400)
        
        # 이미지 다운로드
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Content-Type 확인 (이미지인지 검증)
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            return JSONResponse({
                "status": "error",
                "message": f"이미지가 아닌 파일입니다. Content-Type: {content_type}"
            }, status_code=400)
        
        print(f"📥 이미지 다운로드 완료: {len(response.content)} bytes")
        
        # 특징 벡터 추출
        feature_service = get_feature_service()
        vector = feature_service.extract_features_from_bytes(response.content)
        
        print(f"✅ 벡터 추출 완료: {vector.shape}")
        
        # 특징 추출 후 retraining 트리거 체크
        trigger_retraining_if_needed()
        
        return JSONResponse({
            "status": "success",
            "feature_vector": vector.tolist(),
            "vector_dimension": len(vector),
            "image_url": image_url,
            "image_size_bytes": len(response.content),
            "content_type": content_type,
            "model_info": feature_service.get_vector_info()
        })
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 이미지 다운로드 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"이미지 다운로드 실패: {str(e)}"
        }, status_code=400)
    except Exception as e:
        print(f"❌ 벡터 추출 실패: {e}")
        return JSONResponse({
            "status": "error", 
            "message": str(e)
        }, status_code=500)

@router.get("/feature_service_info/")
async def get_feature_service_info():
    """특징 추출 서비스 정보 반환"""
    try:
        feature_service = get_feature_service()
        info = feature_service.get_vector_info()
        return JSONResponse({
            "status": "success",
            "service_info": info
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)
