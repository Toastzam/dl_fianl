"""
검색 관련 API 라우터
"""
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.dependencies import get_models
from app.services import SearchService
from app.models.requests import ImageUrlRequest

router = APIRouter(prefix="/api", tags=["search"])
search_service = SearchService()

@router.post("/upload_and_search/")
async def upload_and_search(file: UploadFile = File(...)):
    """실제 업로드 및 유사도 검색 API"""
    try:
        # 파일명 유효성 체크 및 정규화
        if not file.filename or not file.filename.strip():
            raise HTTPException(status_code=400, detail="업로드 파일명이 비어 있습니다.")
        
        from app.services.image_service import ImageService
        image_service = ImageService()
        normalized_filename = image_service.normalize_filename(file.filename)
        
        if not normalized_filename or normalized_filename in ['.', '..', '']:
            raise HTTPException(status_code=400, detail="정규화된 파일명이 비어 있습니다.")
        
        file_location = os.path.join(settings.UPLOAD_FOLDER, normalized_filename)
        
        # 파일 저장
        with open(file_location, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"📁 파일 저장 완료: {file_location}")
        
        # 모델 사용 가능 여부 확인
        models = get_models()
        
        if models['models_available'] and models['ap10k_model'] is not None:
            # 실제 모델 사용
            print("🚀 실제 모델 모드로 진입")
            result = await search_service.real_model_search(file_location, normalized_filename)
        else:
            # 더미 모드
            print("🔄 더미 모드로 진입")
            result = await search_service.dummy_search(file_location, normalized_filename)
        
        return JSONResponse(result)
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_with_db_mapping/")
async def search_with_db_mapping(file: UploadFile = File(...)):
    """유사도 검색 + DB 정보 매핑"""
    try:
        # 1. 기존 유사도 검색 수행
        search_response = await upload_and_search(file)
        
        # JSONResponse에서 데이터 추출
        import json
        if hasattr(search_response, 'body'):
            search_data = search_response.body.decode('utf-8')
            search_result = json.loads(search_data)
        else:
            search_result = search_response
        
        if not search_result.get('success'):
            return search_response
        
        # 2. 검색 결과를 실제 DB 정보로 매핑
        from app.services import DogService
        dog_service = DogService()
        mapped_results = dog_service.search_with_db_mapping(search_result.get('results', []))
        
        return JSONResponse({
            'success': True,
            'query_image': search_result.get('query_image'),
            'query_keypoint_image': search_result.get('query_keypoint_image'),
            'results': mapped_results,
            'mode': search_result.get('mode'),
            'total_found': len(mapped_results)
        })
        
    except Exception as e:
        print(f"❌ DB 매핑 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
