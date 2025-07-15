"""
FastAPI 애플리케이션 메인 파일 (리팩토링된 구조)
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.dependencies import lifespan, get_models
from app.routers import search_router, features_router, dogs_router, images_router

# FastAPI 앱 생성
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    redirect_slashes=False
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files 마운트
app.mount("/static", StaticFiles(directory=settings.STATIC_FOLDER), name="static")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_FOLDER), name="uploads")
app.mount("/training", StaticFiles(directory=os.path.abspath(os.path.join(settings.BACKEND_DIR, "..", "training"))), name="training")

# 라우터 등록
app.include_router(search_router)
app.include_router(features_router)
app.include_router(dogs_router)
app.include_router(images_router)

# feature_extraction_service의 FastAPI router 등록
from app.services.feature_extraction_service import router as feature_router
app.include_router(feature_router)

@app.get("/health")
async def health_check():
    """시스템 상태 체크"""
    models = get_models()
    
    status = {
        "status": "healthy",
        "models_available": models['models_available'],
        "ap10k_model_loaded": models['ap10k_model'] is not None,
        "mode": "real_model" if (models['models_available'] and models['ap10k_model'] is not None) else "dummy"
    }
    
    if models['models_available']:
        status["simclr_model_path"] = settings.SIMCLR_MODEL_PATH
        status["message"] = "실제 모델 사용 가능" if models['ap10k_model'] is not None else "모델 로드 대기 중"
    else:
        status["message"] = "더미 모드 - 모델 모듈 없음"
    
    return JSONResponse(status)

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Dog Similarity Search API", 
        "version": settings.VERSION,
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
