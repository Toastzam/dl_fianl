"""
이미지 서빙 관련 API 라우터
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services import ImageService

router = APIRouter(prefix="/api", tags=["images"])
image_service = ImageService()

@router.get("/image/{file_path:path}")
async def serve_image(file_path: str):
    """이미지 파일 서빙 (실제 + 더미)"""
    try:
        return image_service.serve_image(file_path)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 이미지 서빙 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
